"""
Stage 1 - Prototype-based Regression training script for ESM-LLAMA modality alignment
on Esm2LlamaInstructForCausalLM model.

Instead of Contrastive Learning (InfoNCE), this script uses explicit Text Prototypes
pre-computed for each domain (e.g., IPR accession).

- ESM encoder: frozen
- LLaMA decoder: frozen
- Adapter: trainable (sequence -> llama hidden space alignment)
- Target: float32 text prototype vectors (one per domain)

DistributedDataParallel training script implemented from scratch (single-node multi-GPU).
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from typing import Any, Dict, Literal, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers import EsmModel, LlamaForCausalLM

from dataset.dataset_proto import Prot2TextInstructDatasetJSONL
from dataset.dataloader_proto import Prot2TextInstructDataLoaderJSONL
from models import (
    ModalityAdapter,
    ModalityAdapterConfig,
    Esm2LlamaInstructForCausalLM,
)
import scripts.utils_argparse as utils_argparse
import wandb


# ==============================
# Argument Parser
# ==============================
argParser = argparse.ArgumentParser()

# Paths
argParser.add_argument("--esm_path", type=str, required=True)
argParser.add_argument("--llama_path", type=str, required=True)
argParser.add_argument("--root_dataset_dir", type=str, required=True)
argParser.add_argument("--root_csv_dir", type=str, required=True)
argParser.add_argument("--save_checkpoint_dir", type=str, required=True)
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")

# ✅ Prototype float32 tensor file
argParser.add_argument(
    "--prototype_path",
    type=str,
    required=True,
    help="Path to domain_prototypes.pt (dict[str, Tensor], float32 recommended)"
)

# Train config
argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype, required=True)
argParser.add_argument("--batch_size_per_device", type=int, required=True)
argParser.add_argument("--num_epochs", type=int, required=True)
argParser.add_argument("--save_every_epochs", type=int, required=True)
argParser.add_argument("--gradient_accumulation_steps", type=int, required=True)
argParser.add_argument("--learning_rate", type=float, required=True)
argParser.add_argument("--gradient_clipping", type=float, default=None)
argParser.add_argument("--scheduler_gamma", type=float, required=True)
argParser.add_argument("--random_seed", type=int, required=True)

argParser.add_argument("--contrastive_num_segments", type=int, default=1)  # unused, for compatibility

# Splits
argParser.add_argument("--train_split", type=str, required=True)
argParser.add_argument("--eval_split", type=str, required=True)
argParser.add_argument("--debug_trim_train_split", type=int, default=None)
argParser.add_argument("--debug_trim_eval_split", type=int, default=None)


# ==============================
# Model Loader
# ==============================
def load_model(args: Dict[str, Any]) -> PreTrainedModel:
    """
    Load ESM encoder, LLaMA decoder, and modality adapter, then wrap into
    Esm2LlamaInstructForCausalLM.

    NOTE:
    - esm_encoder, llama_decoder are frozen
    - adapter is trainable
    - all modules initially loaded with args["torch_dtype"]
    """

    print(f"[Model] Loading ESM from: {args['esm_path']}")
    esm_encoder = EsmModel.from_pretrained(
        args["esm_path"],
        add_pooling_layer=False,
        torch_dtype=args["torch_dtype"],
        device_map="cpu",
    )

    print(f"[Model] Loading LLaMA from: {args['llama_path']}")
    llama_decoder = LlamaForCausalLM.from_pretrained(
        args["llama_path"],
        torch_dtype=args["torch_dtype"],
        device_map="cpu",
    )

    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)
    adapter.to(dtype=args["torch_dtype"])

    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        llama_decoder=llama_decoder,
    )

    if args["load_model_checkpoint_path"]:
        ckpt_path = args["load_model_checkpoint_path"]
        print(f"[Model] Loading checkpoint: {ckpt_path}")
        model_state_dict = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(model_state_dict)

    # Freeze ESM and LLaMA; only adapter is trainable
    model.esm_encoder.requires_grad_(False)
    model.llama_decoder.requires_grad_(False)

    return model

# ==============================
# 1. Readout 함수 (수학적 안전장치 강화)
# ==============================
def readout_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    readout_fn: Literal["last", "mean", "std", "mix"] = "mix",
) -> torch.Tensor:
    
    # 1. 계산은 무조건 float32로 수행 (안정성 확보)
    embeddings_f = embeddings.to(dtype=torch.float32)
    attn_f = attention_mask.to(dtype=torch.float32)

    # 0으로 나누기 방지 (분모가 0이 되는 것을 막음)
    sum_mask = attn_f.sum(dim=1, keepdim=True).clamp(min=1e-9)

    if readout_fn == "mean":
        masked_embeddings = embeddings_f * attn_f.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        return sum_embeddings / sum_mask

    elif readout_fn == "std":
        mean_embeddings = readout_embeddings(embeddings_f, attn_f, "mean")
        
        # 분산 계산
        diff = embeddings_f - mean_embeddings.unsqueeze(1)
        diff_sq = diff.pow(2)
        masked_diff_sq = diff_sq * attn_f.unsqueeze(-1)
        
        var = masked_diff_sq.sum(dim=1) / sum_mask
        
        # [핵심] sqrt 안에 음수나 0이 들어가는 것을 방지 (NaN 원천 차단)
        return torch.sqrt(var.clamp(min=1e-9))

    elif readout_fn == "mix":
        mean_e = readout_embeddings(embeddings_f, attn_f, "mean")
        std_e = readout_embeddings(embeddings_f, attn_f, "std")
        return torch.cat([mean_e, std_e], dim=1)

    return embeddings_f[:, 0, :]


# ==============================
# 2. Forward 함수 (입력 데이터 세척 추가)
# ==============================
def get_sequence_embeddings(
    model: Esm2LlamaInstructForCausalLM,
    sequence_input_ids: torch.Tensor,
    sequence_attention_mask: torch.Tensor,
) -> torch.Tensor:
    
    # 1. ESM Encoder Output
    with torch.no_grad():
        encoder_out = model.forward(
            protein_input_ids=sequence_input_ids,
            protein_attention_mask=sequence_attention_mask,
            return_encoder_outputs=True,
        )
        token_embeddings = encoder_out[0] # (B, L, H)

    # ======================================================
    # [긴급 처방] ESM 출력을 "세척"하여 Adapter 보호
    # ======================================================
    
    # 2. ESM 출력에 섞인 NaN/Inf를 0 또는 최대값으로 치환
    if torch.isnan(token_embeddings).any() or torch.isinf(token_embeddings).any():
        # 로그는 너무 많이 찍힐 수 있으므로 생략하거나 필요 시 추가
        token_embeddings = torch.nan_to_num(token_embeddings, nan=0.0, posinf=60000.0, neginf=-60000.0)

    # 3. 값이 너무 크면 bfloat16 범위 내로 자름 (Clamping)
    token_embeddings = torch.clamp(token_embeddings, min=-60000, max=60000)

    # 4. Adapter 실행 (이제 깨끗한 입력만 들어감)
    adapter_output = model.adapter(token_embeddings) 

    # 5. Adapter 출력 확인 (여기서 터지면 진짜 학습률 문제임)
    if torch.isnan(adapter_output).any():
        raise ValueError("[NaN TRACE] Adapter output contains NaN (Check Learning Rate!)")

    # 6. Readout 수행
    out = readout_embeddings(
        embeddings=adapter_output,
        attention_mask=sequence_attention_mask,
        readout_fn="mix",
    )

    return out


# ==============================
# Prototype-based Forward Pass
# ==============================
def teacher_forcing_forward_pass(
    rank: int,
    model: Union[DistributedDataParallel, FullyShardedDataParallel],
    data_batch: Dict[str, Any],
    prototypes: Dict[str, torch.Tensor],  # dict[accession/IPR, float32 Tensor]
) -> (torch.Tensor, float):
    """
    Prototype Regression Forward Pass:
    - For each sequence, find its prototype vector by accession (IPR),
    - Compute cosine similarity between sequence embedding and prototype,
    - Loss = 1 - mean(cosine_similarity).
    """
    device = torch.device(f"cuda:{rank}")

    protein_input_ids = data_batch["protein_input_ids"].to(device)
    protein_attention_mask = data_batch["protein_attention_mask"].to(device)
    accessions = data_batch["accession"]  # list[str], length = batch_size

    # 1. Collect valid targets (prototype exists)
    target_list = []
    valid_indices = []

    for i, acc in enumerate(accessions):
        if acc in prototypes:
            # Prototype is stored in float32 on CPU; move to device, keep float32
            target_list.append(prototypes[acc].to(device=device, dtype=torch.float32))
            valid_indices.append(i)

    if len(valid_indices) == 0:
        # No valid prototype for this batch → return zero-loss (still requires_grad=True)
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    target_embeddings = torch.stack(target_list, dim=0)  # (B_valid, H)

    # Filter protein_input_ids & mask to valid ones
    if len(valid_indices) < len(accessions):
        idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
        protein_input_ids = protein_input_ids.index_select(0, idx_tensor)
        protein_attention_mask = protein_attention_mask.index_select(0, idx_tensor)

    # 2. Get protein embeddings
    base_model = model.module if isinstance(model, (DistributedDataParallel, FullyShardedDataParallel)) else model
    protein_output = get_sequence_embeddings(
        base_model, protein_input_ids, protein_attention_mask
    )  # (B_valid, H_readout, float32)

    # 3. Cosine distance loss (in float32)
    protein_norm = torch.nn.functional.normalize(protein_output, p=2, dim=-1)  # (B_valid, H)
    target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=-1)  # (B_valid, H)

    # They are both float32 already
    cos_sim = (protein_norm * target_norm).sum(dim=-1)  # (B_valid,)
    loss = 1.0 - cos_sim.mean()  # scalar

    return loss, cos_sim.mean().item()


# ==============================
# DDP Setup / Cleanup
# ==============================
def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "9901")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# ==============================
# Train / Eval Loop
# ==============================
def train_epoch(
    rank: int,
    current_epoch: int,
    model: Union[DistributedDataParallel, FullyShardedDataParallel],
    dataloader: Prot2TextInstructDataLoaderJSONL,
    optimizer: Optimizer,
    prototypes: Dict[str, torch.Tensor],
    args: Dict[str, Any],
):
    device = torch.device(f"cuda:{rank}")
    model.train()

    ddp_loss = torch.zeros(2, device=device)      # [sum_loss, count]
    ddp_gradnorm = torch.zeros(2, device=device)  # [sum_gradnorm, count]

    optimizer.zero_grad(set_to_none=True)

    t = tqdm(iter(dataloader), disable=(rank != 0))
    for batch_idx, data_batch in enumerate(t):
        loss, cos_pos = teacher_forcing_forward_pass(
            rank=rank,
            model=model,
            data_batch=data_batch,
            prototypes=prototypes,
        )

        # Gradient Accumulation
        loss = loss / args["gradient_accumulation_steps"]

        if rank == 0:
            t.set_postfix({
                "mode": "train",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "loss": f"{loss.item() * args['gradient_accumulation_steps']:.4f}",
                "cos": f"{cos_pos:.3f}",
            })
            wandb.log({
                "train/step_loss": loss.item() * args["gradient_accumulation_steps"],
                "train/cos_sim": cos_pos,
            })

        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1.0

        loss.backward()

        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0:
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(float("inf") if args["gradient_clipping"] is None else args["gradient_clipping"]),
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1.0

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Reduce across ranks
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ddp_gradnorm, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_train_loss = (ddp_loss[0] / ddp_loss[1]).item() if ddp_loss[1] > 0 else 0.0
        avg_gradnorm = (ddp_gradnorm[0] / ddp_gradnorm[1]).item() if ddp_gradnorm[1] > 0 else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(f"[Epoch {current_epoch}] Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6e} | GradNorm: {avg_gradnorm:.4f}")

        wandb.log({
            "train/loss": avg_train_loss,
            "train/lr": current_lr,
            "train/gradnorm": avg_gradnorm,
            "epoch": int(current_epoch),
        })


def eval_epoch(
    rank: int,
    current_epoch: int,
    model: Union[DistributedDataParallel, FullyShardedDataParallel],
    dataloader: Prot2TextInstructDataLoaderJSONL,
    prototypes: Dict[str, torch.Tensor],
    args: Dict[str, Any],
):
    device = torch.device(f"cuda:{rank}")
    model.eval()

    # For simplicity: only rank 0 logs eval
    if rank == 0:
        sum_loss = 0.0
        sum_cos = 0.0
        cnt = 0

        t = tqdm(iter(dataloader))
        for data_batch in t:
            with torch.no_grad():
                loss, cos_pos = teacher_forcing_forward_pass(
                    rank=rank,
                    model=model,
                    data_batch=data_batch,
                    prototypes=prototypes,
                )

            sum_loss += loss.item()
            sum_cos += cos_pos
            cnt += 1

            t.set_postfix({
                "mode": "eval",
                "loss": f"{loss.item():.4f}",
                "cos": f"{cos_pos:.3f}",
            })

        avg_loss = sum_loss / max(cnt, 1)
        avg_cos = sum_cos / max(cnt, 1)

        print(f"[Epoch {current_epoch}] Eval Loss: {avg_loss:.4f} | Cos Sim: {avg_cos:.4f}")
        wandb.log({
            "eval/loss": avg_loss,
            "eval/cos_sim": avg_cos,
            "epoch": int(current_epoch),
        })
    else:
        # Other ranks just consume data to keep sampler in sync
        for _ in dataloader:
            pass


# ==============================
# Per-Process Training
# ==============================
def train_on_device(rank: int, world_size: int, args: Dict[str, Any]):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ------------------
    # Load Prototypes
    # ------------------
    if rank == 0:
        print(f"[Rank {rank}] Loading prototypes from: {args['prototype_path']}")
    prototypes_raw = torch.load(args["prototype_path"], map_location="cpu")
    # Ensure all prototypes are float32 tensors
    prototypes: Dict[str, torch.Tensor] = {}
    for key, value in prototypes_raw.items():
        if isinstance(value, torch.Tensor):
            prototypes[key] = value.to(torch.float32).contiguous()
        else:
            raise TypeError(f"Prototype for key {key} is not a Tensor")

    if rank == 0:
        print(f"[Rank {rank}] Loaded {len(prototypes)} prototypes (float32).")

    try:
        # ------------------
        # W&B Init
        # ------------------
        if rank == 0:
            wandb.init(
                project="graduate",
                config=args,
                name=f"proto_reg_{args['train_split']}",
            )

        # ------------------
        # Tokenizers
        # ------------------
        esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
        llama_tokenizer = AutoTokenizer.from_pretrained(
            args["llama_path"],
            pad_token="<|reserved_special_token_0|>",
        )

        # ------------------
        # Datasets
        # ------------------
        train_dataset = Prot2TextInstructDatasetJSONL(
            root_dir=os.path.join(args["root_dataset_dir"], f"{args['train_split']}"),
            jsonl_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.jsonl"),
            sequence_tokenizer=esm_tokenizer,
            description_tokenizer=llama_tokenizer,
            use_cache=True,
            save_cache=False,
        )
        if args["debug_trim_train_split"]:
            train_dataset.uniprot_df = train_dataset.uniprot_df.iloc[: args["debug_trim_train_split"]]

        eval_dataset = Prot2TextInstructDatasetJSONL(
            root_dir=os.path.join(args["root_dataset_dir"], f"{args['eval_split']}"),
            jsonl_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.jsonl"),
            sequence_tokenizer=esm_tokenizer,
            description_tokenizer=llama_tokenizer,
            use_cache=True,
            save_cache=False,
        )
        if args["debug_trim_eval_split"]:
            eval_dataset.uniprot_df = eval_dataset.uniprot_df.iloc[: args["debug_trim_eval_split"]]

        # ------------------
        # Samplers & Dataloaders
        # ------------------
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        train_loader = Prot2TextInstructDataLoaderJSONL(
            train_dataset,
            batch_size=args["batch_size_per_device"],
            sampler=train_sampler,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        eval_loader = Prot2TextInstructDataLoaderJSONL(
            eval_dataset,
            batch_size=args["batch_size_per_device"],
            sampler=eval_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # ------------------
        # Model / DDP
        # ------------------
        model = load_model(args=args).to(device)
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

        # ------------------
        # Optimizer / Scheduler
        # ------------------
        optimizer = Adam(model.parameters(), lr=args["learning_rate"])
        scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])

        # Load optim/sched state if provided
        if args["load_optimizer_scheduler_checkpoint_path"]:
            opt_ckpt = torch.load(args["load_optimizer_scheduler_checkpoint_path"], map_location="cpu")
            optimizer.load_state_dict(opt_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(opt_ckpt["scheduler_state_dict"])
            if rank == 0:
                print(f"[Rank 0] Loaded optimizer/scheduler state from {args['load_optimizer_scheduler_checkpoint_path']}")

        # ------------------
        # Epoch Loop
        # ------------------
        for epoch_idx in range(1, args["num_epochs"] + 1):
            train_sampler.set_epoch(epoch_idx)

            train_epoch(
                rank=rank,
                current_epoch=epoch_idx,
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                prototypes=prototypes,
                args=args,
            )
            scheduler.step()
            dist.barrier()

            eval_epoch(
                rank=rank,
                current_epoch=epoch_idx,
                model=model,
                dataloader=eval_loader,
                prototypes=prototypes,
                args=args,
            )
            dist.barrier()

            # Save checkpoint (model only)
            if rank == 0 and (
                epoch_idx == 1
                or epoch_idx == args["num_epochs"]
                or epoch_idx % args["save_every_epochs"] == 0
            ):
                ckpt_path = os.path.join(
                    args["save_checkpoint_dir"],
                    f"model_checkpoint_{epoch_idx}.pt",
                )
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"[Rank 0] Saved model checkpoint: {ckpt_path}")

            dist.barrier()

    finally:
        if rank == 0:
            try:
                wandb.finish()
            except Exception:
                pass
        cleanup()


def train_distributed(args: Dict[str, Any]):
    world_size = args["world_size"]
    mp.spawn(
        train_on_device,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    parsed_args.world_size = torch.cuda.device_count()

    # Seed
    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)

    # Timestamped checkpoint dir
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(
        parsed_args.save_checkpoint_dir,
        f"checkpoints_{timestamp}",
    )
    os.makedirs(parsed_args.save_checkpoint_dir, exist_ok=True)

    print("####################")
    for key, value in parsed_args.__dict__.items():
        print(f"{key}: {value}")
    print("####################")

    train_distributed(parsed_args.__dict__)
