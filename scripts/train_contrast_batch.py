# file: scripts/train_stage1_contrastive_ddp_accum.py
"""
Stage 1 - contrastive learning training script for ESM-LLAMA modality alignment
on Esm2LlamaInstructForCausalLM model.

- No LoRA.
- Single-node Multi-GPU DDP from scratch.
- Supports: gradient accumulation (DDP no_sync), AMP (bf16/fp16/none),
  MoCo-style negative queue, inter-epoch evaluation, saving/loading optimizer+scheduler.
- Not supported: FSDP generation, gradient checkpointing, full pretrained save/load.
- On clusters: print -> stdout, tqdm -> stderr.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import math
from typing import Any, Dict, Literal, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers import EsmModel, LlamaModel, LlamaForCausalLM

from dataset.dataset_jsonl import Prot2TextInstructDatasetJSONL
from dataset.dataloader_jsonl import Prot2TextInstructDataLoaderJSONL
from models import (ModalityAdapter, ModalityAdapterConfig, Esm2LlamaInstructForCausalLM)
import scripts.utils_argparse as utils_argparse
import wandb


# ---------------------------
# Arguments
# ---------------------------
argParser = argparse.ArgumentParser()

# model / data
argParser.add_argument("--esm_path", type=str, required=True)
argParser.add_argument("--llama_path", type=str, required=True)
argParser.add_argument("--root_dataset_dir", type=str, required=True)
argParser.add_argument("--root_csv_dir", type=str, required=True)
argParser.add_argument("--save_checkpoint_dir", type=str, required=True)
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")

# training
argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype, required=True)
argParser.add_argument("--batch_size_per_device", type=int, required=True)
argParser.add_argument("--num_epochs", type=int, required=True)
argParser.add_argument("--save_every_epochs", type=int, required=True)
argParser.add_argument("--gradient_accumulation_steps", type=int, default=1)
argParser.add_argument("--target_global_batch_size", type=int, default=0)
argParser.add_argument("--learning_rate", type=float, required=True)
argParser.add_argument("--gradient_clipping", type=float, default=None)
argParser.add_argument("--scheduler_gamma", type=float, required=True)
argParser.add_argument("--random_seed", type=int, required=True)
argParser.add_argument("--contrastive_num_segments", type=int, required=True)

# splits
argParser.add_argument("--train_split", type=str, required=True)
argParser.add_argument("--eval_split", type=str, required=True)
argParser.add_argument("--debug_trim_train_split", type=int, default=None)
argParser.add_argument("--debug_trim_eval_split", type=int, default=None)

# AMP
argParser.add_argument("--amp_dtype", type=str, choices=["none", "bf16", "fp16"], default="bf16")

# MoCo queue
argParser.add_argument("--moco_queue_size", type=int, default=4096)         # K
argParser.add_argument("--moco_use_in_eval", action="store_true", default=False)  # eval uses queue as extra negatives


# ---------------------------
# DDP helpers
# ---------------------------
def all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=0)


# ---------------------------
# Losses
# ---------------------------
class BatchInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, batch_output1: torch.Tensor, batch_output2: torch.Tensor):
        logits = torch.mm(batch_output1, batch_output2.t()) / self.temperature
        numerator = torch.exp(torch.diag(logits)).unsqueeze(1)
        denominator = torch.sum(torch.exp(logits), dim=1, keepdim=True)
        return - torch.log(numerator / denominator).mean()


class SegmentedBatchInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, segment_output1: torch.Tensor, batch_output2: torch.Tensor, labels: torch.Tensor):
        segment_size = segment_output1.size(0)
        logits = torch.mm(segment_output1, batch_output2.t()) / self.temperature
        numerator = torch.exp(logits[torch.arange(segment_size), labels]).unsqueeze(1)
        denominator = torch.sum(torch.exp(logits), dim=1, keepdim=True)
        return - torch.log(numerator / denominator).mean()


# ---------------------------
# MoCo-style queue
# ---------------------------
class MoCoQueue(torch.nn.Module):
    """
    FIFO queue of previous-step keys (no grad). Every rank maintains an identical
    copy by always enqueuing the *all_gathered* keys.
    """
    def __init__(self, dim: int, K: int, device: Union[int, torch.device]):
        super().__init__()
        K = int(K)
        self.register_buffer("queue", torch.randn(K, dim, device=device))
        self.queue = torch.nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long, device=device))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        keys = keys.detach()
        K = self.queue.shape[0]
        n = int(keys.shape[0])
        if n == 0:
            return
        ptr = int(self.ptr.item())

        if ptr + n <= K:
            self.queue[ptr:ptr+n] = keys
            ptr = (ptr + n) % K
        else:
            first = K - ptr
            self.queue[ptr:] = keys[:first]
            remaining = n - first
            self.queue[:remaining] = keys[first:first+remaining]
            ptr = remaining % K

        self.ptr[0] = ptr  # why: keep all ranks consistent

    @torch.no_grad()
    def get(self) -> torch.Tensor:
        return self.queue


# ---------------------------
# Model I/O
# ---------------------------
def load_model(args: Dict[str, Any]) -> PreTrainedModel:
    esm_encoder = EsmModel.from_pretrained(
        args["esm_path"], add_pooling_layer=False, torch_dtype=args["torch_dtype"], device_map="cpu"
    )
    llama_decoder = LlamaForCausalLM.from_pretrained(
        args["llama_path"], torch_dtype=args["torch_dtype"], device_map="cpu"
    )
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size, intermediate_dim=2048, output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config).to(args["torch_dtype"])
    model = Esm2LlamaInstructForCausalLM(esm_encoder=esm_encoder, adapter=adapter, llama_decoder=llama_decoder)

    if args["load_model_checkpoint_path"]:
        print(f"Loading {args['load_model_checkpoint_path']}")
        model_state_dict = torch.load(args["load_model_checkpoint_path"], weights_only=True, map_location="cpu")
        model.load_state_dict(model_state_dict)

    # freeze base enc/dec (only adapter trains)
    model.esm_encoder.requires_grad_(False)
    model.llama_decoder.requires_grad_(False)
    return model


# ---------------------------
# Embedding utilities
# ---------------------------
def readout_embeddings(embeddings: torch.Tensor, attention_mask: torch.Tensor,
                       readout_fn: Literal["last", "mean", "std", "mix"]) -> torch.Tensor:
    embeddings = embeddings.float()
    attention_mask = attention_mask.to(dtype=torch.float32)

    if readout_fn == "last":
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        return embeddings[batch_indices, last_token_indices, :]
    elif readout_fn == "mean":
        masked = embeddings * attention_mask.unsqueeze(-1)
        s = masked.sum(dim=1)
        c = attention_mask.sum(dim=1, keepdim=True)
        return s / c
    elif readout_fn == "std":
        mean = readout_embeddings(embeddings, attention_mask, "mean")
        diff2 = (embeddings - mean.unsqueeze(1)).pow(2) * attention_mask.unsqueeze(-1)
        s = diff2.sum(dim=1)
        c = attention_mask.sum(dim=1, keepdim=True)
        return (s / c).sqrt()
    elif readout_fn == "mix":
        mean = readout_embeddings(embeddings, attention_mask, "mean")
        std = readout_embeddings(embeddings, attention_mask, "std")
        return torch.cat([mean, std], dim=1)
    else:
        raise ValueError(f"Unknown readout_fn: {readout_fn}")


def get_sequence_embeddings(model: Esm2LlamaInstructForCausalLM,
                            sequence_input_ids: torch.Tensor,
                            sequence_attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        encoder_output = model.forward(
            protein_input_ids=sequence_input_ids,
            protein_attention_mask=sequence_attention_mask,
            return_encoder_outputs=True,
        )
        enc_last = encoder_output[0]
        if torch.isnan(enc_last).any() or torch.isinf(enc_last).any():
            raise ValueError("[NaN TRACE] ESM encoder output has NaN/Inf")

    adapter_output = model.adapter(encoder_output[0])  # keep FP32 for stability
    if torch.isnan(adapter_output).any() or torch.isinf(adapter_output).any():
        raise ValueError("[NaN TRACE] Adapter output has NaN/Inf")

    out = readout_embeddings(adapter_output, sequence_attention_mask, readout_fn="mix")
    lengths = sequence_attention_mask.sum(dim=1, keepdim=True)
    if (lengths == 0).any():
        raise ValueError("[MASK ERROR] zero-length sequence found in readout")
    return out


def get_description_embeddings(model: Esm2LlamaInstructForCausalLM,
                               description_input_ids: torch.Tensor,
                               description_attention_mask: torch.Tensor,
                               output_llama_layer: int = 16) -> torch.Tensor:
    llama_model: LlamaModel = model.llama_decoder.model
    hidden_states = llama_model(
        input_ids=description_input_ids,
        attention_mask=description_attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=False,
    )[1]
    layer_out = hidden_states[output_llama_layer].float()
    return readout_embeddings(layer_out, description_attention_mask, readout_fn="mix")


# ---------------------------
# Train/Eval forward (with MoCo queue)
# ---------------------------
def _debug_check_masks_and_ids(name, ids, mask, vocab_size, rank):
    if torch.isnan(ids).any():
        raise ValueError(f"[NaN TRACE] {name}_input_ids has NaN (rank {rank})")
    if torch.isnan(mask.float()).any():
        raise ValueError(f"[NaN TRACE] {name}_attention_mask has NaN (rank {rank})")
    if ((mask != 0) & (mask != 1)).any():
        bad = ((mask != 0) & (mask != 1)).nonzero(as_tuple=True)
        print(f"[MASK WARN] {name}_attention_mask not in {{0,1}} (rank {rank}), example:",
              mask[bad[0][0]].tolist()[:64])
    lengths = mask.sum(dim=1)
    if (lengths == 0).any():
        bad_rows = (lengths == 0).nonzero(as_tuple=True)[0]
        raise ValueError(f"[MASK ERROR] {name} all-padding rows exist (rank {rank}), idx {bad_rows[:5].tolist()}")
    if (ids < 0).any() or (ids >= vocab_size).any():
        bad_low = (ids < 0).nonzero(as_tuple=True)[0][:3].tolist()
        bad_high = (ids >= vocab_size).nonzero(as_tuple=True)[0][:3].tolist()
        raise ValueError(f"[ID RANGE] {name}_input_ids out of [0,{vocab_size-1}] (rank {rank}), low {bad_low}, high {bad_high}")


def teacher_forcing_forward_pass(rank: int,
                                 model: Union[DDP, FullyShardedDataParallel],
                                 data_batch: Dict[str, Any],
                                 contrastive_num_segments: int,
                                 queue: MoCoQueue,
                                 do_enqueue: bool):
    protein_input_ids = data_batch["protein_input_ids"].to(rank)
    protein_attention_mask = data_batch["protein_attention_mask"].to(rank)
    description_input_ids = data_batch["description_input_ids"].to(rank)
    description_attention_mask = data_batch["description_attention_mask"].to(rank)

    base_model = model.module if isinstance(model, DDP) else model
    _debug_check_masks_and_ids("protein", protein_input_ids, protein_attention_mask,
                               base_model.esm_encoder.config.vocab_size, rank)
    _debug_check_masks_and_ids("description", description_input_ids, description_attention_mask,
                               base_model.llama_decoder.config.vocab_size, rank)

    bsz_local = protein_input_ids.size(0)
    segment_size = bsz_local // contrastive_num_segments
    if segment_size * contrastive_num_segments != bsz_local and rank == 0:
        print("WARNING: batch size not divisible by contrastive_num_segments.")

    acc_loss = torch.zeros([], device=protein_input_ids.device)
    loss_fn = SegmentedBatchInfoNCELoss()

    with torch.no_grad():
        desc_out = get_description_embeddings(base_model, description_input_ids, description_attention_mask)
        desc_out = torch.nn.functional.normalize(desc_out, p=2, dim=-1)

        # gather across ranks (positives live here)
        keys_global = all_gather_no_grad(desc_out)  # [global_bsz, dim]

        # get queue negatives
        # get queue negatives (optional)
        neg_bank = None
        if queue is not None:
            qb = queue.get()
            if qb is not None and qb.numel() > 0:
                neg_bank = qb

        if neg_bank is not None:
            keys_all = torch.cat([keys_global, neg_bank], dim=0)
        else:
            keys_all = keys_global

    cos_pos_debug = 0.0
    global_bsz = keys_global.shape[0]
    world_size = dist.get_world_size()
    rank_offset = rank * bsz_local  # index of this rank's local batch in the gathered keys

    for segment_id in range(contrastive_num_segments):
        s_ids = protein_input_ids[segment_id * segment_size:(segment_id + 1) * segment_size]
        s_mask = protein_attention_mask[segment_id * segment_size:(segment_id + 1) * segment_size]

        seg_q = get_sequence_embeddings(base_model, s_ids, s_mask)  # queries

        if rank == 0 and segment_id == 0:
            with torch.no_grad():
                p_norm = seg_q.norm(p=2, dim=-1).mean().item()
                d_norm = keys_all.norm(p=2, dim=-1).mean().item()
                cos_pos_debug = torch.mm(seg_q, keys_global.t()).diag().mean().item()
                print(f"[DEBUG] step check | protein_norm={p_norm:.3f} | keys_norm={d_norm:.3f} | cos_pos={cos_pos_debug:.3f}")

        # positives are within first 'global_bsz' entries
        labels = torch.arange(rank_offset + segment_id * segment_size,
                              rank_offset + (segment_id + 1) * segment_size,
                              device=rank)

        acc_loss += loss_fn(segment_output1=seg_q, batch_output2=keys_all, labels=labels)

    # enqueue keys AFTER using them (avoid using current batch as queue negatives)
    if do_enqueue:
        with torch.no_grad():
            queue.enqueue(keys_global)

    return acc_loss / contrastive_num_segments, cos_pos_debug


# ---------------------------
# DDP setup/teardown
# ---------------------------
def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
# ---- print effective contrastive batch size (once) ----

# ---------------------------
# Epoch loops
# ---------------------------
def train_epoch(rank: int,
                current_epoch: int,
                model: Union[DDP, FullyShardedDataParallel],
                dataloader: Prot2TextInstructDataLoaderJSONL,
                optimizer: Optimizer,
                queue: MoCoQueue,
                args: Dict[str, Any]) -> None:
    model.train()

    use_fp16 = args["amp_dtype"] == "fp16"
    use_bf16 = args["amp_dtype"] == "bf16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16, init_scale=2**16)

    ddp_loss = torch.zeros(2, device=rank)      # [sum_loss, num_batches]
    ddp_gradnorm = torch.zeros(2, device=rank)  # [sum_gradnorm, num_steps]
    optimizer.zero_grad(set_to_none=True)

    accum_steps = int(args["gradient_accumulation_steps"])
    assert accum_steps >= 1

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):
        microstep = batch_idx % accum_steps
        is_sync_step = (microstep == accum_steps - 1)

        autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
        context = (torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else torch.cuda.amp.autocast(enabled=False))

        ddp_context = model.no_sync() if (not is_sync_step) else torch.enable_grad()
        with ddp_context:
            with context:
                loss, cos_pos_debug = teacher_forcing_forward_pass(
                    rank=rank,
                    model=model,
                    data_batch=data_batch,
                    contrastive_num_segments=args["contrastive_num_segments"],
                    queue=queue,
                    do_enqueue=is_sync_step  # why: enqueue once per optimizer step
                )
                loss = loss / accum_steps

            t.set_postfix({
                "mode": "train",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "batch_loss": float(loss.item() * accum_steps),
                "cos_pos": cos_pos_debug,
                "device": f"rank:{rank}",
                "accum": f"{microstep+1}/{accum_steps}",
                "queueK": queue.queue.shape[0],
            })

            if rank == 0:
                wandb.log({
                    "train/step_loss": loss.item() * accum_steps,
                    "train/cos_pos": cos_pos_debug,
                    "train/microstep": microstep + 1,
                    "train/queue_ptr": int(queue.ptr.item()),
                })

            ddp_loss[0] += loss.item() * accum_steps
            ddp_loss[1] += 1

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if is_sync_step:
            if args["gradient_clipping"] is None:
                gradnorm = torch.tensor(0.0, device=rank)
            else:
                if use_fp16:
                    scaler.unscale_(optimizer)
                gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args["gradient_clipping"])
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            if use_fp16:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ddp_gradnorm, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_train_loss = (ddp_loss[0] / ddp_loss[1]).item()
        denom = ddp_gradnorm[1] if ddp_gradnorm[1] > 0 else torch.tensor(1., device=ddp_gradnorm.device)
        avg_gradnorm = (ddp_gradnorm[0] / denom).item()
        current_lr = float(optimizer.param_groups[0]['lr'])
        print(f"[epoch={current_epoch}/{args['num_epochs']}, train_loss={avg_train_loss}, epoch_lr={current_lr}, epoch_gradnorm={avg_gradnorm}]")
        wandb.log({"train/loss": avg_train_loss, "train/lr": current_lr, "train/gradnorm": avg_gradnorm, "epoch": int(current_epoch)})
        if ddp_loss[0] != ddp_loss[0]:
            raise ValueError("NaN detected in the training loss of the epoch, training interrupted.")


def eval_epoch(rank: int,
               current_epoch: int,
               model: Union[DDP, FullyShardedDataParallel],
               dataloader: Prot2TextInstructDataLoaderJSONL,
               queue: MoCoQueue,
               args: Dict[str, Any]) -> None:
    model.eval()
    ddp_loss = torch.zeros(2, device=rank)

    t = tqdm(iter(dataloader))
    with torch.no_grad():
        for data_batch in t:
            eval_queue = queue if args["moco_use_in_eval"] else None

            loss, cos_pos_debug = teacher_forcing_forward_pass(
                rank=rank,
                model=model,
                data_batch=data_batch,
                contrastive_num_segments=args["contrastive_num_segments"],
                queue=eval_queue,
                do_enqueue=False
            )

            t.set_postfix({
                "mode": "eval",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "batch_loss": float(loss.item()),
                "device": f"rank:{rank}"
            })
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        avg_eval_loss = (ddp_loss[0] / ddp_loss[1]).item()
        print(f"[epoch={current_epoch}/{args['num_epochs']}, eval_loss={avg_eval_loss}]")
        wandb.log({"eval/loss": avg_eval_loss, "eval/cos_pos": cos_pos_debug, "epoch": current_epoch})


# ---------------------------
# Per-device worker
# ---------------------------
def train_on_device(rank: int, world_size: int, args: Dict[str, Any]):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    try:
        if args.get("target_global_batch_size", 0):
            denom = args["batch_size_per_device"] * world_size
            accum = math.ceil(args["target_global_batch_size"] / max(1, denom))
            if rank == 0:
                print(f"[accum] target_global_batch_size={args['target_global_batch_size']}, "
                      f"per_device={args['batch_size_per_device']}, world_size={world_size} -> accum={accum}")
            args["gradient_accumulation_steps"] = int(accum)

        effective_global_bsz = args["batch_size_per_device"] * world_size * args["gradient_accumulation_steps"]
        if rank == 0:
            print(f"[effective_global_batch_size]={effective_global_bsz} "
                  f"(per_device={args['batch_size_per_device']}, world_size={world_size}, accum={args['gradient_accumulation_steps']})")

        if rank == 0:
            wandb.init(project="graduate", config=args, name=f"run_{args['train_split']}_{args['num_epochs']}epochs")

        # tokenizers & datasets
        esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
        llama_tokenizer = AutoTokenizer.from_pretrained(args["llama_path"], pad_token='<|reserved_special_token_0|>')

        train_dataset = Prot2TextInstructDatasetJSONL(
            root_dir=os.path.join(args["root_dataset_dir"], f"{args['train_split']}"),
            jsonl_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.jsonl"),
            sequence_tokenizer=esm_tokenizer,
            description_tokenizer=llama_tokenizer,
            use_cache=True, save_cache=True,
        )
        if args["debug_trim_train_split"]:
            train_dataset.usable_file_names = train_dataset.usable_file_names[:args["debug_trim_train_split"]]

        eval_dataset = Prot2TextInstructDatasetJSONL(
            root_dir=os.path.join(args["root_dataset_dir"], f"{args['eval_split']}"),
            jsonl_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.jsonl"),
            sequence_tokenizer=esm_tokenizer,
            description_tokenizer=llama_tokenizer,
            use_cache=True, save_cache=True,
        )
        if args["debug_trim_eval_split"]:
            eval_dataset.usable_file_names = eval_dataset.usable_file_names[:args["debug_trim_eval_split"]]

        # samplers / loaders
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size, shuffle=False)

        train_loader = Prot2TextInstructDataLoaderJSONL(
            train_dataset, batch_size=args["batch_size_per_device"], sampler=train_sampler,
            shuffle=False, num_workers=8, pin_memory=True, drop_last=True,
        )
        print(f"Train dataset loaded on rank:{rank}")

        eval_loader = Prot2TextInstructDataLoaderJSONL(
            eval_dataset, batch_size=args["batch_size_per_device"], sampler=eval_sampler,
            shuffle=False, num_workers=4, pin_memory=True, drop_last=True,
        )
        print(f"Eval dataset loaded on rank:{rank}")

        # model / queue / optim
        model = load_model(args=args).to(rank)
        model = DDP(model, find_unused_parameters=True)
        print(f"DDP model loaded on rank:{rank}")

        # description embedding dim = 2 * llama_hidden (mix: mean+std)
        desc_dim = model.module.llama_decoder.config.hidden_size * 2
        queue = MoCoQueue(dim=desc_dim, K=args["moco_queue_size"], device=rank)

        # keep queue identical across ranks (seed broadcast)
        with torch.no_grad():
            dist.broadcast(queue.queue, src=0)
            dist.broadcast(queue.ptr, src=0)
        if rank == 0:
            world_size = dist.get_world_size()
            local_bsz = args["batch_size_per_device"]
            global_bsz = local_bsz * world_size
            queue_K = queue.queue.shape[0]

            print("========== Contrastive Setup ==========")
            print(f"local batch per GPU      : {local_bsz}")
            print(f"world size (GPUs)        : {world_size}")
            print(f"global batch (positives) : {global_bsz}")
            print(f"MoCo queue size (neg)    : {queue_K}")
            print(f"total keys in InfoNCE    : {global_bsz + queue_K}")
            print("=======================================")




        optimizer = Adam(model.parameters(), lr=args["learning_rate"])
        scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])

        if args["load_optimizer_scheduler_checkpoint_path"]:
            print(f"Loading {args['load_optimizer_scheduler_checkpoint_path']}")
            checkpoint_state_dicts = torch.load(
                args["load_optimizer_scheduler_checkpoint_path"], weights_only=True, map_location="cpu"
            )
            optimizer.load_state_dict(checkpoint_state_dicts["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint_state_dicts["scheduler_state_dict"])

        # epochs
        for epoch_idx in range(1, args["num_epochs"] + 1):
            train_sampler.set_epoch(epoch=epoch_idx)

            train_epoch(rank=rank, current_epoch=epoch_idx, model=model,
                        dataloader=train_loader, optimizer=optimizer, queue=queue, args=args)
            scheduler.step()
            dist.barrier()

            eval_epoch(rank=rank, model=model, current_epoch=epoch_idx,
                       dataloader=eval_loader, queue=queue, args=args)
            dist.barrier()

            # checkpoints
            if (epoch_idx == 1 or epoch_idx == args["num_epochs"] or epoch_idx % args["save_every_epochs"] == 0):
                model_state_dict = model.module.state_dict()
                if rank == 0:
                    model_ckpt = os.path.join(args["save_checkpoint_dir"], f"model_checkpoint_{epoch_idx}.pt")
                    torch.save(model_state_dict, model_ckpt)
                    print(f"Saving {model_ckpt}")

                    opt_sched_ckpt = os.path.join(args["save_checkpoint_dir"], f"optimizer_scheduler_checkpoint_{epoch_idx}.pt")
                    torch.save({"optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict()}, opt_sched_ckpt)
                    print(f"Saving {opt_sched_ckpt}")
                dist.barrier()

    finally:
        if rank == 0:
            try:
                wandb.finish()
            except Exception:
                pass
        cleanup()


# ---------------------------
# Launcher
# ---------------------------
def train_distributed(args: Dict[str, Any]):
    torch.multiprocessing.spawn(train_on_device, args=(args["world_size"], args), nprocs=args["world_size"], join=True)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    parsed_args.world_size = torch.cuda.device_count()

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(parsed_args.save_checkpoint_dir, f"checkpoints_{timestamp}")
    os.makedirs(parsed_args.save_checkpoint_dir, exist_ok=True)

    # compute accumulation from target global batch if requested
    if parsed_args.target_global_batch_size and parsed_args.target_global_batch_size > 0:
        denom = max(1, parsed_args.batch_size_per_device * max(1, parsed_args.world_size))
        accum = math.ceil(parsed_args.target_global_batch_size / denom)
        parsed_args.gradient_accumulation_steps = accum

    print("####################")
    for k, v in parsed_args.__dict__.items():
        print(f"{k}: {v}")
    print("####################")

    train_distributed(parsed_args.__dict__)
