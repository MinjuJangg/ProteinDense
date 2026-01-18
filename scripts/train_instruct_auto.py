"""
Stage 2 - instruction tuning training script for ESM-LLAMA protein description
generation on Esm2LlamaInstructForCausalLM model.

Single-process, NO DDP.
Model shards are placed across multiple GPUs via Hugging Face `device_map="auto"`.

Supports:
- LoRA (load or initialize)
- Gradient Accumulation
- Optional AMP (default: off; use float32 by default as requested)
- Inter-epoch evaluation
- OOM-safe training (skip batch on CUDA OOM)
- W&B logging
- Checkpoint saving (LoRA adapter + optimizer/scheduler states)

Run example:
  CUDA_VISIBLE_DEVICES=0,1 python scripts/train_instruct_device_map.py \
    --esm_path /path/to/esm2_t36_3B_UR50D \
    --llama_path /path/to/Meta-Llama-3.1-8B-Instruct-hf \
    --root_csv_dir /data/jsonl \
    --save_checkpoint_dir /ckpts/stage2 \
    --batch_size_per_device 1 \
    --num_epochs 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --scheduler_gamma 0.95 \
    --train_split train \
    --eval_split eval
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, EsmModel, LlamaForCausalLM

from peft import get_peft_model, LoraConfig
from peft.peft_model import PeftModel

from dataset.dataset_light_jsonl_dense import Prot2TextLightDataset, Prot2TextLightCollater
from models import (
    ModalityAdapter,
    ModalityAdapterConfig,
    Esm2LlamaInstructForCausalLM,
)
import scripts.utils_argparse as utils_argparse
import wandb
from torch.nn.functional import cosine_similarity


# ---------------------------
# Argparse
# ---------------------------
argParser = argparse.ArgumentParser()

# paths
argParser.add_argument("--esm_path", type=str, required=True)
argParser.add_argument("--llama_path", type=str, required=True)
argParser.add_argument("--root_csv_dir", type=str, required=True)
argParser.add_argument("--save_checkpoint_dir", type=str, required=True)
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_adapter_checkpoint_dir", type=str, default="")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")

# training
argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype, default="float32")
argParser.add_argument("--batch_size_per_device", type=int, required=True)
argParser.add_argument("--num_epochs", type=int, required=True)
argParser.add_argument("--save_every_epochs", type=int, default=1)
argParser.add_argument("--gradient_accumulation_steps", type=int, required=True)
argParser.add_argument("--learning_rate", type=float, required=True)
argParser.add_argument("--gradient_clipping", type=float, default=1.0)
argParser.add_argument("--scheduler_gamma", type=float, required=True)
argParser.add_argument("--random_seed", type=int, default=42)
argParser.add_argument("--lora_rank", type=int, default=16)
argParser.add_argument("--fix_modality_adapter", type=utils_argparse.str2bool, default=False)

# data options
argParser.add_argument("--include_text_fields", type=utils_argparse.str2bool, default=True)
argParser.add_argument("--name_dropout", type=float, default=0.0)
argParser.add_argument("--taxonomy_dropout", type=float, default=0.0)
argParser.add_argument("--train_split", type=str, required=True)
argParser.add_argument("--eval_split", type=str, required=True)
argParser.add_argument("--debug_trim_train_split", type=int, default=None)
argParser.add_argument("--debug_trim_eval_split", type=int, default=None)

# stability/perf
argParser.add_argument("--amp", type=utils_argparse.str2bool, default=False)  # default off (float32)
argParser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
argParser.add_argument("--workers", type=int, default=4)
argParser.add_argument("--pin_memory", type=utils_argparse.str2bool, default=True)
argParser.add_argument("--persistent_workers", type=utils_argparse.str2bool, default=True)

# logging
argParser.add_argument("--wandb_project", type=str, default="Prot2Text-Stage2")
argParser.add_argument("--wandb_run_name", type=str, default="")
argParser.add_argument("--log_cosine_every", type=int, default=50)  # steps; 0 to disable


# ---------------------------
# Utilities
# ---------------------------
def first_device() -> torch.device:
    if torch.cuda.is_available():
        # With device_map="auto", inputs should be on cuda:0 typically.
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------
# Model loader (device_map="auto")
# ---------------------------
# def load_model(args: Dict[str, Any]) -> PeftModel:
#     """
#     Build ESM encoder + LLaMA decoder + ModalityAdapter, then wrap with LoRA.
#     All base models are loaded with device_map='auto' to shard across visible GPUs.
#     """
#     # base dtype derived from arg
#     base_dtype = args["torch_dtype"]

#     # 1) encoders/decoders on multiple GPUs
#     esm_encoder = EsmModel.from_pretrained(
#         args["esm_path"],
#         add_pooling_layer=False,
#         torch_dtype=base_dtype,
#         device_map="auto",   # <<< shard across GPUs
#     )
#     llama_decoder = LlamaForCausalLM.from_pretrained(
#         args["llama_path"],
#         torch_dtype=base_dtype,
#         device_map="auto",   # <<< shard across GPUs
#     )

#     # (optional) overwrite base weights if provided
#     # (Note: this expects a single state_dict saved from same class; leave strict=False for safety)
#     if args["load_model_checkpoint_path"]:
#         print(f"[Model] Loading base weights: {args['load_model_checkpoint_path']}")
#         state = torch.load(args["load_model_checkpoint_path"], map_location="cpu", weights_only=True)
#         try:
#             llama_decoder.load_state_dict(state, strict=False)
#         except Exception:
#             pass  # ignore if state does not match decoder strictly

#     # 2) modality adapter on CPU first, cast to dtype
#     adapter_config = ModalityAdapterConfig(
#         input_dim=esm_encoder.config.hidden_size,
#         intermediate_dim=2048,
#         output_dim=llama_decoder.config.hidden_size,
#     )
#     adapter = ModalityAdapter(adapter_config)
#     adapter.to(first_device(), dtype=base_dtype)
#     # adapter.to(dtype=base_dtype)

#     # 3) composite model
#     model = Esm2LlamaInstructForCausalLM(
#         esm_encoder=esm_encoder,
#         adapter=adapter,
#         llama_decoder=llama_decoder,
#     )

#     # 4) LoRA
#     if args["load_adapter_checkpoint_dir"]:
#         print(f"[LoRA] Loading adapter from: {args['load_adapter_checkpoint_dir']}")
#         model = PeftModel.from_pretrained(model, args["load_adapter_checkpoint_dir"], is_trainable=True)
#     else:
#         print("[LoRA] Initializing new LoRA adapter")
#         lora_config = LoraConfig(
#             r=args["lora_rank"],
#             lora_alpha=args["lora_rank"] * 2,
#             lora_dropout=0.1,
#             bias="none",
#             init_lora_weights=True,
#             target_modules=[
#                 "self_attn.q_proj",
#                 "self_attn.k_proj",
#                 "self_attn.v_proj",
#                 "self_attn.o_proj",
#                 "mlp.gate_proj",
#                 "mlp.up_proj",
#                 "mlp.down_proj",
#             ],
#             modules_to_save=(["adapter.fc1", "adapter.fc2"] if not args["fix_modality_adapter"] else None),
#         )
#         model = get_peft_model(model, lora_config)

#     model.print_trainable_parameters()
#     return model

def load_model(args: Dict[str, Any]) -> PeftModel:
    """
    Build ESM encoder + LLaMA decoder + ModalityAdapter, then wrap with LoRA.
    All base models are loaded with device_map='auto' to shard across visible GPUs.
    """
    # base dtype derived from arg
    base_dtype = args["torch_dtype"]

    # 1) encoders/decoders on multiple GPUs
    esm_encoder = EsmModel.from_pretrained(
        args["esm_path"],
        add_pooling_layer=False,
        torch_dtype=base_dtype,
        device_map="auto",   # <<< shard across GPUs
    )
    llama_decoder = LlamaForCausalLM.from_pretrained(
        args["llama_path"],
        torch_dtype=base_dtype,
        device_map="auto",   # <<< shard across GPUs
    )

    # (optional) overwrite base weights if provided
    # llama 부분 로딩 (기존 코드 유지)
    if args["load_model_checkpoint_path"]:
        print(f"[Model] Loading base weights: {args['load_model_checkpoint_path']}")
        state = torch.load(args["load_model_checkpoint_path"], map_location="cpu", weights_only=True)
        try:
            llama_decoder.load_state_dict(state, strict=False)
        except Exception:
            pass  # ignore if state does not match decoder strictly

    # 2) modality adapter on CPU first, cast to dtype
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)

    # ▼▼▼▼▼ [여기부터 추가/수정된 부분] Stage 1 가중치 로드 코드 ▼▼▼▼▼
    if args["load_model_checkpoint_path"]:
        print(f"[Adapter] Attempting to load Stage 1 adapter weights...")
        checkpoint = torch.load(args["load_model_checkpoint_path"], map_location="cpu")
        
        # 체크포인트에서 'adapter.'가 포함된 키만 뽑아서 정리
        adapter_state = {}
        for k, v in checkpoint.items():
            # 키 이름이 'module.adapter.fc1...' 또는 'adapter.fc1...' 형태일 수 있음
            if "adapter." in k:
                # 'adapter.' 뒷부분만 가져옴 (예: fc1.weight)
                clean_k = k.split("adapter.")[-1]
                adapter_state[clean_k] = v
        
        if len(adapter_state) > 0:
            # 어댑터에 가중치 주입
            missing, unexpected = adapter.load_state_dict(adapter_state, strict=False)
            print(f"✅ [Adapter] Successfully loaded {len(adapter_state)} keys from Stage 1 checkpoint!")
            if len(missing) > 0:
                print(f"   (Missing keys: {missing})")
        else:
            print("⚠️ [Adapter] WARNING: Checkpoint loaded but NO adapter weights found inside.")
    # ▲▲▲▲▲ [여기까지 추가] ▲▲▲▲▲

    adapter.to(first_device(), dtype=base_dtype)
    # adapter.to(dtype=base_dtype)

    # 3) composite model
    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        llama_decoder=llama_decoder,
    )

    # 4) LoRA (기존 코드 그대로 유지)
    if args["load_adapter_checkpoint_dir"]:
        print(f"[LoRA] Loading adapter from: {args['load_adapter_checkpoint_dir']}")
        model = PeftModel.from_pretrained(model, args["load_adapter_checkpoint_dir"], is_trainable=True)
    else:
        print("[LoRA] Initializing new LoRA adapter")
        lora_config = LoraConfig(
            r=args["lora_rank"],
            lora_alpha=args["lora_rank"] * 2,
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=True,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
            modules_to_save=(["adapter.fc1", "adapter.fc2"] if not args["fix_modality_adapter"] else None),
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model


# ---------------------------
# Forward pass (teacher forcing)
# ---------------------------
def forward_pass(
    model: PeftModel,
    data_batch: Dict[str, Any],
    use_amp: bool,
    amp_dtype: torch.dtype,
    need_hidden: bool = False,
    device_for_inputs: Optional[torch.device] = None,
):
    """
    With device_map='auto', inputs should be placed on the "first" CUDA device (usually cuda:0).
    HF will dispatch to other devices internally.
    """
    if device_for_inputs is None:
        device_for_inputs = first_device()

    with autocast(enabled=use_amp, dtype=amp_dtype):
        outputs = model(
            input_ids=data_batch["input_ids"].to(device_for_inputs, non_blocking=True),
            attention_mask=data_batch["attention_mask"].to(device_for_inputs, non_blocking=True),
            labels=data_batch["labels"].to(device_for_inputs, non_blocking=True),
            protein_input_ids=data_batch["protein_input_ids"].to(device_for_inputs, non_blocking=True),
            protein_attention_mask=data_batch["protein_attention_mask"].to(device_for_inputs, non_blocking=True),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        loss = outputs.loss
    hidden = outputs.hidden_states if need_hidden else None
    return loss, hidden


# ---------------------------
# Train/Eval loops
# ---------------------------
def train_epoch(
    model: PeftModel,
    current_epoch: int,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scaler: Optional[GradScaler],
    args: Dict[str, Any],
    use_amp: bool,
    amp_dtype: torch.dtype,
):
    model.train()

    sum_loss = 0.0
    cnt_loss = 0
    sum_grad = 0.0
    cnt_grad = 0

    optimizer.zero_grad(set_to_none=True)

    t = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, data_batch in t:
        need_hidden = args["log_cosine_every"] > 0 and (batch_idx % args["log_cosine_every"] == 0)
        try:
            loss, hidden = forward_pass(
                model=model,
                data_batch=data_batch,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                need_hidden=need_hidden,
                device_for_inputs=first_device(),
            )
            # gradient accumulation
            loss = loss / args["gradient_accumulation_steps"]

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        except torch.cuda.OutOfMemoryError:
            print("[WARN] CUDA OOM in train. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        # logging
        step_loss = loss.item() * args["gradient_accumulation_steps"]
        cos_pos_debug = 0.0
        if need_hidden and hidden is not None:
            try:
                prot_h = hidden["protein"].mean(dim=1)
                text_h = hidden["text"].mean(dim=1)
                cos_pos_debug = cosine_similarity(prot_h, text_h).mean().detach().float().item()
            except Exception:
                cos_pos_debug = 0.0

        t.set_postfix(mode="train", epoch=f"{current_epoch}/{args['num_epochs']}", batch_loss=f"{step_loss:.4f}")
        wandb.log(
            {
                "train/step_loss": step_loss,
                "train/cos_pos": cos_pos_debug,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": current_epoch,
            }
        )

        sum_loss += step_loss
        cnt_loss += 1

        # optimizer step on accumulation boundary
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0:
            # grad clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(float("inf") if args["gradient_clipping"] is None else args["gradient_clipping"]),
            )
            sum_grad += float(gradnorm)
            cnt_grad += 1

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    mean_loss = (sum_loss / max(cnt_loss, 1))
    mean_grad = (sum_grad / max(cnt_grad, 1)) if cnt_grad > 0 else 0.0
    print(
        f"[epoch={current_epoch}/{args['num_epochs']}, "
        f"train_loss={mean_loss:.6f}, "
        f"epoch_lr={optimizer.param_groups[0]['lr']:.6e}, "
        f"epoch_gradnorm={mean_grad:.4f}]"
    )
    if not torch.isfinite(torch.tensor(mean_loss)):
        raise ValueError("NaN/Inf detected in training loss.")


def eval_epoch(
    model: PeftModel,
    current_epoch: int,
    dataloader: DataLoader,
    args: Dict[str, Any],
    use_amp: bool,
    amp_dtype: torch.dtype,
):
    model.eval()
    sum_loss = 0.0
    cnt_loss = 0

    t = tqdm(iter(dataloader), total=len(dataloader))
    with torch.no_grad():
        for data_batch in t:
            try:
                loss, _ = forward_pass(
                    model=model,
                    data_batch=data_batch,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    need_hidden=False,
                    device_for_inputs=first_device(),
                )
            except torch.cuda.OutOfMemoryError:
                print("[WARN] CUDA OOM in eval. Skipping batch.")
                torch.cuda.empty_cache()
                continue

            sum_loss += loss.item()
            cnt_loss += 1
            t.set_postfix(mode="eval", epoch=f"{current_epoch}/{args['num_epochs']}", batch_loss=f"{loss.item():.4f}")

    eval_loss_mean = (sum_loss / max(cnt_loss, 1))
    print(f"[epoch={current_epoch}/{args['num_epochs']}, eval_loss={eval_loss_mean:.6f}]")
    wandb.log({"eval/loss": eval_loss_mean, "eval/epoch": current_epoch})


# ---------------------------
# Main
# ---------------------------
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    args_ns = argParser.parse_args()
    args = vars(args_ns)

    torch.manual_seed(args["random_seed"])
    torch.cuda.manual_seed_all(args["random_seed"])

    # checkpoint dir
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    args["save_checkpoint_dir"] = os.path.join(args["save_checkpoint_dir"], f"checkpoints_{timestamp}")
    os.makedirs(args["save_checkpoint_dir"], exist_ok=True)

    print("####################")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("####################")

    # W&B
    run_name = args["wandb_run_name"] or f"stage2_dm_{args['train_split']}_{args['num_epochs']}ep_{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(project=args["wandb_project"], name=run_name, config=args, dir=args["save_checkpoint_dir"], reinit=True)

    # tokenizers
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(args["llama_path"], pad_token="<|reserved_special_token_0|>")
    if getattr(llama_tokenizer, "pad_token_id", None) is None:
        llama_tokenizer.pad_token = "<|reserved_special_token_0|>"

    # datasets
    train_dataset = Prot2TextLightDataset(jsonl_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.jsonl"))
    if args["debug_trim_train_split"]:
        train_dataset.data = train_dataset.data[: args["debug_trim_train_split"]]

    eval_dataset = Prot2TextLightDataset(jsonl_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.jsonl"))
    if args["debug_trim_eval_split"]:
        eval_dataset.data = eval_dataset.data[: args["debug_trim_eval_split"]]

    # collater (shared)
    collater = Prot2TextLightCollater(
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        mode="train",
        include_text_fields=args["include_text_fields"],
        name_dropout=args["name_dropout"],
        taxonomy_dropout=args["taxonomy_dropout"],
    )

    # loaders (NO DistributedSampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        shuffle=True,
        num_workers=args["workers"],
        pin_memory=args["pin_memory"],
        persistent_workers=args["persistent_workers"] and args["workers"] > 0,
        collate_fn=collater,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        shuffle=False,
        num_workers=args["workers"],
        pin_memory=args["pin_memory"],
        persistent_workers=args["persistent_workers"] and args["workers"] > 0,
        collate_fn=collater,
        drop_last=True,
    )

    print(f"Train/Eval datasets ready | train={len(train_dataset)} eval={len(eval_dataset)}")

    # model (device_map="auto")
    model = load_model(args)

    # optimizer/scheduler
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])

    # optionally restore optimizer/scheduler
    if args["load_optimizer_scheduler_checkpoint_path"]:
        print(f"[Opt/Sched] Loading from {args['load_optimizer_scheduler_checkpoint_path']}")
        state = torch.load(args["load_optimizer_scheduler_checkpoint_path"], map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

    # AMP scaler (if you ever set --amp true and dtype==fp16)
    use_amp = bool(args["amp"])
    amp_dtype = torch.float16 if args["amp_dtype"] == "fp16" else torch.bfloat16
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # epochs
    for epoch_idx in range(1, args["num_epochs"] + 1):
        train_epoch(
            model=model,
            current_epoch=epoch_idx,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        scheduler.step()

        eval_epoch(
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        # save checkpoints
        if (epoch_idx == 1) or (epoch_idx == args["num_epochs"]) or (epoch_idx % args["save_every_epochs"] == 0):
            adapter_dir = os.path.join(args["save_checkpoint_dir"], f"adapter_checkpoint_{epoch_idx}")
            model.save_pretrained(adapter_dir)
            print(f"[Save] LoRA adapter saved to {adapter_dir}")

            opt_sch_path = os.path.join(args["save_checkpoint_dir"], f"optimizer_scheduler_{epoch_idx}.pt")
            torch.save(
                {"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()},
                opt_sch_path,
            )
            print(f"[Save] Opt/Sched saved to {opt_sch_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
