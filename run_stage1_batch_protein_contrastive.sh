#!/bin/bash
# ============================================
# Prot2Text-V2 Stage 1 - Contrastive Training
# ============================================

# 기본 설정
BASE_DIR="/mnt/hdd/minju/"
ESM_DIR="$BASE_DIR/data/models/esm2_t36_3B_UR50D"
LLAMA_DIR="$BASE_DIR/data/models/Meta-Llama-3.1-8B-Instruct-hf"
DATA_DIR="/home/minju/Prot2Text_mj/data/Prot2Text-Llama3-Data"
JSONL_DIR="/home/minju/Prot2Text-V2/1213final"
CKPT_DIR="$BASE_DIR/checkpoints_domain"


# =========================
# 학습 하이퍼파라미터
# =========================
BATCH_SIZE=2              
EPOCHS=20
LR=2e-4
GAMMA=0.99
GRAD_ACC=16               #  2 × 4 × 16 = 128
SEGMENTS=1
SEED=42


TRAIN_SPLIT="train"
EVAL_SPLIT="eval"

# =========================
export TOKENIZERS_PARALLELISM=true
export LOGURU_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=


# =========================
python scripts/train_contrast_batch.py \
  --esm_path "$ESM_DIR" \
  --llama_path "$LLAMA_DIR" \
  --root_dataset_dir "$DATA_DIR" \
  --root_csv_dir "$JSONL_DIR" \
  --save_checkpoint_dir "$CKPT_DIR" \
  --torch_dtype bfloat16 \
  --batch_size_per_device $BATCH_SIZE \
  --num_epochs $EPOCHS \
  --save_every_epochs 1 \
  --gradient_accumulation_steps $GRAD_ACC \
  --learning_rate $LR \
  --scheduler_gamma $GAMMA \
  --random_seed $SEED \
  --contrastive_num_segments $SEGMENTS \
  --train_split $TRAIN_SPLIT \
  --eval_split $EVAL_SPLIT \
  --gradient_clipping 1.0 \
  --moco_queue_size 256
