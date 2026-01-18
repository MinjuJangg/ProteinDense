#!/bin/bash
# ============================================
# Prot2Text-V2 Stage 2 - Instruction Tuning
# ============================================

#  기본 설정
BASE_DIR="/mnt/hdd/minju/"
ESM_DIR="$BASE_DIR/data/models/esm2_t36_3B_UR50D"
LLAMA_DIR="$BASE_DIR/data/models/Meta-Llama-3.1-8B-Instruct-hf"
JSONL_DIR="/home/minju/Prot2Text-V2/1213final"
CKPT_DIR="$BASE_DIR/checkpoints_graduate_stage2"


LOAD_MODEL_CKPT="/mnt/hdd/minju/checkpoints_graduate/checkpoints_251214_140454/model_checkpoint_6.pt"


#  학습 하이퍼파라미터
BATCH_SIZE=2
EPOCHS=10
LR=5e-5
GAMMA=0.95
GRAD_ACC=8
SEED=42
LORA_RANK=16


NAME_DROPOUT=0.8
TAXON_DROPOUT=0.8
INCLUDE_TEXT_FIELDS=true

# Adapter 설정
FIX_MODALITY_ADAPTER=false  # true면 adapter는 freeze

# Split 지정
TRAIN_SPLIT="train"
EVAL_SPLIT="test"

export TOKENIZERS_PARALLELISM=true
export LOGURU_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')  # GPU 개수 자동 계산

#  실행
python scripts/train_instruct_auto.py \
  --esm_path "$ESM_DIR" \
  --llama_path "$LLAMA_DIR" \
  --root_csv_dir "$JSONL_DIR" \
  --save_checkpoint_dir "$CKPT_DIR" \
  --load_model_checkpoint_path "$LOAD_MODEL_CKPT" \
  --load_optimizer_scheduler_checkpoint_path "$LOAD_OPT_SCHED_CKPT" \
  --torch_dtype bfloat16 \
  --batch_size_per_device $BATCH_SIZE \
  --num_epochs $EPOCHS \
  --save_every_epochs 1 \
  --gradient_accumulation_steps $GRAD_ACC \
  --learning_rate $LR \
  --scheduler_gamma $GAMMA \
  --random_seed $SEED \
  --fix_modality_adapter $FIX_MODALITY_ADAPTER \
  --lora_rank $LORA_RANK \
  --include_text_fields $INCLUDE_TEXT_FIELDS \
  --name_dropout $NAME_DROPOUT \
  --taxonomy_dropout $TAXON_DROPOUT \
  --train_split $TRAIN_SPLIT \
  --eval_split $EVAL_SPLIT \
  --debug_trim_eval_split 256\
  --gradient_clipping 1.0


#   --load_adapter_checkpoint_dir "/home/minju/Prot2Text-V2/checkpoints_stage2/checkpoints_251009_124522/adapter_checkpoint_1"\