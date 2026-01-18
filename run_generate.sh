#!/bin/bash
# ============================================
# Inference
# ============================================

BASE_DIR="/mnt/hdd/minju/"
ESM_DIR="$BASE_DIR/data/models/esm2_t36_3B_UR50D"
LLAMA_DIR="$BASE_DIR/data/models/Meta-Llama-3.1-8B-Instruct-hf"
JSONL_DIR="/home/minju/Prot2Text-V2/1215/"
GEN_DIR="/home/minju/Prot2Text-V2/1215/"

# Stage1 checkpoint
STAGE1_CKPT="/model_checkpoint_6.pt"

# Stage2  checkpoint 
STAGE2_ADAPTER="/adapter_checkpoint_3"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python scripts/generate_instruct_light_auto.py \
    --esm_path $ESM_DIR \
    --llama_path $LLAMA_DIR \
    --root_csv_dir $JSONL_DIR \
    --save_generation_dir $GEN_DIR \
    --load_model_checkpoint_path $STAGE1_CKPT \
    --load_adapter_checkpoint_dir $STAGE2_ADAPTER \
    --batch_size_per_device 1 \
    --random_seed 42 \
    --generate_split test_generation \
    --max_generation_length 256 \
    --debug_trim_generate_split 100\
    --num_beams 4
