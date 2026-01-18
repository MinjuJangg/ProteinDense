import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ============================================================
# 1. Configuration
# ============================================================

DOMAIN_JSONL_LIST = [
    "train.jsonl",
    "eval.jsonl",
    "test.jsonl",
]

MODEL_PATH = "/mnt/hdd/minju/data/models/Meta-Llama-3.1-8B-Instruct-hf"
OUTPUT_PATH = "/home/minju/Prot2Text-V2/domain_prototype_train.pt"  # 파일명에 8192 명시 추천

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-5  # Whitening stability constant

# ============================================================


print(f"[Init] Device: {DEVICE}")
print("[Init] Loading LLaMA tokenizer + model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# ============================================================

@torch.no_grad()
def encode_text_mix(text: str):
    """
    Text -> LLaMA -> Mean Pooling (4096) + Std Pooling (4096) -> Concat (8192)
    Returns: Normalized 8192-dim tensor (float32, CPU)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(model.device)

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (1, L, 4096)
    
    mask = inputs["attention_mask"].unsqueeze(-1).float() # (1, L, 1)
    sum_mask = mask.sum(dim=1).clamp(min=1.0) # (1, 1)

    masked_hidden = hidden * mask
    mean_emb = masked_hidden.sum(dim=1) / sum_mask # (1, 4096)

    diff_sq = (hidden - mean_emb.unsqueeze(1)) ** 2
    masked_diff_sq = diff_sq * mask
    var_emb = masked_diff_sq.sum(dim=1) / sum_mask
    std_emb = torch.sqrt(var_emb + 1e-9) # (1, 4096)

    mix_emb = torch.cat([mean_emb, std_emb], dim=-1) # (1, 8192)

    mix_emb = torch.nn.functional.normalize(mix_emb, p=2, dim=-1)

    return mix_emb.squeeze(0).float().cpu()

# ============================================================

def main():
    # --------------------------------------------------------
    print("\n[Step 1] Collecting domain descriptions...")
    domain_texts = {}

    for jsonl_path in DOMAIN_JSONL_LIST:
        print(f"  - Reading {jsonl_path}")
        if not os.path.exists(jsonl_path):
            print(f"    [WARN] File not found: {jsonl_path}")
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    domain = row["accession"]
                    desc = row["function_text"].strip()
                    
                    if domain not in domain_texts:
                        domain_texts[domain] = []
                    domain_texts[domain].append(desc)
                except json.JSONDecodeError:
                    continue

    # --------------------------------------------------------
    raw_prototypes = {}
    failed_domains = []

    for domain, texts in tqdm(domain_texts.items(), desc="Encoding"):
        emb_list = []
        for t in texts:
            try:
                emb = encode_text_mix(t)
                emb_list.append(emb)
            except Exception as e:
                pass
        
        if not emb_list:
            failed_domains.append(domain)
            continue

        emb_stack = torch.stack(emb_list, dim=0)
        proto = emb_stack.mean(dim=0) # (8192,)
        raw_prototypes[domain] = proto

    # --------------------------------------------------------
    
    all_raw = torch.stack(list(raw_prototypes.values())) # (N, 8192)
    
    calc_device = DEVICE if torch.cuda.is_available() else "cpu"
    
    try:
        all_raw_dev = all_raw.to(calc_device)
        
        mean_vec = all_raw_dev.mean(dim=0, keepdim=True)
        X = all_raw_dev - mean_vec

        cov = torch.matmul(X.T, X) / (X.shape[0] - 1)

        U, S, Vh = torch.linalg.svd(cov + EPS * torch.eye(cov.size(0), device=calc_device))

        W = U @ torch.diag(1.0 / torch.sqrt(S + EPS)) @ U.T
        


    # --------------------------------------------------------
    final_prototypes = {}
    
    
    mean_vec_cpu = mean_vec.cpu()
    W_cpu = W.cpu()

    for domain, proto in tqdm(raw_prototypes.items(), desc="Whitening"):
        p = proto.unsqueeze(0) - mean_vec_cpu # Center
        p = p @ W_cpu                         # Whiten
        p = torch.nn.functional.normalize(p, p=2, dim=-1) # Re-normalize
        
        final_prototypes[domain] = p.squeeze(0).float()

    # 저장
    print(f"  > Saving to {OUTPUT_PATH}")
    torch.save(final_prototypes, OUTPUT_PATH)

if __name__ == "__main__":
    main()
