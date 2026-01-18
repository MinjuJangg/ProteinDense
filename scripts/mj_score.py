import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any
import evaluate

# =====================================================
#  Disable matplotlib import BEFORE loading bert_score
# =====================================================
sys.modules["matplotlib"] = None
sys.modules["matplotlib.pyplot"] = None
sys.modules["matplotlib.backends"] = None

from bert_score import score as bert_score_score


# ========================
# Helper Functions
# ========================

def normalize_text(text: str) -> str:
    """Lowercase + remove punctuation for exact match."""
    return re.sub(r"[^\w]", "", text.lower())


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    exact = sum(1 for p, r in zip(predictions, references) if normalize_text(p) == normalize_text(r))
    return exact / len(predictions)


def compute_bleu(predictions: List[str], references: List[str], max_order: int = 4):
    """Compute BLEU score with HuggingFace evaluate."""
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references, max_order=max_order)


def compute_rouge(predictions: List[str], references: List[str]):
    """Compute ROUGE using HF evaluate."""
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_bert(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BERTScore using bert_score directly.
    matplotlib import is disabled at top.
    """
    P, R, F1 = bert_score_score(predictions, references, lang="en", verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean())
    }


# ========================
# Core Evaluation
# ========================

def evaluate_json(json_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Evaluate a single JSON file with true/pred fields."""

    with open(json_path, "r") as f:
        data = json.load(f)

    refs, preds = [], []

    for _, item in data.items():
        if "true" in item and "pred" in item:
            refs.append(item["true"])
            preds.append(item["pred"])

    if verbose:
        print(f"Loaded {len(refs)} samples from {json_path}")

    results = {
        "exact_match": compute_exact_match(preds, refs),
        "bleu2": compute_bleu(preds, refs, max_order=2),
        "bleu4": compute_bleu(preds, refs, max_order=4),
        "rouge": compute_rouge(preds, refs),
        "bert": compute_bert(preds, refs)
    }

    return results


# ========================
# CLI
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file or folder")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    # Collect JSON files
    if os.path.isdir(args.json_path):
        json_files = [os.path.join(args.json_path, f) for f in os.listdir(args.json_path) if f.endswith(".json")]
    else:
        json_files = [args.json_path]

    print("=======================================")
    print(f"Evaluating {len(json_files)} file(s)")
    print("=======================================")

    for path in json_files:
        print(f"\n🧩 File: {path}")
        metrics = evaluate_json(path, verbose=args.verbose)

        for key, value in metrics.items():
            print(f"{key}: {value}")
