"""
Light-weight dataset and data collater class for protein function prediction
instruction tuning. JSONL-backed strict parity version — identical to the CSV
implementation including NaN handling and potential TypeErrors.

This version intentionally replicates pandas.read_csv() behavior:

- Missing keys -> KeyError
- None, "", "NaN", "None", "null" (case-insensitive) -> float('nan')
- No automatic fallback to "unknown"
- TypeErrors may occur at the exact same points as in the CSV version:
  * len(NaN)
  * NaN + eos_token
  * NaN concatenation during chat prompt
- Otherwise identical tokenization, padding, truncation, and dropout
"""

import json
import random
from typing import Dict, List, Literal, Optional

import torch
import torch.utils.data
from transformers import PreTrainedTokenizer


# ==============================
# Dataset (JSONL strict CSV parity)
# ==============================

class Prot2TextLightDataset(torch.utils.data.Dataset):
    """Dataset class loading directly from a JSONL file (one JSON per line)."""
    def __init__(self, jsonl_path: str):
        super().__init__()
        self.data: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


# ==============================
# Collater (strict CSV parity)
# ==============================

class Prot2TextLightCollater:
    def __init__(
        self,
        sequence_tokenizer: PreTrainedTokenizer,
        description_tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "inference"] = "train",
        include_text_fields: bool = True,
        name_dropout: float = 0.8,
        taxonomy_dropout: float = 0.8,
        max_sequence_length: Optional[int] = 1021,
        max_description_length: Optional[int] = 512,
        system_message: str = (
            "You are a scientific assistant specialized in protein function "
            "predictions. Given the sequence embeddings and other information "
            "of a protein, describe its function clearly and concisely in "
            "professional language. "
        ),
        placeholder_token: str = '<|reserved_special_token_1|>',
    ):
        self.sequence_tokenizer = sequence_tokenizer
        self.description_tokenizer = description_tokenizer
        self.mode = mode

        self.include_text_fields = include_text_fields
        self.name_dropout = name_dropout
        self.taxonomy_dropout = taxonomy_dropout

        self.max_sequence_length = max_sequence_length
        self.max_description_length = max_description_length
        self.system_message = system_message
        self.placeholder_token = placeholder_token

    # -------------------------------
    # pandas.read_csv() NaN behavior
    # -------------------------------
    def _safe_get(self, item: Dict[str, str], key: str):
        """
        Mimic pandas.read_csv NaN conversion.
        - Missing key -> KeyError
        - None, "", "NaN", "None", "null" -> float('nan')
        - Non-string, non-null values returned as-is (no type coercion)
        """
        if key not in item:
            raise KeyError(f"Missing key: {key}")
        val = item[key]
        # pandas: None → NaN
        if val is None:
            return float('nan')
        # pandas: "", "NaN", "None", "null" (case-insensitive) → NaN
        if isinstance(val, str) and val.lower() in {"", "nan", "none", "null"}:
            return float('nan')
        # non-missing: return as-is
        return val

    # -------------------------------
    # Collation
    # -------------------------------
    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # ---- strict key access (no defaults) ----
        accessions   = [self._safe_get(item, "AlphaFoldDB") for item in batch]
        fullnames    = [self._safe_get(item, "Full Name") for item in batch]
        taxons       = [self._safe_get(item, "taxon") for item in batch]
        sequences    = [self._safe_get(item, "sequence") for item in batch]
        descriptions = [self._safe_get(item, "function") for item in batch]

        # ---- name / taxon dropout (identical to CSV) ----
        fullnames = [
            fullname
            if isinstance(fullname, str) and random.random() > self.name_dropout
            else "unknown"
            for fullname in fullnames
        ]
        taxons = [
            taxon
            if isinstance(taxon, str) and random.random() > self.taxonomy_dropout
            else "unknown"
            for taxon in taxons
        ]

        # ---- random cropping ----
        # identical behavior: len(NaN) raises TypeError
        for i in range(len(sequences)):
            if len(sequences[i]) > self.max_sequence_length:  # will fail for NaN
                start = random.randint(0, len(sequences[i]) - self.max_sequence_length)
                sequences[i] = sequences[i][start:start + self.max_sequence_length]

        # ---- tokenize sequences (ESM) ----
        self.sequence_tokenizer.padding_side = "right"
        seq_tok = self.sequence_tokenizer(
            sequences,
            truncation=True,
            padding="longest",
            max_length=self.max_sequence_length + 2,
            return_tensors="pt",
        )
        sequence_input_ids = seq_tok["input_ids"]
        sequence_attention_mask = seq_tok["attention_mask"]
        sequence_lens = sequence_attention_mask.sum(dim=1).tolist()

        # ---- build user messages ----
        if self.include_text_fields:
            user_messages = [
                f"Protein name: {fullname}; Taxon: {taxon}; "
                + "Sequence embeddings: " + self.placeholder_token * seq_len
                for fullname, taxon, seq_len in zip(fullnames, taxons, sequence_lens)
            ]
        else:
            user_messages = [
                "Sequence embeddings: " + self.placeholder_token * sequence_lens
                for sequence_lens in sequence_lens
            ]

        prompt_conversations = [
            [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_msg},
            ]
            for user_msg in user_messages
        ]
        
        # ---- tokenize prompts (LLM; left padding) ----
        self.description_tokenizer.padding_side = "left"
        prompt_tok = self.description_tokenizer.apply_chat_template(
            prompt_conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding="longest",
            return_tensors="pt",
            return_dict=True,
        )
        prompt_input_ids = prompt_tok["input_ids"]
        prompt_attention_mask = prompt_tok["attention_mask"]

        # ---- tokenize descriptions (LLM; right padding, +EOS) ----
        # identical: NaN + eos_token will raise TypeError here
        self.description_tokenizer.padding_side = "right"
        desc_tok = self.description_tokenizer(
            [d + self.description_tokenizer.eos_token for d in descriptions],
            add_special_tokens=False,
            truncation=True,
            padding="longest",
            max_length=self.max_description_length,
            return_tensors="pt",
        )
        description_input_ids = desc_tok["input_ids"]
        description_attention_mask = desc_tok["attention_mask"]

        # ---- truncate for parity ----
        if description_input_ids.size(1) > self.max_description_length:
            description_input_ids = description_input_ids[:, :self.max_description_length]
            description_attention_mask = description_attention_mask[:, :self.max_description_length]

        # ---- labels ----
        labels = description_input_ids.clone()
        labels[description_attention_mask == 0] = -100

        # ---- assemble batch ----
        if self.mode == "train":
            return {
                "name": accessions,
                "protein_input_ids": sequence_input_ids,
                "protein_attention_mask": sequence_attention_mask,
                "input_ids": torch.cat([prompt_input_ids, description_input_ids], dim=1),
                "attention_mask": torch.cat([prompt_attention_mask, description_attention_mask], dim=1),
                "labels": torch.cat([
                    torch.full_like(prompt_input_ids, fill_value=-100),
                    labels,
                ], dim=1),
                "description_input_ids": description_input_ids,
                "description_attention_mask": description_attention_mask,
            }

        elif self.mode == "inference":
            return {
                "name": accessions,
                "protein_input_ids": sequence_input_ids,
                "protein_attention_mask": sequence_attention_mask,
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "description_input_ids": description_input_ids,
            }

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
