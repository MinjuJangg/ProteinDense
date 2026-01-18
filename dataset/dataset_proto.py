import os
import torch
import pandas as pd
from typing import Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer


class Prot2TextInstructDatasetJSONL:


    def __init__(
        self,
        root_dir: Union[str, os.PathLike],
        jsonl_path: Union[str, os.PathLike],
        sequence_tokenizer: PreTrainedTokenizer,
        description_tokenizer: PreTrainedTokenizer,
        max_sequence_length: int = 1021,
        max_description_length: int = 512,
        system_message: str = (
            "You are a scientific assistant specialized in protein function prediction. "
            "Given the sequence embeddings and other information of a protein, "
            "describe its function clearly and concisely in professional language."
        ),
        placeholder_token: str = "<|reserved_special_token_1|>",
        use_cache: bool = True,
        save_cache: bool = True,
    ):
        self.root_dir = root_dir
        self.processed_dir = os.path.join(root_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        self.uniprot_df = pd.read_json(jsonl_path, lines=True)
        self.sequence_tokenizer = sequence_tokenizer
        self.description_tokenizer = description_tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_description_length = max_description_length
        self.system_message = system_message
        self.placeholder_token = placeholder_token
        self.use_cache = use_cache
        self.save_cache = save_cache

        cached = len([f for f in os.listdir(self.processed_dir) if f.endswith(".pt")])
        print(f"Loaded {len(self.uniprot_df)} samples")
        print(f"Cache directory: {self.processed_dir} ({cached} cached files found)")
        print(f"Cache policy: use_cache={use_cache}, save_cache={save_cache}")

    # -------------------------------
    def __len__(self):
        return len(self.uniprot_df)

    # -------------------------------
    def __getitem__(self, idx: int):
        row = self.uniprot_df.iloc[idx]
        # domain-level → IPR accession
        acc = str(row.get("accession", f"sample_{idx}"))
        pt_path = os.path.join(self.processed_dir, f"{acc}.pt")

        if self.use_cache and os.path.exists(pt_path):
            try:
                return torch.load(pt_path)
            except Exception as e:
                print(f"[WARN] Failed to load cache for {acc}: {e}")

        data = self._compose_and_tokenize_chat(row)

        if self.save_cache:
            try:
                torch.save(data, pt_path)
            except Exception as e:
                print(f"[WARN] Failed to save cache for {acc}: {e}")

        return data

    # -------------------------------
    def _compose_and_tokenize_chat(self, row: pd.Series) -> Dict[str, Any]:
        """JSONL row → tokenized data dict"""

        sequence = str(row.get("sequence", ""))
        description = str(row.get("function_text", ""))
        fullname = str(row.get("name", row.get("Full Name", "unknown")))
        taxon = str(row.get("taxon", "unknown"))

        accession = str(row.get("accession", None))
        parent_accession = str(row.get("parent_accession", None))

        # truncate sequence
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]

        # Tokenize description
        desc_ids = self.description_tokenizer(
            [description], add_special_tokens=False, return_tensors="pt"
        )["input_ids"]

        if desc_ids.size(-1) > self.max_description_length:
            desc_ids = desc_ids[:, : self.max_description_length]
            description = self.description_tokenizer.decode(desc_ids[0])

        # Prompt
        user_message = (
            f"Protein name: {fullname} ; Taxon: {taxon} ; Sequence embeddings: "
            + self.placeholder_token * (len(sequence) + 2)
        )
        prompt_msg = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]

        prompt_ids = self.description_tokenizer.apply_chat_template(
            prompt_msg,
            add_generation_prompt=True,
            tokenize=True,
            padding=False,
            return_tensors="pt",
        )

        sequence_ids = self.sequence_tokenizer(
            [sequence], add_special_tokens=True, return_tensors="pt"
        )["input_ids"]

        description_ids = self.description_tokenizer(
            [description + self.description_tokenizer.eos_token],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        return {
            "sequence_input_ids": sequence_ids,
            "prompt_input_ids": prompt_ids,
            "description_input_ids": description_ids,
            "name": fullname,
            "taxon": taxon,

            "accession": accession,
            "parent_accession": parent_accession,
        }
