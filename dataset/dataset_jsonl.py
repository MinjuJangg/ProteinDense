# # import os
# # import pandas as pd
# # import torch
# # from tqdm import tqdm
# # from typing import Any, Dict, Optional, Union
# # from transformers import PreTrainedTokenizer


# # class Prot2TextInstructDatasetJSONL:
# # # class Prot2TextInstructDataset:
# #     """
# #     JSONL 기반 Prot2Text-Instruct (NoGraph) 버전.
# #     domain_captioned_merged.jsonl 파일을 그대로 사용.
# #     """

# #     def __init__(
# #         self,
# #         root_dir: Union[str, os.PathLike],
# #         jsonl_path: Union[str, os.PathLike],
# #         sequence_tokenizer: PreTrainedTokenizer,
# #         description_tokenizer: PreTrainedTokenizer,
# #         max_sequence_length: Optional[int] = 1021,
# #         max_description_length: Optional[int] = 512,
# #         system_message: str = (
# #             "You are a scientific assistant specialized in protein function "
# #             "prediction. Given the sequence embeddings and other information "
# #             "of a protein, describe its function clearly and concisely in "
# #             "professional language."
# #         ),
# #         placeholder_token: str = '<|reserved_special_token_1|>',
# #     ):
# #         self.root_dir = root_dir
# #         self.processed_dir = os.path.join(root_dir, "processed")
# #         os.makedirs(self.processed_dir, exist_ok=True)

# #         # JSONL 파일 읽기
# #         self.uniprot_df = pd.read_json(jsonl_path, lines=True)
# #         self.sequence_tokenizer = sequence_tokenizer
# #         self.description_tokenizer = description_tokenizer
# #         self.max_sequence_length = max_sequence_length
# #         self.max_description_length = max_description_length
# #         self.system_message = system_message
# #         self.placeholder_token = placeholder_token

# #         print(f"Loaded {len(self.uniprot_df)} entries from {jsonl_path}")
# #         print(f"➡ Output directory: {self.processed_dir}")

# #     # -------------------------------
# #     def process_text(self):
# #         """JSONL 기반 전처리 (CSV 버전과 동일한 로직)"""
# #         for _, row in tqdm(self.uniprot_df.iterrows(), total=len(self.uniprot_df)):
# #             try:
# #                 acc = str(row.get("AlphaFoldDB", row.get("accession", "unknown")))
# #                 data = self._compose_and_tokenize_chat(row)

# #                 if not data:
# #                     continue

# #                 save_path = os.path.join(self.processed_dir, f"{acc}.pt")
# #                 torch.save(data, save_path)
# #             except Exception as e:
# #                 print(f"[Error] Failed to process {acc}: {e}")

# #         print(f"\nJSONL preprocessing complete for {len(self.uniprot_df)} entries.")
# #         print(f"➡ Saved under {self.processed_dir}")

# #     # -------------------------------
# #     def _compose_and_tokenize_chat(self, row: pd.Series) -> Dict[str, torch.Tensor]:
# #         """Meta-Llama chat prompt 구성 및 토크나이징"""
# #         try:
# #             sequence = str(row["sequence"])
# #             description = str(row.get("function_text", ""))
# #             fullname = str(row.get("Full Name", row.get("name", "unknown")))
# #             taxon = str(row.get("taxon", "unknown"))

# #             # 1. Trim
# #             if self.max_sequence_length and len(sequence) > self.max_sequence_length:
# #                 sequence = sequence[: self.max_sequence_length]

# #             desc_ids = self.description_tokenizer(
# #                 [description],
# #                 add_special_tokens=False,
# #                 return_tensors="pt",
# #             )["input_ids"]

# #             if self.max_description_length and desc_ids.size(-1) > self.max_description_length:
# #                 desc_ids = desc_ids[:, : self.max_description_length]
# #                 description = self.description_tokenizer.decode(desc_ids[0])

# #             # 2. User prompt
# #             user_message = (
# #                 f"Protein name: {fullname} ; Taxon: {taxon} ; Sequence embeddings: "
# #                 + self.placeholder_token * (len(sequence) + 2)
# #             )

# #             prompt_conversation = [
# #                 {"role": "system", "content": self.system_message},
# #                 {"role": "user", "content": user_message},
# #             ]
# #             prompt_ids = self.description_tokenizer.apply_chat_template(
# #                 prompt_conversation,
# #                 add_generation_prompt=True,
# #                 tokenize=True,
# #                 padding=False,
# #                 return_tensors="pt",
# #             )

# #             # 3. Tokenize sequence + description
# #             sequence_ids = self.sequence_tokenizer(
# #                 [sequence],
# #                 add_special_tokens=True,
# #                 return_tensors="pt",
# #             )["input_ids"]

# #             description_ids = self.description_tokenizer(
# #                 [description + self.description_tokenizer.eos_token],
# #                 add_special_tokens=False,
# #                 return_tensors="pt",
# #             )["input_ids"]

# #             return {
# #                 "sequence_input_ids": sequence_ids,
# #                 "prompt_input_ids": prompt_ids,
# #                 "description_input_ids": description_ids,
# #                 "name": fullname,
# #                 "taxon": taxon,
# #             }

# #         except Exception as e:
# #             print(f"[Warning] Failed to process row: {e}")
# #             return {}

# #     # -------------------------------
# #     def __len__(self):
# #         return len(self.uniprot_df)

# #     def __getitem__(self, idx: int):
# #         row = self.uniprot_df.iloc[idx]
# #         return self._compose_and_tokenize_chat(row)

# import os
# import torch
# import pandas as pd
# from tqdm import tqdm
# from transformers import PreTrainedTokenizer

# class Prot2TextInstructDatasetJSONL:
#     """
#     Hybrid version:
#     - if processed/*.pt exists → use cached tensors
#     - else → read JSONL, tokenize on the fly, and optionally cache it
#     """

#     def __init__(
#         self,
#         root_dir: str,
#         jsonl_path: str,
#         sequence_tokenizer: PreTrainedTokenizer,
#         description_tokenizer: PreTrainedTokenizer,
#         max_sequence_length: int = 1021,
#         max_description_length: int = 512,
#         system_message: str = (
#             "You are a scientific assistant specialized in protein function prediction. "
#             "Given the sequence embeddings and other information of a protein, "
#             "describe its function clearly and concisely in professional language."
#         ),
#         placeholder_token: str = "<|reserved_special_token_1|>",
#         use_cache: bool = True,  # ✅ 캐시 사용 여부
#         save_cache: bool = True, # ✅ 캐시 자동 저장 여부
#     ):
#         self.root_dir = root_dir
#         self.processed_dir = os.path.join(root_dir, "processed")
#         os.makedirs(self.processed_dir, exist_ok=True)

#         self.uniprot_df = pd.read_json(jsonl_path, lines=True)
#         self.sequence_tokenizer = sequence_tokenizer
#         self.description_tokenizer = description_tokenizer
#         self.max_sequence_length = max_sequence_length
#         self.max_description_length = max_description_length
#         self.system_message = system_message
#         self.placeholder_token = placeholder_token
#         self.use_cache = use_cache
#         self.save_cache = save_cache

#         cached = len([f for f in os.listdir(self.processed_dir) if f.endswith(".pt")])
#         print(f"✅ Hybrid dataset loaded: {len(self.uniprot_df)} samples")
#         print(f"📦 Cache directory: {self.processed_dir} ({cached} .pt files found)")
#         print(f"➡ Cache policy: use_cache={use_cache}, save_cache={save_cache}\n")

#     # -------------------------------
#     def __len__(self):
#         return len(self.uniprot_df)

#     # -------------------------------
#     def __getitem__(self, idx):
#         row = self.uniprot_df.iloc[idx]
#         acc = str(row.get("AlphaFoldDB", row.get("accession", f"sample_{idx}")))
#         pt_path = os.path.join(self.processed_dir, f"{acc}.pt")

#         # ✅ 캐시가 존재하면 바로 로드
#         if self.use_cache and os.path.exists(pt_path):
#             try:
#                 return torch.load(pt_path)
#             except Exception as e:
#                 print(f"[WARN] Failed to load cache for {acc}: {e}")

#         # ✅ 아니면 JSONL 즉석 토크나이징
#         data = self._compose_and_tokenize_chat(row)

#         # ✅ 필요 시 캐시 저장
#         if self.save_cache:
#             try:
#                 torch.save(data, pt_path)
#             except Exception as e:
#                 print(f"[WARN] Failed to save cache for {acc}: {e}")

#         return data

#     # -------------------------------
#     def _compose_and_tokenize_chat(self, row):
#         """JSONL 1개 row를 토크나이징하여 tensor dict 생성"""
#         sequence = str(row.get("sequence", ""))
#         description = str(row.get("function_text", ""))
#         fullname = str(row.get("Full Name", row.get("name", "unknown")))
#         taxon = str(row.get("taxon", "unknown"))

#         # 길이 제한
#         if len(sequence) > self.max_sequence_length:
#             sequence = sequence[: self.max_sequence_length]

#         desc_ids = self.description_tokenizer(
#             [description], add_special_tokens=False, return_tensors="pt"
#         )["input_ids"]
#         if desc_ids.size(-1) > self.max_description_length:
#             desc_ids = desc_ids[:, : self.max_description_length]
#             description = self.description_tokenizer.decode(desc_ids[0])

#         # user/system 프롬프트 생성
#         user_message = (
#             f"Protein name: {fullname} ; Taxon: {taxon} ; Sequence embeddings: "
#             + self.placeholder_token * (len(sequence) + 2)
#         )
#         prompt_conversation = [
#             {"role": "system", "content": self.system_message},
#             {"role": "user", "content": user_message},
#         ]
#         prompt_ids = self.description_tokenizer.apply_chat_template(
#             prompt_conversation,
#             add_generation_prompt=True,
#             tokenize=True,
#             padding=False,
#             return_tensors="pt",
#         )

#         # 시퀀스/디스크립션 토크나이징
#         sequence_ids = self.sequence_tokenizer(
#             [sequence], add_special_tokens=True, return_tensors="pt"
#         )["input_ids"]
#         description_ids = self.description_tokenizer(
#             [description + self.description_tokenizer.eos_token],
#             add_special_tokens=False,
#             return_tensors="pt",
#         )["input_ids"]

#         return {
#             "sequence_input_ids": sequence_ids,
#             "prompt_input_ids": prompt_ids,
#             "description_input_ids": description_ids,
#             "name": fullname,
#             "taxon": taxon,
#             "accession": accession,
#         }

import os
import torch
import pandas as pd
from typing import Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer


class Prot2TextInstructDatasetJSONL:
    """
    JSONL 기반 Prot2Text Dataset (Hybrid cache 지원)
    Prototype Regression 학습을 위해 accession 및 parent_accession 포함.
    """

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
        description = str(row.get("function", ""))
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

            # ★ Prototype Regression Training 필수 key
            
            "accession": accession,
            "parent_accession": parent_accession,
            "protein_input_ids": sequence_ids,      # align.py 호환용
            "function": description,  
            "description_text": description
            
        }
