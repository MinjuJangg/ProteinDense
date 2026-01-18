"""
Light-weight dataset and data collater class for protein function prediction 
instruction tuning. To be used with Esm2LlamaInstructForCausalLM. 

Such flexible implementation is designed to fetch raw text data from a CSV file 
and perform tokenization and padding on-the-fly. This is useful when the default 
user message and chat template is not suitable for the task at hand.

Can only be used if the model is not requiring graph-related data.

Every batch from DataLoader will contain following attributes:
    * Training mode (train-eval with teacher-forcing): 
        - graph related features:
            None
        - amino-acid sequence: 
            - protein_input_ids (bsz, max_seq_len+2)  # bos and eos tokens
            - protein_attention_mask (bsz, max_seq_len+2)  # right padding
        - concatenated chat:
            - input_ids (bsz, max_prompt_len+max_text_len+1)
            - attention_mask (bsz, max_prompt_len+max_text_len+1)
            - labels (bsz, max_prompt_len+max_text_len+1)
        - standalone description for contrastive learning: 
            - description_input_ids (bsz, max_text_len+1)  # eos token only
            - description_attention_mask (bsz, max_text_len+1)  # right padding
            
        ids       = [left-pad + bos  + prompt & description + eot  + right-pad]
        mask      = [0s       + 1    + 1s     & 1s          + 1    + 0s       ]
        labels    = [-100s    + -100 + -100s  & description + eot  + -100s    ]
        desc_ids  =                         [ & description + eot  + right-pad]
        desc_mask =                         [ & 1s          + 1    + 0s       ] 
        
    * Inference mode (iterative generation):
        - graph related features: 
            None
        - amino-acid sequence: 
            - protein_input_ids (bsz, max_seq_len+2)  # bos and eos tokens
            - protein_attention_mask (bsz, max_seq_len+2)  # right padding
        - prompt chat: 
            - input_ids (bsz, max_prompt_len)
            - attention_mask (bsz, max_prompt_len)
            - description_input_ids (bsz, max_text_len+1)  # for evaluation

        ids      = [left-pad + bos + prompt & ]
        mask     = [0s       + 1   + 1s     & ]
        desc_ids =                        [ & description + eot + right-pad]

Example of usage: 
>>> from torch.utils.data import DataLoader
>>> from transformers import AutoTokenizer
>>> from dataset import Prot2TextLightDataset, Prot2TextLightCollater
>>> esm_tokenizer = AutoTokenizer.from_pretrained("/data/esm2_t33_650M_UR50D")
>>> llama_tokenizer = AutoTokenizer.from_pretrained(
        "/data/Meta-Llama-3.1-8B-Instruct-hf", 
        pad_token='<|reserved_special_token_0|>'
    )
>>> train_dataset = Prot2TextLightDataset("./data/train.csv")
>>> train_collater = Prot2TextLightCollater(
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        mode="train"
    )
>>> train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=train_collater, 
        pin_memory=True, 
        drop_last=True
    )
"""

import random
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
import torch.utils.data
from transformers import PreTrainedTokenizer


class Prot2TextLightDataset(torch.utils.data.Dataset): 
    """Dataset class loading directly from single CSV file."""
    def __init__(self, csv_path: str):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            column_name: self.data.iloc[idx][column_name] 
            for column_name in self.data.columns
        }
    

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

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # group data across batch
        accessions = [item["AlphaFoldDB"] for item in batch]
        fullnames = [item["Full Name"] for item in batch]
        taxons = [item["taxon"] for item in batch]
        sequences = [item["sequence"] for item in batch]
        descriptions = [item["function"] for item in batch]
        
        # replace nan in name and taxon with unknown
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

        # for each sequence in sequences
        # if the sequence is origianlly longer than max_sequence_length, take a segment of that length randomly 
        # else do nothing
        for i in range(len(sequences)):
            if len(sequences[i]) > self.max_sequence_length:
                start = random.randint(0, len(sequences[i]) - self.max_sequence_length)
                sequences[i] = sequences[i][start:start + self.max_sequence_length]

        # truncate and tokenize sequences
        self.sequence_tokenizer.padding_side = "right"
        tokenized_sequences = self.sequence_tokenizer(
            sequences, 
            truncation=True, 
            padding="longest", 
            max_length=self.max_sequence_length + 2,  # including bos and eos tokens of esm tokenizer
            return_tensors="pt"
        )
        sequence_input_ids = tokenized_sequences["input_ids"]
        sequence_attention_mask = tokenized_sequences["attention_mask"]

        # apply chat template
        sequence_lens = sequence_attention_mask.sum(dim=1).tolist()

        if self.include_text_fields: 
            user_messages = [
                # (
                #     (f"Protein name: {fullname}; " if fullname != "unknown" else "")
                #     + (f"Taxon: {taxon}; " if taxon != "unknown" else "") 
                #     + "Sequence embeddings: " + self.placeholder_token * sequence_len
                # )
                (
                    f"Protein name: {fullname}; Taxon: {taxon}; "
                    + "Sequence embeddings: " + self.placeholder_token * sequence_len
                )
                for fullname, taxon, sequence_len in zip(fullnames, taxons, sequence_lens)
            ]
        else: 
            user_messages = [
                "Sequence embeddings: " + self.placeholder_token * sequence_lens
                for sequence_lens in sequence_lens
            ]

        prompt_conversations = [
            [
                {"role": "system", "content": self.system_message}, 
                {"role": "user", "content": user_message}
            ]
            for user_message in user_messages
        ]

        # tokenize prompts
        self.description_tokenizer.padding_side = "left"
        tokenized_prompts = self.description_tokenizer.apply_chat_template(
            prompt_conversations, 
            add_generation_prompt=True, 
            tokenize=True, 
            padding="longest", 
            return_tensors="pt", 
            return_dict=True
        )
        prompt_input_ids = tokenized_prompts["input_ids"]
        prompt_attention_mask = tokenized_prompts["attention_mask"]

        # tokenize descriptions
        self.description_tokenizer.padding_side = "right"
        tokenized_descriptions = self.description_tokenizer(
            [description + self.description_tokenizer.eos_token for description in descriptions], 
            add_special_tokens=False,  # do not add bos token to the beginning
            truncation=True, 
            padding="longest", 
            max_length=self.max_description_length, 
            return_tensors="pt"
        )
        description_input_ids = tokenized_descriptions["input_ids"]
        description_attention_mask = tokenized_descriptions["attention_mask"]

        # truncate descriptions
        if description_input_ids.size(1) > self.max_description_length:
            description_input_ids = description_input_ids[:, :self.max_description_length]
            description_attention_mask = description_attention_mask[:, :self.max_description_length]

        # prepare labels
        labels = description_input_ids.clone()
        labels[description_attention_mask == 0] = -100

        # assemble
        if self.mode == "train": 
            return {
                "name": accessions,
                "protein_input_ids": sequence_input_ids, 
                "protein_attention_mask": sequence_attention_mask, 
                "input_ids": torch.cat([
                    prompt_input_ids, 
                    description_input_ids, 
                ], dim=1), 
                "attention_mask": torch.cat([
                    prompt_attention_mask, 
                    description_attention_mask, 
                ], dim=1),
                "labels": torch.cat([
                    torch.full_like(
                        prompt_input_ids, 
                        fill_value=-100, 
                    ), 
                    labels,
                ], dim=1), 
                "description_input_ids": description_input_ids,
                "description_attention_mask": description_attention_mask
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
