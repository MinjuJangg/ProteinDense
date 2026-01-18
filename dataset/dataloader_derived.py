"""
Derived DataLoader class for protein function prediction supervised fine tuning. 
To be used with Prot2TextInstructDataset for Esm2LlamaForCausalLM. 

The `Prot2TextInstructDataset` is designed to be preprocessed with an instruct 
model (ex. `Meta-Llama-3.1-8B-Instruct-hf`), but can be adapted to other base 
models (ex. `Llama-3.2-1B`) if tokenizers of both models share the same 
vocabulary. 

This derived version (`Prot2TextDerivedDataLoader`) is thus designed to adapt 
datasets that are already preprocessed for the instruct model. It replaces 
special tokens and reorganize tokenized input ids, making them suitable for the 
base language model.

Every batch from DataLoader will contain following attributes:
    * Training mode (train-eval with teacher-forcing): 
        - graph related features: 
            - x: (sum_num_nodes, num_node_features)
            - edge_index: (2, sum_num_edges)
            - edge_type: (sum_num_edges,)
            - batch: (sum_num_nodes,)
        - amino-acid sequence: 
            - protein_input_ids (bsz, max_seq_len+2)  # bos and eos tokens
            - protein_attention_mask (bsz, max_seq_len+2)  # left padding
        - concatenated prompt and description:
            - input_ids (bsz, prompt_len+1+max_text_len+1)  # bos and eos tokens
            - attention_mask (bsz, prompt_len+1+max_text_len+1)  # right padding
            - labels (bsz, prompt_len+1+max_text_len+1)
        - standalone description for reward model training:
            - description_input_ids (bsz, max_text_len+1)  # eos token only
            - description_attention_mask (bsz, max_text_len+1)  # right padding

        ids       = [bos  + prompt + bos  & description + eos  + right-pad]
        mask      = [1    + 1s     + 1    & 1s          + 1    + 0s       ]
        labels    = [-100 + -100s  + -100 & description + eos  + -100s    ]
        desc_ids  =                     [ & description + eos  + right-pad]
        desc_mask =                     [ & 1s          + 1    + 0s       ]

    * Inference mode (iterative generation):
        - graph related features: 
            - x: (sum_num_nodes, num_node_features)
            - edge_index: (2, sum_num_edges)
            - edge_type: (sum_num_edges,)
            - batch: (sum_num_nodes,)
        - amino-acid sequence: 
            - protein_input_ids (bsz, max_seq_len+2)  # bos and eos tokens
            - protein_attention_mask (bsz, max_seq_len+2)  # left padding
        - prompt with inference head: 
            - input_ids (bsz, prompt_len+1)  # bos token at the end
            - attention_mask (bsz, prompt_len+1)
        - standalone description for evaluation:
            - description_input_ids (bsz, max_text_len+1)
            - description_attention_mask (bsz, max_text_len+1)

        ids       = [bos + prompt + bos & ]
        mask      = [1   + 1s     + 1   & ]
        desc_ids  =                   [ & description + eos + right-pad]
        desc_mask =                   [ & 1s          + 1   + 0s       ]

Example of usage:
>>> from transformers import AutoTokenizer
>>> from dataset import Prot2TextInstructDataset, Prot2TextDerivedDataLoader
>>> esm_tokenizer = AutoTokenizer.from_pretrained("/data/esm2_t33_650M_UR50D")
>>> llama_tokenizer = AutoTokenizer.from_pretrained(
        "/data/Llama-3.2-1B", 
        pad_token='<|reserved_special_token_0|>'
    )
>>> train_dataset = Prot2TextInstructDataset(
        root_dir="/data/Prot2Text-Llama3-Data/train", 
        csv_path="./data/train.csv", 
        sequence_tokenizer=esm_tokenizer, 
        description_tokenizer=llama_tokenizer,  # pass the base model tokenizer
        skip_download=True,  # assume data is already downloaded
        skip_reload=True,  # assume data is already preprocessed
    )
>>> train_dataloader = Prot2TextDerivedDataLoader(
        dataset=train_dataset, 
        mode="train", 
        batch_size=2, 
        shuffle=True, 
    )
"""


from typing import Dict, List, Literal, Optional, Union

import torch
import torch.utils.data
import torch_geometric
import torch_geometric.data
import torch_geometric.loader.dataloader
from transformers import PreTrainedTokenizer

from .dataset import Prot2TextInstructDataset


class Prot2TextDerivedCollater(torch_geometric.loader.dataloader.Collater): 
    def __init__(
            self, 
            dataset: Prot2TextInstructDataset,
            tokenizer: PreTrainedTokenizer, 
            mode: Literal["train", "inference"],
            original_eos_token_id: int, 
            prompt_sentence: str,
            **kwargs, 
    ):
        super().__init__(dataset=dataset, **kwargs)
        self.tokenizer = tokenizer
        self.mode = mode
        self.prompt_sentence = prompt_sentence

        self.prompt_input_ids = tokenizer(
            [tokenizer.bos_token + prompt_sentence + tokenizer.bos_token], 
            add_special_tokens=False, 
            return_tensors="pt", 
            return_attention_mask=False, 
        )["input_ids"]

        self.seq_pad_token_id = dataset.sequence_tokenizer.pad_token_id
        self.text_pad_token_id = tokenizer.pad_token_id
        self.old_text_eos_token_id = original_eos_token_id
        self.new_text_eos_token_id = tokenizer.eos_token_id

    def __call__(
            self, 
            batch: List[Dict[str, Union[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # prepare graph related features and name
        data_batch = torch_geometric.data.Batch.from_data_list(
            batch, 
            exclude_keys=[
                "sequence_input_ids", 
                "prompt_input_ids", 
                "description_input_ids", 
            ]
        )

        # prepare attn mask, left pad and stack sequences
        pad_sequence_input_ids = self._pad_sequence(
            [data["sequence_input_ids"][0] for data in batch], 
            padding_value=self.seq_pad_token_id, 
            padding_side="left"
        )
        pad_sequence_attention_mask = self._pad_sequence(
            [torch.ones_like(data["sequence_input_ids"][0]) for data in batch], 
            padding_value=0, 
            padding_side="left"
        )

        # prepare attn mask and expand prompts
        pad_prompt_input_ids = self.prompt_input_ids.repeat(len(batch), 1).to(
            pad_sequence_input_ids.device
        )
        pad_prompt_attention_mask = torch.ones_like(pad_prompt_input_ids)

        # prepare attn mask, right pad and stack descriptions 
        description_input_ids = [data["description_input_ids"][0] for data in batch]
        pad_description_input_ids = self._pad_sequence(
            description_input_ids, 
            padding_value=self.text_pad_token_id, 
            padding_side="right"
        )
        pad_description_attention_mask = self._pad_sequence(
            [torch.ones_like(data["description_input_ids"][0]) for data in batch], 
            padding_value=0, 
            padding_side="right"
        )
        pad_labels = self._pad_sequence(
            description_input_ids, 
            padding_value=-100,
            padding_side="right"
        )

        # replace special tokens in descriptions
        pad_description_input_ids.masked_fill_(
            pad_description_input_ids == self.old_text_eos_token_id,
            self.new_text_eos_token_id
        )
        pad_labels.masked_fill_(
            pad_labels == self.old_text_eos_token_id,
            self.new_text_eos_token_id
        )

        # decode back the descriptions in text
        descriptions = self.tokenizer.batch_decode(
            description_input_ids,
            skip_special_tokens=True,
        )

        # update text features
        if self.mode == "train":
            data_batch.update({
                "input_ids": torch.cat([
                    pad_prompt_input_ids, 
                    pad_description_input_ids
                ], dim=1), 
                "attention_mask": torch.cat([
                    pad_prompt_attention_mask, 
                    pad_description_attention_mask
                ], dim=1), 
                "labels": torch.cat([
                    torch.full_like(
                        pad_prompt_input_ids, 
                        fill_value=-100
                    ),
                    pad_labels
                ], dim=1),
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,
                "description_input_ids": pad_description_input_ids,
                "description_attention_mask": pad_description_attention_mask,
                "descriptions": descriptions, 
            })
        elif self.mode == "inference":
            data_batch.update({
                "input_ids": pad_prompt_input_ids, 
                "attention_mask": pad_prompt_attention_mask, 
                "description_input_ids": pad_description_input_ids,
                "description_attention_mask": pad_description_attention_mask,
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,
            })
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        return data_batch

    @staticmethod
    def _pad_sequence(
        sequences: List[torch.Tensor],
        padding_value: Union[float, int],
        padding_side: Literal["left", "right"] = "right",
    ) -> torch.Tensor:
        """
        Modified version of torch.nn.utils.rnn.pad_sequence with optional 
        padding side.
        Such feature is naturally supported by PyTorch 2.6.0+ and it's 
        recommended to use the built-in version for better efficiency.
        
        * sequences as input must be a list of 1D tensors. 
        """
        max_len = max(sequence.shape[-1] for sequence in sequences)
        padded_sequences = []
        for sequence in sequences: 
            padding = torch.full(
                size=(max_len - sequence.shape[-1],),
                fill_value=padding_value,
                dtype=sequence.dtype,
                device=sequence.device,
            )
            if padding_side == "left":
                padded_sequences.append(torch.cat([padding, sequence], dim=-1))
            elif padding_side == "right":
                padded_sequences.append(torch.cat([sequence, padding], dim=-1))
            else:
                raise ValueError(f"Invalid padding side: {padding_side}")
        return torch.stack(padded_sequences, dim=0)


class Prot2TextDerivedDataLoader(torch.utils.data.DataLoader): 
    """
    DataLoader class proteins, forming batch inputs.

    (1) Compose graph related features with PyG's Batch.from_data_list;
    (2) dynamically pad sequences, prompts and descriptions; 
    (3) replace special tokens and reorganize tokenized input ids; 
    (4) stack then concatenate these text features under different modes.
    
    Args: 
        dataset: 
            `Prot2TextInstructDataset` class to load data from. Both 
            `Prot2TextInstructDataLoader` and `Prot2TextDerivedDataLoader` should 
            use the same dataset, but with different sequence tokenizers. Read 
            the docstring of `Prot2TextInstructDataset` for more details.
        mode: 
            - "train": training-evaluation with teacher-forcing. Input ids will 
                be concatenated with labels (prompt + description) for 
                training.
            - "inference": iterative generation. Input ids will only contain 
                prompt (prompt + bos) for generation.
        batch_size: 
            Number of samples per batch.
        shuffle: 
            Whether to shuffle the data. If a sampler is provided, the shuffling 
            behavior should be controlled by the sampler and this argument 
            should be set to False.
        follow_batch: 
            PyG specific feature to be passed to Collater class. When working 
            with batched graph data, follow_batch is used to indicate which 
            attributes should an extra batch indices tensor be created for. If 
            not set, an extra `batch` tensor will be automatically created to 
            indicate which graph each node belongs to. 
        exclude_keys: 
            List of keys to exclude from the batch. The exclusion will be applied 
            at the very end of the collation process.
        original_eos_token_id: 
            End-of-sequence token id of the instruct model tokenizer that is used 
            in the preprocessing stage of the dataset. Such token will be 
            replaced by the eos token id of the base model tokenizer that is 
            given in the derived scenario.
        prompt_sentence:
            Prompt sentence to be used in the derived scenario. 
        kwargs: 
            Additional arguments to pass to DataLoader class.
    """
    def __init__(
            self, 
            dataset: Prot2TextInstructDataset, 
            mode: Literal["train", "inference"] = "train",
            batch_size: int = 1, 
            shuffle: bool = True,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            original_eos_token_id: int = 128009,
            prompt_sentence: str = (
                "Predict protein description based on the amino-acid sequence embeddings."
            ),
            **kwargs, 
    ):
        # override collate_fn
        kwargs.pop("collate_fn", None)
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        collater = Prot2TextDerivedCollater(
            dataset=dataset,
            tokenizer=dataset.description_tokenizer, 
            mode=mode, 
            follow_batch=follow_batch,
            exclude_keys=exclude_keys, 
            original_eos_token_id=original_eos_token_id,
            prompt_sentence=prompt_sentence,
        )
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collater, 
            **kwargs
        )

    def __len__(self):
        """Mark class as Sized."""
        return super().__len__()

    def __iter__(self):
        """Mark class as Iterable."""
        return super().__iter__()
