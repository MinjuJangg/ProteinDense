"""
DataLoader class for protein function prediction instruction tuning. To be used 
with Prot2TextInstructDataset for Esm2LlamaInstructForCausalLM. 

Every batch from DataLoader will contain following attributes:
    * Training mode (train-eval with teacher-forcing): 
        - graph related features: 
            - x: (sum_num_nodes, num_node_features)
            - edge_index: (2, sum_num_edges)
            - edge_type: (sum_num_edges,)
            - batch: (sum_num_nodes,)
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
            - x: (sum_num_nodes, num_node_features)
            - edge_index: (2, sum_num_edges)
            - edge_type: (sum_num_edges,)
            - batch: (sum_num_nodes,)
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
>>> from transformers import AutoTokenizer
>>> from dataset import Prot2TextInstructDataset, Prot2TextInstructDataLoader
>>> esm_tokenizer = AutoTokenizer.from_pretrained("/data/esm2_t33_650M_UR50D")
>>> llama_tokenizer = AutoTokenizer.from_pretrained(
        "/data/Meta-Llama-3.1-8B-Instruct-hf", 
        pad_token='<|reserved_special_token_0|>'
    )
>>> train_dataset = Prot2TextInstructDataset(
        root_dir="/data/Prot2Text-Llama3-Data/train", 
        csv_path="./data/train.csv", 
        sequence_tokenizer=esm_tokenizer, 
        description_tokenizer=llama_tokenizer, 
        skip_download=True,  # assume data is already downloaded
        skip_reload=True,  # assume data is already preprocessed
    )
>>> train_dataloader = Prot2TextInstructDataLoader(
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


class Prot2TextInstructCollater(torch_geometric.loader.dataloader.Collater): 
    def __init__(
            self, 
            dataset: Prot2TextInstructDataset,
            tokenizer: PreTrainedTokenizer, 
            mode: Literal["train", "inference"],
            **kwargs, 
    ):
        super().__init__(dataset=dataset, **kwargs)
        self.tokenizer = tokenizer
        self.mode = mode
        self.seq_pad_token_id = self.dataset.sequence_tokenizer.pad_token_id
        self.text_pad_token_id = tokenizer.pad_token_id

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

        # prepare attn mask, right pad and stack sequences
        sequence_input_ids = [data["sequence_input_ids"][0] for data in batch]
        pad_sequence_input_ids = self._pad_sequence(
            sequence_input_ids, 
            padding_value=self.seq_pad_token_id, 
            padding_side="right"
        )
        pad_sequence_attention_mask = self._pad_sequence(
            [torch.ones_like(data["sequence_input_ids"][0]) for data in batch], 
            padding_value=0, 
            padding_side="right"
        )

        # prepare attn mask, left pad and stack prompts
        prompt_input_ids = [data["prompt_input_ids"][0] for data in batch]
        pad_prompt_input_ids = self._pad_sequence(
            prompt_input_ids, 
            padding_value=self.text_pad_token_id, 
            padding_side="left"
        )
        pad_prompt_attention_mask = self._pad_sequence(
            [torch.ones_like(data["prompt_input_ids"][0]) for data in batch], 
            padding_value=0, 
            padding_side="left"
        )

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
            })
        elif self.mode == "inference":
            data_batch.update({
                "input_ids": pad_prompt_input_ids, 
                "attention_mask": pad_prompt_attention_mask, 
                "description_input_ids": pad_description_input_ids,
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,
            })
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # remove excluded keys
        if self.exclude_keys: 
            data_batch = {
                k: v for k, v in data_batch.items() 
                if k not in self.exclude_keys
            }

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


class Prot2TextInstructDataLoader(torch.utils.data.DataLoader): 
    """
    DataLoader class proteins, forming batch inputs.
    
    (1) Compose graph related features, 
    (2) dynamically pad sequences, prompts and descriptions, then
    (3) stack then concatenate these text features under different modes.

    Args: 
        dataset: 
            `Prot2TextInstructDataset` class to load data from.
        mode: 
            - "train": training-evaluation with teacher-forcing. Input ids will 
                be concatenated with labels (system + user + assistant) for 
                training.
            - "inference": iterative generation. Input ids will only contain 
                prompt (system + user) for generation.
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
            **kwargs, 
    ):
        # override collate_fn
        kwargs.pop("collate_fn", None)
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        collater = Prot2TextInstructCollater(
            dataset=dataset,
            tokenizer=dataset.description_tokenizer, 
            mode=mode, 
            follow_batch=follow_batch,
            exclude_keys=exclude_keys
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
