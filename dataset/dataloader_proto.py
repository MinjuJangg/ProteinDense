# import torch
# import torch.utils.data
# from typing import Dict, List, Optional, Literal, Union
# from transformers import PreTrainedTokenizer
import torch
import torch.utils.data
from typing import Dict, List, Optional, Literal, Union, Any
from transformers import PreTrainedTokenizer


# ==========================
# Collate Function
# ==========================
class Prot2TextInstructJSONLCollater:


    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "inference"],
        seq_pad_token_id: int,
        text_pad_token_id: int,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.seq_pad_token_id = seq_pad_token_id
        self.text_pad_token_id = text_pad_token_id
        self.exclude_keys = exclude_keys or []

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1) Sequence
        sequence_input_ids = [b["sequence_input_ids"].squeeze(0).long() for b in batch]
        pad_sequence_input_ids = self._pad(sequence_input_ids, self.seq_pad_token_id)
        pad_sequence_attention_mask = self._pad(
            [torch.ones_like(x) for x in sequence_input_ids],
            padding_value=0,
        )

        # 2) Prompt (left-pad)
        prompt_input_ids = [b["prompt_input_ids"].squeeze(0) for b in batch]
        pad_prompt_input_ids = self._pad(prompt_input_ids, self.text_pad_token_id, "left")
        pad_prompt_attention_mask = self._pad(
            [torch.ones_like(x) for x in prompt_input_ids],
            padding_value=0,
            padding_side="left",
        )

        # 3) Description
        description_input_ids = [b["description_input_ids"].squeeze(0) for b in batch]
        pad_description_input_ids = self._pad(description_input_ids, self.text_pad_token_id)
        pad_description_attention_mask = self._pad(
            [torch.ones_like(x) for x in description_input_ids],
            padding_value=0,
        )
        pad_labels = self._pad(description_input_ids, padding_value=-100)

        # --------------------------
        # assemble
        # --------------------------
        if self.mode == "train":
            out = {
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,

                "input_ids": torch.cat([pad_prompt_input_ids, pad_description_input_ids], dim=1),
                "attention_mask": torch.cat([pad_prompt_attention_mask, pad_description_attention_mask], dim=1),

                "labels": torch.cat(
                    [
                        torch.full_like(pad_prompt_input_ids, -100),
                        pad_labels,
                    ],
                    dim=1,
                ),
            }
        else:  # inference
            out = {
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,
                "input_ids": pad_prompt_input_ids,
                "attention_mask": pad_prompt_attention_mask,
            }


        out["accession"] = [b["accession"] for b in batch]
        out["parent_accession"] = [b["parent_accession"] for b in batch]

        # remove unwanted keys
        if self.exclude_keys:
            out = {k: v for k, v in out.items() if k not in self.exclude_keys}

        return out

    # pad utility
    def _pad(self, seqs, padding_value: int, padding_side="right"):
        max_len = max(s.size(-1) for s in seqs)
        pads = []
        for s in seqs:
            pad_len = max_len - s.size(-1)
            if pad_len == 0:
                pads.append(s)
                continue
            pad_tensor = torch.full((pad_len,), padding_value, dtype=s.dtype)
            if padding_side == "left":
                pads.append(torch.cat([pad_tensor, s], dim=0))
            else:
                pads.append(torch.cat([s, pad_tensor], dim=0))
        return torch.stack(pads, dim=0)


# ==========================
# DataLoader class
# ==========================
class Prot2TextInstructDataLoaderJSONL(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset,
        mode: str = "train",
        batch_size: int = 1,
        shuffle: bool = True,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)

        collater = Prot2TextInstructJSONLCollater(
            tokenizer=dataset.description_tokenizer,
            mode=mode,
            seq_pad_token_id=dataset.sequence_tokenizer.pad_token_id,
            text_pad_token_id=dataset.description_tokenizer.pad_token_id,
            exclude_keys=exclude_keys,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collater,
            **kwargs,
        )
