import torch
import torch.utils.data
from typing import Dict, List, Literal, Optional, Union
from transformers import PreTrainedTokenizer


class Prot2TextInstructJSONLCollater:
    """
    Collate function for Prot2TextInstructDatasetJSONL (NoGraph version).

    원본 Prot2TextInstructCollater와 동일한 텍스트 패딩/결합 로직 및
    출력 키셋을 유지하되, 그래프 관련 기능만 제거했습니다.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "inference"],
        seq_pad_token_id: int,
        text_pad_token_id: int,
        follow_batch: Optional[List[str]] = None,   # 인터페이스 호환용 (미사용)
        exclude_keys: Optional[List[str]] = None,   # 원본과 동일하게 최종 단계에서 필터링
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.seq_pad_token_id = int(seq_pad_token_id)
        self.text_pad_token_id = int(text_pad_token_id)
        self.exclude_keys = exclude_keys or []

        # 안전장치: pad token 존재 여부
        if self.seq_pad_token_id is None:
            raise ValueError("sequence_tokenizer must have a valid pad_token_id.")
        if self.text_pad_token_id is None:
            raise ValueError("description_tokenizer must have a valid pad_token_id.")

    def __call__(self, batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # ---------------------------
        # (1) Sequence (protein) - right pad
        # ---------------------------
        # 입력은 각 항목이 (1, L) 형태라고 가정 -> squeeze(0)로 (L,)
        sequence_input_ids = [b["sequence_input_ids"].squeeze(0).long() for b in batch]
        pad_sequence_input_ids = self._pad_sequence(sequence_input_ids, self.seq_pad_token_id, "right")
        pad_sequence_attention_mask = self._pad_sequence(
            [torch.ones_like(x, dtype=torch.long) for x in sequence_input_ids],
            padding_value=0,
            padding_side="right",
        )

        # ---------------------------
        # (2) Prompt - left pad
        # ---------------------------
        prompt_input_ids = [b["prompt_input_ids"].squeeze(0).long() for b in batch]
        pad_prompt_input_ids = self._pad_sequence(prompt_input_ids, self.text_pad_token_id, "left")
        pad_prompt_attention_mask = self._pad_sequence(
            [torch.ones_like(x, dtype=torch.long) for x in prompt_input_ids],
            padding_value=0,
            padding_side="left",
        )

        # ---------------------------
        # (3) Description - right pad
        # ---------------------------
        description_input_ids = [b["description_input_ids"].squeeze(0).long() for b in batch]
        pad_description_input_ids = self._pad_sequence(description_input_ids, self.text_pad_token_id, "right")
        pad_description_attention_mask = self._pad_sequence(
            [torch.ones_like(x, dtype=torch.long) for x in description_input_ids],
            padding_value=0,
            padding_side="right",
        )
        pad_labels = self._pad_sequence(
            description_input_ids, padding_value=-100, padding_side="right"
        )

        # ---------------------------
        # (4) Teacher Forcing or Inference
        # ---------------------------
        if self.mode == "train":
            out = {
                "input_ids": torch.cat([pad_prompt_input_ids, pad_description_input_ids], dim=1),
                "attention_mask": torch.cat([pad_prompt_attention_mask, pad_description_attention_mask], dim=1),
                "labels": torch.cat(
                    [
                        torch.full_like(pad_prompt_input_ids, fill_value=-100),
                        pad_labels,
                    ],
                    dim=1,
                ),
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,
                "description_input_ids": pad_description_input_ids,
                "description_attention_mask": pad_description_attention_mask,
            }
        elif self.mode == "inference":
            # 원본과 동일: labels, description_attention_mask 미포함
            out = {
                "input_ids": pad_prompt_input_ids,
                "attention_mask": pad_prompt_attention_mask,
                "description_input_ids": pad_description_input_ids,
                "protein_input_ids": pad_sequence_input_ids,
                "protein_attention_mask": pad_sequence_attention_mask,
            }
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # ---------------------------
        # (5) exclude_keys 적용 (원본과 동일한 위치/동작)
        # ---------------------------
        if self.exclude_keys:
            out = {k: v for k, v in out.items() if k not in self.exclude_keys}

        return out

    @staticmethod
    def _pad_sequence(
        sequences: List[torch.Tensor],
        padding_value: Union[float, int],
        padding_side: Literal["left", "right"] = "right",
    ) -> torch.Tensor:
        """
        원본 Collater의 _pad_sequence 로직과 동일.
        sequences: 1D long 텐서 리스트
        """
        max_len = max(seq.shape[-1] for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            pad_len = max_len - seq.shape[-1]
            if pad_len < 0:
                raise ValueError("Negative pad length encountered.")
            if pad_len == 0:
                padded_sequences.append(seq)
                continue
            padding = torch.full(
                (pad_len,),
                fill_value=padding_value,
                dtype=seq.dtype,
                device=seq.device,
            )
            if padding_side == "left":
                padded_sequences.append(torch.cat([padding, seq], dim=-1))
            elif padding_side == "right":
                padded_sequences.append(torch.cat([seq, padding], dim=-1))
            else:
                raise ValueError(f"Invalid padding side: {padding_side}")
        return torch.stack(padded_sequences, dim=0)


class Prot2TextInstructDataLoaderJSONL(torch.utils.data.DataLoader):
    """
    DataLoader class for Prot2TextInstructDatasetJSONL (NoGraph version).

    원본 Prot2TextInstructDataLoader와 동일한 인터페이스/동작을 유지하며,
    collater만 JSONL 전용으로 교체했습니다.
    """
    def __init__(
        self,
        dataset,
        mode: Literal["train", "inference"] = "train",
        batch_size: int = 1,
        shuffle: bool = True,
        follow_batch: Optional[List[str]] = None,       # 인터페이스 호환용 (미사용)
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # 원본과 동일: 외부에서 collate_fn 주는 것 방지
        kwargs.pop("collate_fn", None)

        collater = Prot2TextInstructJSONLCollater(
            tokenizer=dataset.description_tokenizer,
            mode=mode,
            seq_pad_token_id=dataset.sequence_tokenizer.pad_token_id,
            text_pad_token_id=dataset.description_tokenizer.pad_token_id,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collater,
            **kwargs,
        )

    def __len__(self):
        """Mark class as Sized."""
        return super().__len__()

    def __iter__(self):
        """Mark class as Iterable."""
        return super().__iter__()
