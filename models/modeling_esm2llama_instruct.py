"""
Configuration class for the assembled Esm2LlamaInstructForCausalLM model. 

Esm2LlamaInstructForCausalLM = EsmModel + ModalityAdapter + LlamaForCausalLM

For training/evaluation under teacher-forcing scenario, the model `forward` 
function shall take following arguments: 
    * input_ids: (bsz, prompt_len+description_len)  # whole chat template
    * attention_mask: (bsz, prompt_len+description_len)  # left & right padding
    * position_ids: (bsz, prompt_len+description_len)  # optional
    * past_key_values: None
    * labels: (bsz, prompt_len+description_len)  # -100 for padding & prompt
    * protein_input_ids: (bsz, prot_seq_len)  # either ids or embeds
    * protein_attention_mask: (bsz, prot_seq_len)  # right padding
    * protein_position_ids: (bsz, prot_seq_len)  # optional
    * protein_head_mask: (num_heads,) or (num_layers, num_heads)  # optional
    * protein_inputs_embeds: (bsz, prot_seq_len, hidden_size)  # optional
    * use_cache: False
    * return_decoder_inputs: False

For inference, the model `generate` function shall take following arguments: 
    * inputs: (bsz, prompt_len)  # prompt part of chat template
    * attention_mask: (bsz, prompt_len)  # left padding
    * protein_input_ids: (bsz, prot_seq_len)  # either ids or embeds
    * protein_attention_mask: (bsz, prot_seq_len)  # right padding
    * protein_inputs_embeds: (bsz, prot_seq_len, hidden_size)  # optional
"""


from typing import Optional, Tuple, Union

import torch
from transformers import Cache, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.esm.modeling_esm import EsmModel
from transformers.models.llama import LlamaForCausalLM

from .configuration_esm2llama_instruct import (
    ModalityAdapterConfig, 
    Esm2LlamaInstructConfig
)


class ModalityAdapter(PreTrainedModel):
    """2-layer adapter to match the hidden size of different modalities."""
    config_class = ModalityAdapterConfig  # configuration class for this model

    def __init__(self, config: ModalityAdapterConfig):
        super().__init__(config)
        self.config = config
        self.fc1 = torch.nn.Linear(config.input_dim, config.intermediate_dim)
        self.fc2 = torch.nn.Linear(config.intermediate_dim, config.output_dim)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=config.dropout_rate)
        self.ln1 = torch.nn.LayerNorm(normalized_shape=config.intermediate_dim)  # DEPRECATED
        self.ln2 = torch.nn.LayerNorm(normalized_shape=config.output_dim)  # DEPRECATED
        self.post_init()  # initialize weights and apply final processing

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        
        hidden_states = hidden_states.to(self.fc1.weight.device, non_blocking=True)  #################%%%%###################
        # input: (bsz, seq_len, input_dim)
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        # interm: (bsz, seq_len, interm_dim)
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        return hidden_states  # (bsz, seq_len, output_dim)


class Esm2LlamaInstructForCausalLM(PreTrainedModel):
    """
    Esm2LlamaInstructForCausalLM model for protein function prediction.
    Similar to `EncoderDecoderModel` but with more complicated architecture.
    Initialize with either a configuration OR all three components.
    `kwargs` can override standalone attributes in `Esm2LlamaInstructConfig`.
    """
    config_class = Esm2LlamaInstructConfig  # configuration class for this model

    def __init__(
            self, 
            config: Optional[Esm2LlamaInstructConfig] = None, 
            esm_encoder: Optional[EsmModel] = None, 
            adapter: Optional[ModalityAdapter] = None,
            llama_decoder: Optional[LlamaForCausalLM] = None, 
            **kwargs
        ):
        if config is not None:  # components ignored if config is provided
            super().__init__(config)
            self.esm_encoder = EsmModel(
                config.esm_config, 
                add_pooling_layer=False
            )
            self.adapter = ModalityAdapter(config.adapter_config)
            self.llama_decoder = LlamaForCausalLM(config.llama_config)
        else: 
            config = Esm2LlamaInstructConfig(
                esm_config=esm_encoder.config,
                adapter_config=adapter.config,
                llama_config=llama_decoder.config, 
                **kwargs  # override standalone attributes
            ) 
            super().__init__(config)
            self.esm_encoder = esm_encoder
            self.adapter = adapter
            self.llama_decoder = llama_decoder
            
    # def prepare_decoder_inputs(
    #         self, 
    #         input_ids: torch.LongTensor,
    #         encoder_hidden_states: torch.FloatTensor,
    #         attention_mask: Optional[torch.LongTensor] = None,
    #         encoder_attention_mask: Optional[torch.LongTensor] = None, 
    # ): 
    #     """
    #     Embed and replace placeholder in `input_ids` by encoder hidden states.
    #     `input_ids` must be passed to locate placeholder for replacement.
    #     """
    #     # preparation
    #     batch_size, seq_len = input_ids.size()
    #     _, encoder_seq_len, _ = encoder_hidden_states.size()
    #     if attention_mask is None: 
    #         attention_mask = torch.ones(
    #             (batch_size, seq_len), 
    #             dtype=torch.long, 
    #             device=input_ids.device
    #         )
    #     if encoder_attention_mask is None: 
    #         encoder_attention_mask = torch.ones(
    #             (batch_size, encoder_seq_len), 
    #             dtype=torch.long, 
    #             device=encoder_hidden_states.device
    #         )
    #     inputs_embeds = self.llama_decoder.get_input_embeddings()(input_ids)
    #     # replacement
    #     placeholder_mask = input_ids == self.config.placeholder_id
    #     encoder_mask = encoder_attention_mask.bool()
    #     inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
    #     return inputs_embeds, attention_mask
    def prepare_decoder_inputs(
            self, 
            input_ids: torch.LongTensor,
            encoder_hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None, 
    ): 
        """
        Embed and replace placeholder in `input_ids` by encoder hidden states.
        Hugging Face 원본 스타일 유지 + dev로 통일.
        """
        # 기준 dev = LLaMA 임베딩이 올라간 디바이스
        dev = self.llama_decoder.get_input_embeddings().weight.device

        batch_size, seq_len = input_ids.size()
        _, encoder_seq_len, _ = encoder_hidden_states.size()

        # attention mask 기본값
        if attention_mask is None: 
            attention_mask = torch.ones(
                (batch_size, seq_len), 
                dtype=torch.long, 
                device=input_ids.device   # 원본 코드 스타일 유지
            )

        # encoder attention mask 기본값
        if encoder_attention_mask is None: 
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_seq_len), 
                dtype=torch.long, 
                device=encoder_hidden_states.device  # 원본 코드 스타일 유지
            )

        # inputs_embeds 생성
        inputs_embeds = self.llama_decoder.get_input_embeddings()(input_ids)

        # placeholder 위치 교체
        placeholder_mask = input_ids == self.config.placeholder_id
        encoder_mask = encoder_attention_mask.bool()
        inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]

        # ✅ 마지막에 전부 dev로 맞춤 (CPU/GPU mismatch 방지)
        input_ids = input_ids.to(dev, non_blocking=True)
        encoder_hidden_states = encoder_hidden_states.to(dev, non_blocking=True)
        attention_mask = attention_mask.to(dev, non_blocking=True)
        encoder_attention_mask = encoder_attention_mask.to(dev, non_blocking=True)
        inputs_embeds = inputs_embeds.to(dev, non_blocking=True)

        return inputs_embeds, attention_mask


    def forward(
            self, 
            # chat template text inputs
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            labels: Optional[torch.LongTensor] = None,
            # protein amino-acid sequence inputs
            protein_input_ids: Optional[torch.LongTensor] = None,
            protein_attention_mask: Optional[torch.LongTensor] = None,
            protein_position_ids: Optional[torch.LongTensor] = None, 
            protein_head_mask: Optional[torch.LongTensor] = None,
            protein_inputs_embeds: Optional[torch.FloatTensor] = None,
            # behavior control arguments
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_encoder_outputs: bool = False,
            return_adapter_outputs: bool = False, 
            return_decoder_inputs: bool = False,
            cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]: 
        """
        Compute encoder and adapter outputs, then pass to decoder.
        `input_ids` is expected to be [prompt + description] in teacher-forcing 
        scenario and [prompt] only in first iteration of inference (with 
        return_decoder_inputs=True). 
        Attention: possible concatenation of the mask and labels should be 
        handled before calling this method.
        `inputs_embeds` not allowed due to placeholder replacement scheme. 
        """
        # esm_encoder forward
        encoder_output = self.esm_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            position_ids=protein_position_ids,
            head_mask=protein_head_mask,
            inputs_embeds=protein_inputs_embeds,
            use_cache=False, # because config.esm_config.is_decoder=False
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        encoder_hidden_states = encoder_output[0]
        encoder_attention_mask = protein_attention_mask
        if return_encoder_outputs:
            return encoder_output
        # adapter forward
        adapter_output = self.adapter(encoder_hidden_states)
        if return_adapter_outputs:
            return adapter_output, encoder_attention_mask
        # decoder input preparation
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids, 
            encoder_hidden_states=adapter_output, 
            attention_mask=attention_mask, 
            encoder_attention_mask=encoder_attention_mask, 
        )
        if return_decoder_inputs:
            return inputs_embeds, attention_mask
        # llama_decoder forward
        return self.llama_decoder.forward(
            input_ids=None,
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            return_dict=return_dict, 
            cache_position=cache_position
        )

    def generate(
        self,
        inputs: torch.LongTensor,  # alias of `input_ids`
        attention_mask: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Do inference based on given input prompt. 
        `inputs` is expected to be [prompt] only. 
        Output will not keep the input prompt due to input in form of embeds.
        Generation behavior can be controlled by `args` and `kwargs`, read 
        `GenerationMixin.generate` for more info. 
        """
        # get decoder inputs
        prompt_inputs_embeds, prompt_attention_mask = self(
            input_ids=inputs, 
            attention_mask=attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            protein_inputs_embeds=protein_inputs_embeds,
            use_cache=False, 
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            return_decoder_inputs=True
        )
        # do generate on llama_decoder
        return self.llama_decoder.generate(
            inputs_embeds=prompt_inputs_embeds, 
            attention_mask=prompt_attention_mask, 
            **kwargs
        )

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for all submodules that support it.
        Attention! Model need to be in train mode before calling this method.
        """
        if hasattr(self.esm_encoder, "gradient_checkpointing_enable"):
            self.esm_encoder.gradient_checkpointing_enable()
        if hasattr(self.llama_decoder, "gradient_checkpointing_enable"):
            self.llama_decoder.gradient_checkpointing_enable()
        # simple adapter no need to implement gradient checkpointing

    def gradient_checkpointing_disable(self):
        if hasattr(self.esm_encoder, "gradient_checkpointing_disable"):
            self.esm_encoder.gradient_checkpointing_disable()
        if hasattr(self.llama_decoder, "gradient_checkpointing_disable"):
            self.llama_decoder.gradient_checkpointing_disable()
            