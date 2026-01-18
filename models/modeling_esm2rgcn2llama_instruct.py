"""
Configuration class for the assembled Esm2LlamaInstructForCausalLM model. 

Esm2Rgcn2LlamaInstructForCausalLM = EsmModel + RgcnAdapter + LlamaForCausalLM

For training/evaluation under teacher-forcing scenario, the model `forward` 
function shall take following arguments: 
    * input_ids: (bsz, prompt_len+description_len)  # whole chat template
    * attention_mask: (bsz, prompt_len+description_len)  # left & right padding
    * position_ids: (bsz, prompt_len+description_len)  # optional
    * past_key_values: None
    * labels: (bsz, prompt_len+description_len)  # -100 for padding & prompt
    * protein_input_ids: (bsz, prot_seq_len)  # either ids or embeds
    * protein_attention_mask: (bsz, prot_seq_len)  # left padding
    * protein_position_ids: (bsz, prot_seq_len)  # optional
    * protein_head_mask: (num_heads,) or (num_layers, num_heads)  # optional
    * protein_inputs_embeds: (bsz, prot_seq_len, hidden_size)  # optional
    * graph_edge_index: (2, sum_num_edges)
    * graph_edge_type: (sum_num_edges,)
    * graph_batch: (sum_num_nodes,)  # optional
    * use_cache: False
    * return_decoder_inputs: False

For inference, the model `generate` function shall take following arguments: 
    * inputs: (bsz, prompt_len)  # prompt part of chat template
    * attention_mask: (bsz, prompt_len)  # left padding
    * protein_input_ids: (bsz, prot_seq_len)  # either ids or embeds
    * protein_attention_mask: (bsz, prot_seq_len)  # left padding
    * protein_inputs_embeds: (bsz, prot_seq_len, hidden_size)  # optional
    * graph_edge_index: (2, sum_num_edges)
    * graph_edge_type: (sum_num_edges,)
    * graph_batch: (sum_num_nodes,)  # optional
"""


from typing import Optional, Tuple, Union

import torch
import torch_geometric
import torch_geometric.backend
import torch_geometric.nn
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import index_sort, scatter
from torch_geometric.utils.sparse import index2ptr
from transformers import Cache, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.esm.modeling_esm import EsmModel
from transformers.models.llama import LlamaForCausalLM

from .configuration_esm2rgcn2llama_instruct import (
    RgcnAdapterConfig, 
    Esm2Rgcn2LlamaInstructConfig
)


class RgcnConvLayer(torch_geometric.nn.RGCNConv):
    """Modified `torch_geometric.nn.RGCNConv` Layer for Flexible Precision."""
    def forward(
            self,
            x: Union[OptTensor, Tuple[OptTensor, torch.Tensor]],
            edge_index: Adj,
            edge_type: OptTensor = None
    ):
        """Forward pass of the RGCN layer."""
        # handle input node features
        x_l = x[0] if isinstance(x, tuple) else x  # left node features
        if x_l is None:  # default to indices if no features provided
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r = x[1] if isinstance(x, tuple) else x_l  # right node features

        # define the size of the input graph
        size = (x_l.size(0), x_r.size(0))
        # extract edge types for SparseTensor input
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # initialize the output tensor, specify additional dtype for flexible precision
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device, dtype=x.dtype)

        # handle weight decomposition (basis or block diagonal)
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels
            )

        if self.num_blocks is not None:  # Block-diagonal-decomposition
            if not torch.is_floating_point(x_r) and self.num_blocks is not None:
                raise ValueError(
                    'Block-diagonal decomposition not supported for non-continuous input features.'
                )
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, i == edge_type)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out = out + h.contiguous().view(-1, self.out_channels)

        else:  # no decomposition (standard weight handling)
            use_segment_matmul = torch_geometric.backend.use_segment_matmul

            # heuristic for enabling `segment_matmul` optimization
            if use_segment_matmul is None:
                segment_count = scatter(
                    torch.ones_like(edge_type), 
                    edge_type, 
                    dim_size=self.num_relations
                )
                self._use_segment_matmul_heuristic_output = (
                    torch_geometric.backend.use_segment_matmul_heuristic(
                        num_segments=self.num_relations,
                        max_segment_size=int(segment_count.max()),
                        in_channels=self.weight.size(1),
                        out_channels=self.weight.size(2),
                    )
                )
                assert self._use_segment_matmul_heuristic_output is not None
                use_segment_matmul = self._use_segment_matmul_heuristic_output

            # segment matmul optimization
            if (
                    use_segment_matmul
                    and torch_geometric.typing.WITH_SEGMM
                    and not torch_geometric.is_compiling()
                    and self.num_bases is None
                    and x_l.is_floating_point()
                    and isinstance(edge_index, torch.Tensor)
            ):
                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = index_sort(edge_type, max_value=self.num_relations)
                        edge_index = edge_index[:, perm]
                edge_type_ptr = index2ptr(edge_type, self.num_relations)
                out = self.propagate(edge_index, x=x_l, edge_type_ptr=edge_type_ptr, size=size)

            else:  # loop through relations without optimization
                for i in range(self.num_relations):
                    tmp = masked_edge_index(edge_index, i == edge_type)

                    if not torch.is_floating_point(x_r):
                        out = out + self.propagate(
                            tmp, 
                            x=weight[i, x_l], 
                            edge_type_ptr=None, 
                            size=size
                        )
                    else:
                        h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                        out = out + (h @ weight[i])

        # incorporate root embeddings
        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        # add bias if applicable
        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(self) -> torch.Tensor:
        """Placeholder for edge feature updates (if applicable)."""
        pass


class RgcnAdapter(PreTrainedModel):
    """
    Relational Graph Convolutional Network adapter to match the hidden size of 
    different modalities.
    """
    config_class = RgcnAdapterConfig  # configuration class for this model

    def __init__(self, config: RgcnAdapterConfig):
        super().__init__(config)
        self.config = config
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=config.dropout_rate)
        self.fc1 = torch.nn.Linear(config.input_dim, config.intermediate_dim)
        self.rgcn_layers = torch.nn.ModuleList([
            RgcnConvLayer(
                in_channels=config.intermediate_dim,
                out_channels=config.intermediate_dim,
                num_relations=config.n_relations,
            )
            for _ in range(config.n_layers)
        ])
        self.fc2 = torch.nn.Linear(config.intermediate_dim, config.output_dim)
        self.post_init()  # initialize weights and apply final processing

    def forward(
            self, 
            hidden_states: torch.FloatTensor,  # (bsz, seq_len, input_dim)
            attention_mask: torch.LongTensor,  # (bsz, seq_len)
            edge_index: torch.LongTensor,  # (2, sum_num_edges)
            edge_type: torch.LongTensor,  # (sum_num_edges,)
            batch: Optional[torch.LongTensor] = None  # (sum_num_nodes,)
    ) -> torch.FloatTensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)  # (bsz, seq_len, interm_dim)

        # create mask for node embeddings to be updated by RGCN layers
        nodes_mask = attention_mask.clone().bool()  # (bsz, seq_len)
        nodes_mask[:, 0] = False  # exclude bos token
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(0, batch_size)  # (bsz,)
        eos_indices = attention_mask.sum(dim=1) - 1  # (bsz,)
        nodes_mask[batch_indices, eos_indices] = False  # exclude eos token

        # compute RGCN layers on node embeddings only
        nodes_hidden_states = hidden_states[nodes_mask]  # (sum_num_nodes, interm_dim)
        for layer in self.rgcn_layers:
            nodes_hidden_states = layer(nodes_hidden_states, edge_index, edge_type)
            nodes_hidden_states = self.activation(nodes_hidden_states)
            nodes_hidden_states = self.dropout(nodes_hidden_states)

        # update node hidden states with RGCN outputs
        hidden_states[nodes_mask] = nodes_hidden_states

        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        return hidden_states  # (bsz, seq_len, output_dim)


class Esm2Rgcn2LlamaInstructForCausalLM(PreTrainedModel):
    """
    Esm2Rgcn2LlamaInstructForCausalLM model for protein function prediction.
    Similar to `EncoderDecoderModel` but with more complicated architecture.
    Initialize with either a configuration OR all three components.
    `kwargs` can override standalone attributes in `Esm2Rgcn2LlamaInstructConfig`.
    """
    config_class = Esm2Rgcn2LlamaInstructConfig  # configuration class for this model

    def __init__(
            self, 
            config: Optional[Esm2Rgcn2LlamaInstructConfig] = None, 
            esm_encoder: Optional[EsmModel] = None, 
            adapter: Optional[RgcnAdapter] = None,
            llama_decoder: Optional[LlamaForCausalLM] = None, 
            **kwargs
        ):
        if config is not None:  # components ignored if config is provided
            super().__init__(config)
            self.esm_encoder = EsmModel(
                config.esm_config, 
                add_pooling_layer=False
            )
            self.adapter = RgcnAdapter(config.adapter_config)
            self.llama_decoder = LlamaForCausalLM(config.llama_config)
        else: 
            config = Esm2Rgcn2LlamaInstructConfig(
                esm_config=esm_encoder.config,
                adapter_config=adapter.config,
                llama_config=llama_decoder.config, 
                **kwargs  # override standalone attributes
            ) 
            super().__init__(config)
            self.esm_encoder = esm_encoder
            self.adapter = adapter
            self.llama_decoder = llama_decoder
            
    def prepare_decoder_inputs(
            self, 
            input_ids: torch.LongTensor,
            encoder_hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None, 
    ): 
        """
        Embed and replace placeholder in `input_ids` by encoder hidden states.
        `input_ids` must be passed to locate placeholder for replacement.
        """
        # preparation
        batch_size, seq_len = input_ids.size()
        _, encoder_seq_len, _ = encoder_hidden_states.size()
        if attention_mask is None: 
            attention_mask = torch.ones(
                (batch_size, seq_len), 
                dtype=torch.long, 
                device=input_ids.device
            )
        if encoder_attention_mask is None: 
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_seq_len), 
                dtype=torch.long, 
                device=encoder_hidden_states.device
            )
        inputs_embeds = self.llama_decoder.get_input_embeddings()(input_ids)
        # replacement
        placeholder_mask = input_ids == self.config.placeholder_id
        encoder_mask = encoder_attention_mask.bool()
        inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
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
            # graph-related inputs
            graph_edge_index: Optional[torch.LongTensor] = None,
            graph_edge_type: Optional[torch.LongTensor] = None,
            graph_batch: Optional[torch.LongTensor] = None,
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
        adapter_output = self.adapter(
            hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            edge_index=graph_edge_index,
            edge_type=graph_edge_type,
            batch=graph_batch 
        )
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
        graph_edge_index: Optional[torch.LongTensor] = None,
        graph_edge_type: Optional[torch.LongTensor] = None,
        graph_batch: Optional[torch.LongTensor] = None,
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
            graph_edge_index=graph_edge_index,
            graph_edge_type=graph_edge_type,
            graph_batch=graph_batch,
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
            