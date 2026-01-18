import inspect
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import Cache, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    CausalLMOutputWithPast, 
    BaseModelOutputWithPoolingAndCrossAttentions
)
from transformers.models.esm.modeling_esm import (
    EsmEmbeddings, 
    EsmEncoder, 
    EsmPreTrainedModel, 
    EsmModel
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig

from .configuration_esm2llama_legacy import Esm2LlamaConfig, EsmEncoderConfig


class EsmEncoderModel(EsmModel):
    """
    EsmModel behaving as an encoder with extra adapter as last step to match 
    hidden size of protein embeddings. The class inherits all attributes and 
    methods from the EsmModel class except pooler and contact prediction not
    available in this encoder model. adapter is the only additional attribute 
    and the last step in the forward pass.

    Args:
        config:
            Configuration for initialization of the model. This configuration 
            is of same attributes as EsmConfig with `decoder_hidden_size` the 
            only additional attribute to indicate hidden size required by the 
            decoder. `is_decoder` must be False to use the model as an encoder.
    """
    config_class = EsmEncoderConfig  # configuration class for this model architecture

    def __init__(self, config: EsmEncoderConfig):
        assert not config.is_decoder, (
            "EsmModel as encoder must be initialized with `is_decoder=False`. "
        )

        EsmPreTrainedModel.__init__(self, config)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)

        # pooler erased to return last hidden state as protein sequence embeddings
        self.pooler = None
        # contact head erased to avoid error if number of hidden layers modified 
        # when loading from pretrained
        self.contact_head = None

        # add adapter to match output hidden size of the encoder and the input 
        # hidden size of the decoder
        self.adapter = (
            torch.nn.Linear(
                in_features=config.hidden_size, 
                out_features=config.decoder_hidden_size
            )
            if (
                config.decoder_hidden_size is not None 
                and config.decoder_hidden_size != config.hidden_size
            ) else None
        )
        self.post_adapter_layernorm = (
            torch.nn.LayerNorm(
                normalized_shape=config.decoder_hidden_size, 
                eps=config.layer_norm_eps
            )
            if self.adapter is not None else None
        )

        # initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Modified forward function based on that of EsmModel with extra adapter 
        as the last step if hidden sizes of encoder and decoder are not the same.

        Input keyword arguments are identical to the method with same name of 
        its parent class, read the documentation from `EsmModel.forward` for 
        more information.
        """
        return_dict = (
            kwargs["return_dict"] 
            if kwargs["return_dict"] is not None 
            else self.config.use_return_dict
        )
        esm_model_output = super().forward(**kwargs)
        if self.adapter is None:
            return esm_model_output
        else:
            # sequence output can be accessed by indexing [0] in either case 
            # (Tuple or ModelOutput)
            adapter_output = self.post_adapter_layernorm(self.adapter(esm_model_output[0]))
            if not return_dict:
                return (adapter_output, ) + esm_model_output[1:]
            else:
                return BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=adapter_output,
                    pooler_output=esm_model_output.pooler_output,
                    past_key_values=esm_model_output.past_key_values,
                    hidden_states=esm_model_output.hidden_states,
                    attentions=esm_model_output.attentions,
                    cross_attentions=esm_model_output.cross_attentions,
                )

    def predict_contacts(self, *args, **kwargs):
        """
        Deprecated contact prediction in the encoder because of the potential 
        error while loading from pretrained if number of hidden layers is 
        modified.
        """
        raise NotImplementedError(
            "Contact prediction not available in EsmEncoderModel, use vanilla "
            "EsmModel instead. "
        )


class Esm2LlamaForCausalLM(LlamaForCausalLM):
    """
    Esm-to-Llama model with EsmEncoderModel as protein sequence encoder and 
    LlamaForCausalLM as description generator. The output embeddings of 
    EsmEncoderModel is concatenated with the prompt sentence as the input of 
    LlamaModel in a decoder-only style. This model inherits all attributes and 
    methods from LlamaForCausalLM. esm_encoder is the only additional attribute.

    Args:
        config:
            Configuration for initialization of the model.
        esm_model:
            Pass the EsmEncoderModel as the encoder part of the whole model 
            directly. IF given, the EsmConfig part of passed Esm2LlamaConfig 
            will be ignored, or a LlamaConfig may be passed directly.
    """
    config_class = Esm2LlamaConfig  # configuration class for this model architecture

    def __init__(
            self, 
            config: Union[Esm2LlamaConfig, LlamaConfig], 
            esm_model: Optional[EsmEncoderModel] = None
    ):
        # normal initialization of LlamaForCausalLM as the decoder part
        super().__init__(config)

        # add self.esm_encoder: EsmEncoderModel as extra attribute to LlamaForCausalLM
        if esm_model is None:
            if hasattr(config, "esm_config"):
                # post initialization procedure of esm_encoder will be done 
                # inside the initialization of itself
                self.esm_encoder = EsmEncoderModel(config.esm_config)
            else:
                raise ValueError(
                    "The passed configuration must be an Esm2LlamaConfig if "
                    "EsmEncoderModel is not given. "
                )
        else:
            self.esm_encoder = esm_model
            if not isinstance(self.config, Esm2LlamaConfig):
                self.config = Esm2LlamaConfig(esm_model.config, **self.config.__dict__)
            else:
                self.config.esm_config = esm_model.config

        self.post_init()

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            pretrained_esm_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            pretrained_llama_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            esm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> "Esm2LlamaForCausalLM":
        """
        Instantiates a Esm2LlamaForCausalLM from either (1) a pretrained 
        Esm-to-Llama model, or (2) a pretrained LlamaModel and/or a pretrained 
        EsmModel. Any unspecified parts of the model will be intialized with 
        default values.

        return_unused_kwargs is currently not supported in the loading behavior. 
        # TODO complete this case.

        Args:
            pretrained_model_name_or_path:
                Esm-to-Llama model name or path to load pretrained Esm-to-Llama 
                model weights. If given, pretrained EsmModel and LlamaModel name 
                or path will be ignored.
            pretrained_esm_model_name_or_path:
                Esm model name or path to load pretrained EsmModel model weights 
                as encoder part of the whole model.
            pretrained_llama_model_name_or_path:
                Llama model name or path to load pretrained LlamaForCausalLM 
                model weights as decoder part of the whole model.
            esm_kwargs:
                Configuration attributes to override the values in EsmEncoderConfig 
                which is either loaded from pretrained or initialized with default 
                values. The EsmEncoderConfig is then used to instantiate the
                EsmEncoderModel. This entry behaves differently depending on 
                whether a `config` is provided or automatically loaded. Parameters 
                controlling the loading behaviors of EsmEncoderModel such as
                `cache_dir` and `force_download` can also be passed if 
                `pretrained_esm_model_name_or_path` is given, read the 
                documentation from `PreTrainedModel.from_pretrained` for more 
                information.
            kwargs:
                Configuration attributes to override the values in Esm2LlamaConfig 
                or LlamaConfig. This entry behaves differently depending on whether 
                a `config` is provided or automatically loaded. Parameters 
                controlling the loading behaviors of Esm-to-Llama Model or 
                LlamaForCausalLm such as `cache_dir` and `force_download` can also 
                be passed if `pretrained_model_name_or_path` or
                `pretrained_llama_model_name_or_path` is given.
        """
        # case (1): instantiate from a pretrained Esm-to-Llama model
        if pretrained_model_name_or_path is not None:
            config = Esm2LlamaConfig.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                esm_kwargs=esm_kwargs,
                **kwargs
            )
            # updates of the configuration class are done, pass only those 
            # controlling the loading behavior then
            behavior_arg_keywords = list(
                inspect.signature(PreTrainedModel.from_pretrained).parameters.keys()
            )
            behavior_kwargs = {
                key: kwargs[key] 
                for key in kwargs.keys() 
                if key in behavior_arg_keywords
            }
            model = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config,
                **behavior_kwargs
            )

        # case (2-1): instantiate from a pretrained LlamaModel and a pretrained EsmModel
        elif (
            pretrained_esm_model_name_or_path is not None 
            and pretrained_llama_model_name_or_path is not None
        ):
            esm_model = EsmEncoderModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_esm_model_name_or_path,
                **esm_kwargs if esm_kwargs else {}
            )
            # suppress logging "esm weights not initialized from the checkpoint"
            cls._keys_to_ignore_on_load_missing = [r"esm_encoder"]
            model = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_llama_model_name_or_path,
                esm_model=esm_model,
                **kwargs,
            )
            cls._keys_to_ignore_on_load_missing = []

        # case (2-2): instantiate from a pretrained EsmModel
        elif pretrained_esm_model_name_or_path is not None:
            esm_model: EsmEncoderModel = EsmEncoderModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_esm_model_name_or_path,
                **esm_kwargs if esm_kwargs else {}
            )
            llama_config = LlamaConfig(**kwargs)
            model = cls(config=llama_config, esm_model=esm_model)

        # case (2-3): instantiate from a pretrained LlamaModel
        elif pretrained_llama_model_name_or_path is not None:
            esm_model = EsmEncoderModel(
                EsmEncoderConfig(**esm_kwargs if esm_kwargs else {})
            )
            # suppress logging "esm weights not initialized from the checkpoint"
            cls._keys_to_ignore_on_load_missing = [r"esm_encoder"]
            model = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_llama_model_name_or_path,
                esm_model=esm_model,
                **kwargs
            )
            cls._keys_to_ignore_on_load_missing = []

        else:
            raise ValueError(
                "Either pretrained name or path of Esm-to-Llama model, EsmModel "
                "or LlamaForCausalLM should be passed. Use initialization method "
                "instead if none of the above three can be provided. "
            )
        return model

    def _concatenate_encoder_decoder_input(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forming of inputs embeddings, attention mask and potential labels by 
        concatenating encoder outputs and decoder text inputs. All input_ids 
        related arguments should be prepared before passing to the model as this 
        build-in function does not concatenate the prompt with the protein 
        functionality description.
        """
        # convert input_ids to inputs_embeds if necessary
        # all following steps will take inputs_embeds as the flag
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # fill missing attention masks with default values 
        # if corresponding embeddings are given to facilitate concat
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_attention_mask = torch.ones(
                size=encoder_hidden_states.shape[:2],
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
        if attention_mask is None and inputs_embeds is not None:
            attention_mask = torch.ones(
                size=inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device
            )

        # concatenate to form inputs_embeds
        if encoder_hidden_states is not None:
            if inputs_embeds is not None:
                inputs_embeds = torch.cat((encoder_hidden_states, inputs_embeds), dim=1)
            else:
                inputs_embeds = encoder_hidden_states

        # concatenate to form attention_mask
        if encoder_attention_mask is not None:
            if attention_mask is not None:
                attention_mask = torch.cat((encoder_attention_mask, attention_mask), dim=1)
            else:
                attention_mask = encoder_attention_mask

        # form new label by extension with value -100 
        # to avoid computation of loss on the protein embeddings
        # if labels is None, the returning dict will keep it that way
        if labels is not None and encoder_hidden_states is not None:
            encoder_labels = torch.full(
                size=encoder_hidden_states.shape[:2],
                fill_value=-100,
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
            labels = torch.cat((encoder_labels, labels), dim=1)

        return {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
        }

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            protein_input_ids: Optional[torch.LongTensor] = None,
            protein_attention_mask: Optional[torch.LongTensor] = None,
            protein_head_mask: Optional[torch.LongTensor] = None,
            protein_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_encoder_output: bool = False,
            cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions, CausalLMOutputWithPast]:
        """
        The Esm2LlamaForCausalLM forward method, overrides the __call__ special 
        method. The Esm2LlamaForCausalLM instance should be called instead of 
        this function since the former takes care of running the pre and 
        postprocessing steps. This function is constructed based on the method 
        of same name from LlamaForCausalLM with two additional steps: firstly 
        computation of the encoder and secondly reformation of model forward 
        arguments including attention masks and position ids.

        Args:
            input_ids:
                (batch_size, sequence_length) indices of input sequence tokens 
                for LlamaForCausalLM. If past_key_values with actual cache is 
                passed, only the last token have to be input.
            attention_mask:
                (batch_size, sequence_length) attention mask on pad tokens of 
                input sequences with values selected in [0, 1]. If past_key_values 
                is used, only the last token have to be input.
            position_ids:
                (batch_size, sequence_length) indices of positions of each token 
                in sequence for the computation of positional embeddings in 
                LlamaForCausalLM. Only has effect in text sequential generation.
            past_key_values:
                Precomputed kv-cache for every layer of LlamaForCausalLM to speed 
                up sequential text generation.
            inputs_embeds:
                (batch_size, sequence_length, decoder_hidden_size) embedded 
                representation of input sequence. If provided, `input_ids` will 
                be ignored.
            labels:
                (batch_size, sequence_length) labels for computing the masked LM 
                loss with values selected in [-100, 0, 1, ..., config.vocab_size]. 
                Tokens with indices set to `-100` will not be included in the 
                computation of the loss. eos tokens should be included to let the 
                model learn when to end sequential generation through training. 
                Only has effect in training with teacher forcing.
            protein_input_ids:
                (batch_size, protein_sequence_length) indices of input protein 
                sequence tokens for EsmEncoderModel. The protein amino-acid 
                sequences should be left padded for better results.
            protein_attention_mask:
                (batch_size, protein_sequence_length) attention mask on pad tokens 
                of input protein sequences with values selected in [0, 1] for the 
                computation of EsmEncoderModel.
            protein_head_mask:
                (num_heads) or (num_layers, num_heads) head mask with values 
                selected in [0, 1].
            protein_inputs_embeds:
                (batch_size, protein_sequence_length, encoder_hidden_size) embedded 
                representation of protein input sequence. If provided, 
                `protein_input_ids` will be ignored.
            use_cache:
                If True, past_key_values of LlamaForCausalLM will be returned to 
                speed up sequential text generation.
            output_attentions:
                If True, attention scores of all layers in LlamaForCausalLM will 
                be returned. Specially when `return_encoder_output=True`, attention 
                scores of all layers in EsmEncoderModel will be returned instead.
            output_hidden_states:
                If True, hidden states of all layers in LlamaForCausalLM will be 
                returned. Specially when `return_encoder_output=True`, hidden 
                states of all layers in EsmEncoderModel will be returned instead.
            return_dict:
                If True, a ModelOutput instance will be returned instead of a 
                plain tuple.
            return_encoder_output:
                If True and if the encoder is not skipped, the output of the 
                encoder will be returned and the forward pass of the decoder will 
                not be computed. Experimental feature.
            cache_position:
                (sequence_length) indices depicting the position of the whole 
                input sequence including the protein embedding part and the 
                generated text sequence. Contrarily to `position_ids`, this tensor 
                is not affected by padding. It is used to update the cache in the 
                correct position and to infer the complete sequence length.
        """
        # compute encoder in the very first iteration of text sequence generation 
        # and in training with teacher forcing
        encoder_hidden_states: Optional[torch.Tensor] = None
        encoder_attention_mask: Optional[torch.Tensor] = None
        if protein_input_ids is not None or protein_inputs_embeds is not None:
            # cache will not be used since EsmEncoderModel is always of config.is_decoder=False
            # default values will be used for position_ids inside forward pass of the encoder
            encoder_output = self.esm_encoder(
                input_ids=protein_input_ids if protein_inputs_embeds is None else None,
                attention_mask=protein_attention_mask,
                head_mask=protein_head_mask,
                inputs_embeds=protein_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            if return_encoder_output:
                return encoder_output
            encoder_hidden_states = encoder_output[0]
            encoder_attention_mask = protein_attention_mask

        # get concatenated inputs_embeds, attention_mask and labels
        concatenated_input = self._concatenate_encoder_decoder_input(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )

        return super().forward(
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **concatenated_input
        )

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generates protein description text from protein amino-acid sequences. 
        Output sequence will include the prompt and thus need to be sliced by 
        last appearance of bos token to restore the generated description.

        Args:
            inputs:
                (batch_size, prompt_sequence_length) indices of prompt sequence 
                tokens for LlamaForCausalLM. The prompt should end with bos token 
                for better results and the consistency with the fine-tuning 
                process. If not given, single bos token will be used.
            kwargs:
                attention_mask:
                    (batch_size, prompt_sequence_length) attention mask on pad 
                    tokens for prompt sequences with values selected in [0, 1].
                protein_input_ids:
                    (batch_size, protein_sequence_length) indices of input 
                    protein sequence tokens for EsmEncoderModel.
                protein_attention_mask:
                    (batch_size, protein_sequence_length) attention mask on pad 
                    tokens of input protein sequences with values selected in 
                    [0, 1] for the computation of EsmEncoderModel.
                protein_inputs_embeds:
                    (batch_size, protein_sequence_length, encoder_hidden_size) 
                    embedded representation of input protein sequence. If 
                    provided, `protein_input_ids` will be ignored.

            Parameters controlling the generation behaviors such as 
            `generation_config` and `stopping_criteria` can also be 
            passed, read the documentation from `GenerationMixin.generate` 
            for more information.
        """
        # get input embeds of encoder part first, 
        # to be used for every step in sequential generation
        encoder_attention_mask = kwargs.pop("protein_attention_mask", None)
        encoder_output = self.forward(
            protein_input_ids=kwargs.pop("protein_input_ids", None),
            protein_attention_mask=encoder_attention_mask,
            protein_inputs_embeds=kwargs.pop("protein_inputs_embeds", None),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            return_encoder_output=True,
        )
        encoder_hidden_states = encoder_output[0]

        # if prompts are not given, assign default bos token id as initial decoder text inputs
        attention_mask = kwargs.pop("attention_mask", None)
        if inputs is None:
            inputs = torch.full(
                size=(encoder_hidden_states.size(0), 1),
                fill_value=self.config.bos_token_id,
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
            attention_mask = torch.ones_like(inputs)

        # get concatenated inputs_embeds, attention_mask
        # labels are not used and loss function will not be computed in generation
        concatenated_input = self._concatenate_encoder_decoder_input(
            input_ids=inputs,
            attention_mask=attention_mask,
            inputs_embeds=kwargs.pop("inputs_embeds", None),
            labels=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        concatenated_input.update(kwargs)
        return super().generate(**concatenated_input)
