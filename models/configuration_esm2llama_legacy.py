"""
Legacy configuration classes for Esm2LlamaModel. Migrated from previous projects 
on protein function description under a decoder-only base language decoder 
structure. 

Esm2LlamaConfig = LlamaConfig (+ EsmEncoderConfig as additional attribute)
"""


import os
from typing import Any, Dict, Optional, Union

from transformers import EsmConfig, LlamaConfig


class EsmEncoderConfig(EsmConfig):
    """Configuration class of EsmEncoderModel model."""

    def __init__(
            self, 
            *args, 
            decoder_hidden_size: Optional[int] = None, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decoder_hidden_size: int = decoder_hidden_size


class Esm2LlamaConfig(LlamaConfig):
    """
    Configuration class of Esm2LlamaModel model. 

    Args:
        esm_config:
            Configuration to be used with the EsmEncoderConfig encoder 
            configuration. The value is either an instance of EsmEncoderConfig 
            or a dict of parameters to be passed to initialize the 
            EsmEncoderConfig (read the documentation from `EsmEncoderConfig` 
            for more information in this case). If not given, a default
            configuration is used.
        kwargs:
            Keyword arguments for initialization of LlamaConfig, read the 
            documentation from `LlamaConfig` for more information. Parameters 
            controlling the model outputs can also be passed, read the 
            documentation from `PretrainedConfig` for more information.
    """
    def __init__(
                self, 
                *args, 
                esm_config: Optional[Union[EsmEncoderConfig, Dict[str, Any]]] = None, 
                **kwargs
        ):
        # normal initialization of LlamaConfig with keyword arguments
        super().__init__(*args, **kwargs)

        # add self.esm_config: EsmEncoderConfig as extra attribute to LlamaConfig
        if esm_config is None or isinstance(esm_config, dict):
            self.esm_config = EsmEncoderConfig(**esm_config if esm_config else {})
        elif isinstance(esm_config, EsmEncoderConfig):
            self.esm_config = esm_config
        else:
            raise ValueError(
                "esm_config must be a EsmEncoderConfig, or a dict of "
                "initialization parameters. Use from_pretrained method instead "
                "if the esm_config shall be loaded from a pretrained model "
                "name or path. "
            )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            pretrained_esm_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            pretrained_llama_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            esm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> "Esm2LlamaConfig":
        """
        Instantiates a Esm2LlamaConfig from either (1) a pretrained 
        Esm-to-Llama model, or (2) a pretrained LlamaForCausalLM and/or a 
        pretrained EsmModel. The configuration of any unspecified parts of 
        the model will be initialized with default values.

        return_unused_kwargs is currently not supported in the loading 
        behavior. # TODO complete this case.

        Args:
            pretrained_model_name_or_path:
                Esm-to-Llama model name or path to load predefined Esm-to-Llama 
                model configuration. If given, pretrained EsmModel and 
                LlamaForCausalLM name or path will be ignored.
            pretrained_esm_model_name_or_path:
                Esm model name or path to load predefined EsmModel configuration 
                as encoder part of the whole model.
            pretrained_llama_model_name_or_path:
                Llama model name or path to load predefined LlamaForCausalLM 
                model configuration as decoder part of the whole model.
            esm_kwargs:
                Configuration attributes to override values in EsmEncoderConfig 
                which is either loaded from pretrained or initialized with 
                default values. Behavior concerning key/value pairs whose keys 
                are not configuration attributes is controlled by the 
                return_unused_kwargs keyword parameter. Parameters controlling 
                the loading behaviors of Esm configuration such as `cache_dir` 
                and `force_download` can also be passed if 
                `pretrained_esm_model_name_or_path` is given, read the 
                documentation from `PretrainedConfig.from_pretrained` for more 
                information.
            kwargs:
                Configuration attributes to override the loaded values in 
                Esm2LlamaConfig or LlamaConfig. Parameters controlling the 
                loading behaviors of Esm-to-Llama or Llama configuration such 
                as `cache_dir` and `force_download` can also be passed if 
                `pretrained_model_name_or_path` or 
                `pretrained_llama_model_name_or_path` is given.
        """
        # case (1): instantiate from a pretrained Esm-to-Llama model
        if pretrained_model_name_or_path is not None:
            config = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path, 
                **kwargs
            )
            if esm_kwargs:
                config.esm_config.update(esm_kwargs)

        # case (2-1): instantiate from pretrained LlamaModel and EsmModel
        elif (
            pretrained_esm_model_name_or_path is not None 
            and pretrained_llama_model_name_or_path is not None
        ):
            config = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_llama_model_name_or_path,
                **kwargs
            )
            config.esm_config = EsmEncoderConfig.from_pretrained(
                pretrained_model_name_or_path=pretrained_esm_model_name_or_path,
                **esm_kwargs if esm_kwargs else {}
            )

        # case (2-2): instantiate from a pretrained EsmModel
        elif pretrained_esm_model_name_or_path is not None:
            esm_config = EsmEncoderConfig.from_pretrained(
                pretrained_model_name_or_path=pretrained_esm_model_name_or_path,
                **esm_kwargs if esm_kwargs else {}
            )
            config = cls(esm_config=esm_config, **kwargs)

        # case (2-3): instantiate from a pretrained LlamaModel
        elif pretrained_llama_model_name_or_path is not None:
            config = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_llama_model_name_or_path,
                **kwargs
            )
            config.esm_config = EsmEncoderConfig(**esm_kwargs if esm_kwargs else {})

        else:
            raise ValueError(
                "Either pretrained name or path of Esm-to-Llama model, EsmModel "
                "or LlamaForCausalLM should be passed. Use initialization "
                "method instead if none of the above three can be provided. "
            )
        return config
