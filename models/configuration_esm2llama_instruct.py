"""
Configuration class for the assembled Esm2LlamaInstructForCausalLM model. 

Esm2LlamaInstructConfig = EsmConfig + ModalityAdapterConfig + LlamaConfig
"""


from transformers import EsmConfig, LlamaConfig, PretrainedConfig


class ModalityAdapterConfig(PretrainedConfig):
    """Configuration class of the 2-layer non-linear adapter."""
    model_type = "modality_adapter"  # unique identifier of the model

    def __init__(
            self, 
            input_dim: int, 
            intermediate_dim: int,
            output_dim: int, 
            dropout_rate: float = 0.3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate


class Esm2LlamaInstructConfig(PretrainedConfig):
    """
    Configuration class of Esm2LlamaInstructForCausalLM model.
    placeholder_id: Token id in chat template to be replaced by ESM embeddings.
    """
    model_type = "esm2llama_instruct"  # unique identifier of the model

    def __init__(
            self, 
            # model components
            esm_config: EsmConfig, 
            adapter_config: ModalityAdapterConfig,
            llama_config: LlamaConfig, 
            # standalone attributes
            placeholder_id: int = 128003, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.esm_config = esm_config
        self.adapter_config = adapter_config
        self.llama_config = llama_config
        self.placeholder_id = placeholder_id
