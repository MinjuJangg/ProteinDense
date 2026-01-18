"""
Configuration class for the assembled Esm2Rgcn2LlamaInstructForCausalLM model. 

Esm2LlamaInstructConfig = EsmConfig + RgcnAdapterConfig + LlamaConfig
"""


from transformers import EsmConfig, LlamaConfig, PretrainedConfig


class RgcnAdapterConfig(PretrainedConfig):
    """Configuration class of the Relational Graph Convolutional Network adapter."""
    model_type = "rgcn_adapter"  # unique identifier of the model

    def __init__(
            self, 
            input_dim: int, 
            intermediate_dim: int,
            output_dim: int, 
            n_relations: int = 7,
            n_layers: int = 6,
            dropout_rate: float = 0.2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate


class Esm2Rgcn2LlamaInstructConfig(PretrainedConfig):
    """
    Configuration class of Esm2Rgcn2LlamaInstructForCausalLM model.
    placeholder_id: Token id in chat template to be replaced by ESM embeddings.
    """
    model_type = "esm2rgcn2llama_instruct"  # unique identifier of the model

    def __init__(
            self, 
            # model components
            esm_config: EsmConfig, 
            adapter_config: RgcnAdapterConfig,
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
