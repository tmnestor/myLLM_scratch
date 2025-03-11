import os
from transformers import AutoModel, AutoConfig
from mini_lm_model import SentenceTransformer


def load_pretrained_weights(custom_model, model_name="paraphrase-MiniLM-L3-v2"):
    """
    Load pretrained weights from local files into our custom model architecture

    Args:
        custom_model: Instance of our custom MiniLMModel or SentenceTransformer
        model_name: Model name without organization prefix. Supported models:
                   - paraphrase-MiniLM-L3-v2 (default)
                   - all-MiniLM-L6-v2
                   - paraphrase-MiniLM-L6-v2
                   - all-MiniLM-L12-v2

    Returns:
        Model with loaded weights
    """
    # Validate input parameters
    supported_models = [
        "paraphrase-MiniLM-L3-v2",
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
    ]
    if model_name not in supported_models:
        raise ValueError(
            f"Unsupported model: {model_name}. Must be one of {supported_models}"
        )

    if custom_model is None:
        raise ValueError("custom_model cannot be None")

    if not isinstance(custom_model, SentenceTransformer):
        raise TypeError(
            f"custom_model must be an instance of SentenceTransformer, got {type(custom_model)}"
        )

    # Get environment variable for model path
    model_path = os.environ.get("LLM_MODELS_PATH")
    if not model_path:
        raise EnvironmentError("LLM_MODELS_PATH environment variable is not set")

    # Form the path to the local model
    model_folder = model_name.split("/")[-1] if "/" in model_name else model_name
    local_model_path = os.path.join(model_path, model_folder)

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model folder not found at {local_model_path}")

    print(f"Loading pretrained weights from local path: {local_model_path}")

    # Load config to validate architecture match
    config = AutoConfig.from_pretrained(local_model_path, local_files_only=True)

    # Get number of layers from our custom model using the correct attribute name
    num_layers = len(custom_model.model.encoder.layer)

    # Validate architecture matches
    if (
        config.hidden_size
        != custom_model.model.embeddings.word_embeddings.embedding_dim
    ):
        raise ValueError(
            f"Hidden size mismatch: pretrained={config.hidden_size}, custom={custom_model.model.embeddings.word_embeddings.embedding_dim}"
        )
    if config.num_hidden_layers != num_layers:
        raise ValueError(
            f"Layer count mismatch: pretrained={config.num_hidden_layers}, custom={num_layers}"
        )
    if (
        config.num_attention_heads
        != custom_model.model.encoder.layer[0].attention.num_attention_heads
    ):
        raise ValueError(
            f"Attention head count mismatch: pretrained={config.num_attention_heads}, custom={custom_model.model.encoder.layer[0].attention.num_attention_heads}"
        )

    # Load pretrained model
    pretrained_model = AutoModel.from_pretrained(
        local_model_path, local_files_only=True
    )
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = custom_model.state_dict()

    # Create mappings between pretrained weights and custom model
    mapping = {
        # Embeddings
        "embeddings.word_embeddings.weight": "model.embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight": "model.embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "model.embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight": "model.embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias": "model.embeddings.LayerNorm.bias",
    }

    # For each transformer layer, map the weights
    for i in range(num_layers):
        base_src = f"encoder.layer.{i}"
        base_dst = f"model.encoder.layer.{i}"  # Updated to use layer instead of layers

        # Attention weights
        mapping.update(
            {
                f"{base_src}.attention.self.query.weight": f"{base_dst}.attention.self.query.weight",
                f"{base_src}.attention.self.query.bias": f"{base_dst}.attention.self.query.bias",
                f"{base_src}.attention.self.key.weight": f"{base_dst}.attention.self.key.weight",
                f"{base_src}.attention.self.key.bias": f"{base_dst}.attention.self.key.bias",
                f"{base_src}.attention.self.value.weight": f"{base_dst}.attention.self.value.weight",
                f"{base_src}.attention.self.value.bias": f"{base_dst}.attention.self.value.bias",
                f"{base_src}.attention.output.dense.weight": f"{base_dst}.attention.output.dense.weight",
                f"{base_src}.attention.output.dense.bias": f"{base_dst}.attention.output.dense.bias",
                f"{base_src}.attention.output.LayerNorm.weight": f"{base_dst}.attention.output.LayerNorm.weight",
                f"{base_src}.attention.output.LayerNorm.bias": f"{base_dst}.attention.output.LayerNorm.bias",
                # Feed-forward weights
                f"{base_src}.intermediate.dense.weight": f"{base_dst}.intermediate.dense.weight",
                f"{base_src}.intermediate.dense.bias": f"{base_dst}.intermediate.dense.bias",
                f"{base_src}.output.dense.weight": f"{base_dst}.output.dense.weight",
                f"{base_src}.output.dense.bias": f"{base_dst}.output.dense.bias",
                f"{base_src}.output.LayerNorm.weight": f"{base_dst}.output.LayerNorm.weight",
                f"{base_src}.output.LayerNorm.bias": f"{base_dst}.output.LayerNorm.bias",
            }
        )

    # Pooler weights
    mapping.update(
        {
            "pooler.dense.weight": "model.pooler.dense.weight",
            "pooler.dense.bias": "model.pooler.dense.bias",
        }
    )

    # Transfer weights with validation
    missing_keys = []
    for pretrained_name, custom_name in mapping.items():
        if pretrained_name in pretrained_dict and custom_name in custom_dict:
            if pretrained_dict[pretrained_name].shape != custom_dict[custom_name].shape:
                print(
                    f"Shape mismatch for {pretrained_name}: pretrained={pretrained_dict[pretrained_name].shape}, custom={custom_dict[custom_name].shape}"
                )
            else:
                custom_dict[custom_name] = pretrained_dict[pretrained_name]
        else:
            missing_keys.append((pretrained_name, custom_name))

    if missing_keys:
        print("\nWarning: Some weights could not be loaded:")
        for pretrained_name, custom_name in missing_keys:
            print(f"  {pretrained_name} -> {custom_name}")

    # Load the updated weights into our model
    custom_model.load_state_dict(custom_dict)
    print("Successfully loaded pretrained weights into custom model")

    return custom_model
