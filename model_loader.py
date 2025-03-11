import os
import torch
from transformers import AutoModel, AutoConfig
from mini_lm_model import SentenceTransformer


def load_pretrained_weights(custom_model, model_name="paraphrase-MiniLM-L3-v2", model_type=None):
    """
    Load pretrained weights from local files into our custom model architecture

    Args:
        custom_model: Instance of our custom transformer model (SentenceTransformer)
        model_name: Model name without organization prefix. Supported models:
                   - paraphrase-MiniLM-L3-v2 (default)
                   - all-MiniLM-L6-v2
                   - paraphrase-MiniLM-L6-v2
                   - all-MiniLM-L12-v2
                   - modernbert
        model_type: Type of model architecture ('minilm' or 'modernbert'). If None,
                   it will be inferred from model_name.

    Returns:
        Model with loaded weights
    """
    # Infer model type if not provided
    if model_type is None:
        if "MiniLM" in model_name:
            model_type = "minilm"
        elif model_name.lower() == "modernbert":
            model_type = "modernbert"
        else:
            raise ValueError(f"Cannot infer model_type from model_name: {model_name}. Please specify model_type explicitly.")
    
    # Validate input parameters
    supported_minilm_models = [
        "paraphrase-MiniLM-L3-v2",
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
    ]
    
    # Standardize model type to lowercase
    model_type = model_type.lower()
    
    # Validate and standardize model name based on model type
    if model_type == "minilm" and model_name not in supported_minilm_models:
        raise ValueError(
            f"Unsupported MiniLM model: {model_name}. Must be one of {supported_minilm_models}"
        )
    elif model_type == "modernbert":
        # Standardize the model name to modernbert-base for consistency
        model_name = "modernbert-base"

    if custom_model is None:
        raise ValueError("custom_model cannot be None")

    if not isinstance(custom_model, SentenceTransformer):
        raise TypeError(
            f"custom_model must be an instance of SentenceTransformer, got {type(custom_model)}"
        )

    # Always use LLM_MODELS_PATH, but adjust folder name for ModernBERT
    model_path_env = "LLM_MODELS_PATH"
    
    # Get environment variable for model path
    model_path = os.environ.get(model_path_env)
    if not model_path:
        raise EnvironmentError(f"{model_path_env} environment variable is not set")

    # Form the path to the local model
    if model_type == "minilm":
        model_folder = model_name.split("/")[-1] if "/" in model_name else model_name
    else:  # modernbert - use correct capitalization for the folder
        model_folder = "ModernBERT-base"
        
    local_model_path = os.path.join(model_path, model_folder)

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model folder not found at {local_model_path}")

    print(f"Loading pretrained weights from local path: {local_model_path}")

    # Store model type for later use
    model_type = model_type.lower()
    
    # Get number of layers from our custom model using the correct attribute name
    num_layers = len(custom_model.model.encoder.layer)
    
    if model_type == "minilm":
        # Load config to validate architecture match for MiniLM models
        config = AutoConfig.from_pretrained(local_model_path, local_files_only=True)

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
    
    else:  # ModernBERT
        # For ModernBERT, we might need to use a different loading approach
        try:
            # Try loading with the transformers approach first
            config = AutoConfig.from_pretrained(local_model_path, local_files_only=True)
            pretrained_model = AutoModel.from_pretrained(
                local_model_path, local_files_only=True
            )
            pretrained_dict = pretrained_model.state_dict()
            
            # Validate basic architecture
            if config.hidden_size != custom_model.model.embeddings.word_embeddings.embedding_dim:
                print(f"Warning: Hidden size mismatch: pretrained={config.hidden_size}, custom={custom_model.model.embeddings.word_embeddings.embedding_dim}")
                
            if hasattr(config, 'num_hidden_layers') and config.num_hidden_layers != num_layers:
                print(f"Warning: Layer count mismatch: pretrained={config.num_hidden_layers}, custom={num_layers}")
            
        except Exception as e:
            # If transformers loading fails, try direct PyTorch loading
            print(f"Transformers loading failed: {e}")
            print("Attempting direct PyTorch loading...")
            
            # Try to load state dict directly
            model_file = os.path.join(local_model_path, "pytorch_model.bin")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found at {model_file}")
                
            pretrained_dict = torch.load(model_file, map_location=torch.device('cpu'))
    
    # Get custom model state dict
    custom_dict = custom_model.state_dict()

    # Create mappings between pretrained weights and custom model
    if model_type == "minilm":
        # Standard MiniLM weight mapping
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
            base_dst = f"model.encoder.layer.{i}"

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
        
    else:  # ModernBERT
        # Create a more flexible mapping strategy that can handle ModernBERT's structure
        # First, initialize with potential standard names
        mapping = {}
        
        # Define a function to try different naming patterns
        def add_mapping_patterns(pretrained_dict, custom_dict, mapping):
            # Common embedding patterns
            embedding_patterns = [
                ("embeddings.word_embeddings.weight", "model.embeddings.word_embeddings.weight"),
                ("bert.embeddings.word_embeddings.weight", "model.embeddings.word_embeddings.weight"),
                ("transformer.embeddings.word_embeddings.weight", "model.embeddings.word_embeddings.weight"),
                ("embeddings.position_embeddings.weight", "model.embeddings.position_embeddings.weight"),
                ("bert.embeddings.position_embeddings.weight", "model.embeddings.position_embeddings.weight"),
                ("transformer.embeddings.position_embeddings.weight", "model.embeddings.position_embeddings.weight"),
                ("embeddings.token_type_embeddings.weight", "model.embeddings.token_type_embeddings.weight"),
                ("bert.embeddings.token_type_embeddings.weight", "model.embeddings.token_type_embeddings.weight"),
                ("transformer.embeddings.token_type_embeddings.weight", "model.embeddings.token_type_embeddings.weight"),
                ("embeddings.LayerNorm.weight", "model.embeddings.LayerNorm.weight"),
                ("bert.embeddings.LayerNorm.weight", "model.embeddings.LayerNorm.weight"),
                ("transformer.embeddings.LayerNorm.weight", "model.embeddings.LayerNorm.weight"),
                ("embeddings.LayerNorm.bias", "model.embeddings.LayerNorm.bias"),
                ("bert.embeddings.LayerNorm.bias", "model.embeddings.LayerNorm.bias"),
                ("transformer.embeddings.LayerNorm.bias", "model.embeddings.LayerNorm.bias"),
            ]
            
            # Add embedding patterns to mapping if they exist in both dicts
            for src, dst in embedding_patterns:
                if src in pretrained_dict and dst in custom_dict:
                    mapping[src] = dst
            
            # Pooler patterns
            pooler_patterns = [
                ("pooler.dense.weight", "model.pooler.dense.weight"),
                ("bert.pooler.dense.weight", "model.pooler.dense.weight"),
                ("transformer.pooler.dense.weight", "model.pooler.dense.weight"),
                ("pooler.dense.bias", "model.pooler.dense.bias"),
                ("bert.pooler.dense.bias", "model.pooler.dense.bias"),
                ("transformer.pooler.dense.bias", "model.pooler.dense.bias"),
            ]
            
            # Add pooler patterns to mapping if they exist in both dicts
            for src, dst in pooler_patterns:
                if src in pretrained_dict and dst in custom_dict:
                    mapping[src] = dst
                    
            # Process encoder layers
            # Check different naming patterns for encoder layers
            encoder_prefixes = ["encoder", "bert.encoder", "transformer.encoder"]
            layer_name_patterns = ["layer", "layers"]
            
            # Try different combinations
            for prefix in encoder_prefixes:
                for layer_pattern in layer_name_patterns:
                    for i in range(num_layers):
                        base_src = f"{prefix}.{layer_pattern}.{i}"
                        base_dst = f"model.encoder.layer.{i}"
                        
                        # Skip if not a valid pattern for this model
                        pattern_exists = False
                        for key in pretrained_dict.keys():
                            if key.startswith(base_src):
                                pattern_exists = True
                                break
                                
                        if not pattern_exists:
                            continue
                            
                        # Attention components
                        attention_patterns = [
                            (".attention.self.query.weight", ".attention.self.query.weight"),
                            (".attention.self.query.bias", ".attention.self.query.bias"),
                            (".attention.self.key.weight", ".attention.self.key.weight"),
                            (".attention.self.key.bias", ".attention.self.key.bias"),
                            (".attention.self.value.weight", ".attention.self.value.weight"),
                            (".attention.self.value.bias", ".attention.self.value.bias"),
                            (".attention.output.dense.weight", ".attention.output.dense.weight"),
                            (".attention.output.dense.bias", ".attention.output.dense.bias"),
                            (".attention.output.LayerNorm.weight", ".attention.output.LayerNorm.weight"),
                            (".attention.output.LayerNorm.bias", ".attention.output.LayerNorm.bias"),
                        ]
                        
                        for src_suffix, dst_suffix in attention_patterns:
                            src = base_src + src_suffix
                            dst = base_dst + dst_suffix
                            if src in pretrained_dict and dst in custom_dict:
                                mapping[src] = dst
                                
                        # Feed-forward components
                        ff_patterns = [
                            (".intermediate.dense.weight", ".intermediate.dense.weight"),
                            (".intermediate.dense.bias", ".intermediate.dense.bias"),
                            (".output.dense.weight", ".output.dense.weight"),
                            (".output.dense.bias", ".output.dense.bias"),
                            (".output.LayerNorm.weight", ".output.LayerNorm.weight"),
                            (".output.LayerNorm.bias", ".output.LayerNorm.bias"),
                        ]
                        
                        for src_suffix, dst_suffix in ff_patterns:
                            src = base_src + src_suffix
                            dst = base_dst + dst_suffix
                            if src in pretrained_dict and dst in custom_dict:
                                mapping[src] = dst
        
        # Try to map weights based on various naming patterns
        add_mapping_patterns(pretrained_dict, custom_dict, mapping)
        
        # Print mapping statistics
        print(f"Found {len(mapping)} weight mappings for ModernBERT model")

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
