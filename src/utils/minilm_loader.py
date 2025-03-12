"""
Specialized loader for MiniLM model weights.

This module provides a dedicated loader for MiniLM models that correctly maps
the weights from HuggingFace format to our model architecture.
"""

import os
import torch
from collections import OrderedDict

def load_minilm_weights(model, weights_path):
    """
    Load MiniLM weights into our custom model architecture.
    
    This is a specialized loader that handles the specific mapping from
    HuggingFace MiniLM weights to our custom model structure.
    
    Args:
        model: Our MiniLMForSentenceEmbedding model
        weights_path: Path to the MiniLM weights (directory or file)
        
    Returns:
        tuple: (model with loaded weights, diagnostics)
    """
    # Resolve weights path (could be relative, name, or absolute path)
    resolved_path = weights_path
    
    # If not a direct path, check environment variable
    if not os.path.exists(resolved_path):
        env_path = os.environ.get("LLM_MODELS_PATH")
        if env_path:
            full_path = os.path.join(env_path, weights_path)
            if os.path.exists(full_path):
                resolved_path = full_path
                print(f"Found model at {resolved_path}")
            else:
                raise ValueError(f"Model path not found: {weights_path} or {full_path}")
        else:
            raise ValueError(f"Model path not found: {weights_path} and LLM_MODELS_PATH not set")
    
    # Find the weights file
    if os.path.isdir(resolved_path):
        # Check for common filenames in MiniLM models
        possible_filenames = [
            "pytorch_model.bin",
            "model.safetensors",
            "model.pt",
            "model.bin"
        ]
        
        for filename in possible_filenames:
            model_file = os.path.join(resolved_path, filename)
            if os.path.exists(model_file):
                break
        else:
            print(f"Available files in {resolved_path}:")
            for file in os.listdir(resolved_path):
                print(f"  - {file}")
            raise ValueError(f"Could not find model weights file in {resolved_path}")
    else:
        model_file = resolved_path
    
    print(f"Loading MiniLM weights from: {model_file}")
    
    # Load weights based on file format
    if model_file.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            
            state_dict = OrderedDict()
            with safe_open(model_file, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            print(f"Loaded safetensors weights from {model_file}")
        except ImportError:
            print("safetensors package not found, trying to load with torch...")
            state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    else:
        # Regular PyTorch format
        state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    
    # Get our model's state dict for reference
    target_dict = model.state_dict()
    
    # Print sample keys and key structures from both dicts for debugging
    source_keys = list(state_dict.keys())
    print(f"Source state_dict contains {len(source_keys)} parameters")
    print("Sample source keys (MiniLM):", source_keys[:5])
    
    # Analyze key patterns in source
    source_prefixes = set()
    for key in source_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            source_prefixes.add(parts[0] + '.' + parts[1])
    print(f"Source key prefixes: {sorted(list(source_prefixes))}")
    
    # Show encoder layer keys if present
    encoder_keys = [k for k in source_keys if k.startswith("encoder.layer")]
    if encoder_keys:
        print(f"Found {len(encoder_keys)} encoder layer keys. Examples:")
        print(encoder_keys[:5])
    
    # Analyze target model keys
    target_keys = list(target_dict.keys())
    print(f"Target model contains {len(target_keys)} parameters")
    print("Sample target keys (Our model):", target_keys[:5])
    
    # Analyze key patterns in target
    target_prefixes = set()
    for key in target_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            target_prefixes.add(parts[0] + '.' + parts[1])
    print(f"Target key prefixes: {sorted(list(target_prefixes))}")
    
    # Create parameter mapping from loaded state dict to our model
    # First, let's see what our model's layer organization looks like by checking a few parameters
    sample_keys = []
    for key in target_keys:
        if "encoder" in key and "layer" in key:
            sample_keys.append(key)
        if len(sample_keys) >= 3:
            break
    
    print(f"Sample model parameter keys: {sample_keys}")
    
    # Create systematic mapping between source and target parameters
    mapping = {}
    
    # Map embeddings parameters
    embedding_key_pairs = [
        ("embeddings.word_embeddings.weight", "model.embeddings.word_embeddings.weight"),
        ("embeddings.position_embeddings.weight", "model.embeddings.position_embeddings.weight"),
        ("embeddings.token_type_embeddings.weight", "model.embeddings.token_type_embeddings.weight"),
        ("embeddings.LayerNorm.weight", "model.embeddings.LayerNorm.weight"),
        ("embeddings.LayerNorm.bias", "model.embeddings.LayerNorm.bias"),
    ]
    
    for source, target in embedding_key_pairs:
        if source in source_keys and target in target_keys:
            mapping[source] = target
    
    # Determine target layer name format based on model inspection
    # Initialize model_uses_layer
    model_uses_layer = False
    
    # Check for layer keys vs layers keys to determine the format
    layer_keys = [key for key in target_keys if "model.encoder.layer." in key and "layers" not in key]
    layers_keys = [key for key in target_keys if "model.encoder.layers." in key]
    
    if len(layer_keys) > 0:
        model_uses_layer = True
        print(f"Model uses 'model.encoder.layer' format (found {len(layer_keys)} keys)")
    elif len(layers_keys) > 0:
        model_uses_layer = False
        print(f"Model uses 'model.encoder.layers' format (found {len(layers_keys)} keys)")
    else:
        print("Warning: Could not determine if model uses 'layer' or 'layers' from parameter names - defaulting to 'layer'")
        model_uses_layer = True  # Default to "layer" if we can't determine
    
    target_layer_key = "layer" if model_uses_layer else "layers"
    print(f"Target model uses 'encoder.{target_layer_key}' format")
    
    # Show example target parameter with the correct format
    target_format = f"model.encoder.{target_layer_key}"
    target_examples = [key for key in target_keys if target_format in key]
    if target_examples:
        print(f"Example target parameter: {target_examples[0]}")
    else:
        print(f"Warning: No parameters found with '{target_format}' pattern")
    
    # Check if our model has self-attention structure
    has_self_attention = False
    
    test_keys = [
        f"model.encoder.{target_layer_key}.0.attention.self.query.weight",
        f"model.encoder.{target_layer_key}.0.attention.query.weight"
    ]
    
    for key in test_keys:
        if key in target_keys:
            has_self_attention = "self" in key
            print(f"Found attention parameter: {key}")
            break
    
    # Determine layer format for mapping
    if has_self_attention:
        print("Using standard BERT/MiniLM layer mapping (with self-attention)")
        layer_format = "standard"
    else:
        print("Using simplified layer mapping (without self-attention)")
        layer_format = "simplified"
    
    # Let's examine our model architecture to understand the parameter structure
    model_layers = []
    for name, _ in model.named_modules():
        model_layers.append(name)
    
    encoder_layers_found = []
    for layer in model_layers:
        # Look for both layer and layers patterns
        if 'encoder.layers' in layer or 'encoder.layer.' in layer:
            encoder_layers_found.append(layer)
    
    # Initialize key variables
    num_layers = 0
    
    # Determine if model uses "layer" or "layers" in its parameter names
    model_uses_layer = False  # Default to "layers"
    layer_keys = [key for key in target_keys if "model.encoder.layer." in key and "layers" not in key]
    layers_keys = [key for key in target_keys if "model.encoder.layers." in key]
    
    if len(layer_keys) > 0:
        model_uses_layer = True
        print(f"Model uses 'model.encoder.layer' format (found {len(layer_keys)} keys)")
    elif len(layers_keys) > 0:
        model_uses_layer = False
        print(f"Model uses 'model.encoder.layers' format (found {len(layers_keys)} keys)")
    else:
        print("Warning: Could not determine if model uses 'layer' or 'layers' from parameter names")
        # Fallback to module inspection
    
    # Let's print more detailed model structure for debugging
    print("\nModel structure details:")
    if hasattr(model, "model"):
        print("- SentenceEncoder wrapper found")
        base_model = model.model
        
        # Check if model attributes are accessible
        if hasattr(base_model, "encoder"):
            print("- Has encoder attribute")
            if hasattr(base_model.encoder, "layers"):
                print(f"- Encoder has 'layers' attribute: {type(base_model.encoder.layers)}")
                try:
                    num_layers = len(base_model.encoder.layers)
                    print(f"- Number of layers found: {num_layers}")
                except:
                    print("- Could not determine number of layers")
            elif hasattr(base_model.encoder, "layer"):
                print(f"- Encoder has 'layer' attribute: {type(base_model.encoder.layer)}")
                model_uses_layer = True  # Important! Model uses "layer" not "layers"
                try:
                    num_layers = len(base_model.encoder.layer)
                    print(f"- Number of layers found: {num_layers}")
                except:
                    print("- Could not determine number of layers")
        else:
            print("- No encoder attribute found")
    else:
        print("- No model wrapper found")
    
    # Let's check the model parameter names to infer structure
    encoder_layer_pattern = 0
    
    # Search pattern depends on whether model uses "layer" or "layers"
    layer_key = "layer" if model_uses_layer else "layers"
    print(f"Searching for encoder.{layer_key} in parameter names")
    
    for key in target_keys:
        if f"encoder.{layer_key}" in key:
            parts = key.split('.')
            try:
                # Extract layer number from key pattern
                layer_idx = int(parts[parts.index(layer_key) + 1])
                encoder_layer_pattern = max(encoder_layer_pattern, layer_idx + 1)
            except (ValueError, IndexError):
                pass
    
    if encoder_layer_pattern > 0:
        num_layers = encoder_layer_pattern
        print(f"Inferred {num_layers} encoder layers from parameter names")
    
    # If we still don't have layers, check source dictionary to infer model structure
    if num_layers == 0:
        source_layer_pattern = 0
        for key in source_keys:
            if "encoder.layer" in key:
                parts = key.split('.')
                try:
                    # Extract layer number from key pattern like encoder.layer.0.attention...
                    layer_idx = int(parts[parts.index('layer') + 1])
                    source_layer_pattern = max(source_layer_pattern, layer_idx + 1)
                except (ValueError, IndexError):
                    pass
        
        if source_layer_pattern > 0:
            num_layers = source_layer_pattern
            print(f"Using source model's layer count: {num_layers}")
    
    # Add encoder layer mappings dynamically with more flexible architecture detection
    if num_layers > 0:
        print(f"Mapping {num_layers} encoder layers")
        # Check if model uses attention.self structure
        has_self_attention = any("attention.self" in layer for layer in encoder_layers_found)
        print(f"Model uses 'attention.self' structure: {has_self_attention}")
        
        # Define mapping templates for different encoder layer architectures
        # Use the correct target layer key (layer or layers) based on model inspection
        target_format = f"model.encoder.{target_layer_key}.{{}}"
        
        # These templates can be customized based on the specific model architecture
        mapping_templates = {
            "standard": {
                # Source pattern -> Target pattern with dynamic target format
                "encoder.layer.{}.attention.self.query.weight": target_format + ".attention.self.query.weight",
                "encoder.layer.{}.attention.self.query.bias": target_format + ".attention.self.query.bias",
                "encoder.layer.{}.attention.self.key.weight": target_format + ".attention.self.key.weight",
                "encoder.layer.{}.attention.self.key.bias": target_format + ".attention.self.key.bias",
                "encoder.layer.{}.attention.self.value.weight": target_format + ".attention.self.value.weight",
                "encoder.layer.{}.attention.self.value.bias": target_format + ".attention.self.value.bias",
                "encoder.layer.{}.attention.output.dense.weight": target_format + ".attention.output.dense.weight",
                "encoder.layer.{}.attention.output.dense.bias": target_format + ".attention.output.dense.bias",
                "encoder.layer.{}.attention.output.LayerNorm.weight": target_format + ".attention.output.LayerNorm.weight",
                "encoder.layer.{}.attention.output.LayerNorm.bias": target_format + ".attention.output.LayerNorm.bias",
                "encoder.layer.{}.intermediate.dense.weight": target_format + ".intermediate.dense.weight",
                "encoder.layer.{}.intermediate.dense.bias": target_format + ".intermediate.dense.bias",
                "encoder.layer.{}.output.dense.weight": target_format + ".output.dense.weight",
                "encoder.layer.{}.output.dense.bias": target_format + ".output.dense.bias",
                "encoder.layer.{}.output.LayerNorm.weight": target_format + ".output.LayerNorm.weight",
                "encoder.layer.{}.output.LayerNorm.bias": target_format + ".output.LayerNorm.bias",
            },
            "simplified": {
                # Simplified attention structure (no .self)
                "encoder.layer.{}.attention.query.weight": target_format + ".attention.query.weight",
                "encoder.layer.{}.attention.query.bias": target_format + ".attention.query.bias",
                "encoder.layer.{}.attention.key.weight": target_format + ".attention.key.weight", 
                "encoder.layer.{}.attention.key.bias": target_format + ".attention.key.bias",
                "encoder.layer.{}.attention.value.weight": target_format + ".attention.value.weight",
                "encoder.layer.{}.attention.value.bias": target_format + ".attention.value.bias",
                # Rest is the same as standard
                "encoder.layer.{}.attention.output.dense.weight": target_format + ".attention.output.dense.weight",
                "encoder.layer.{}.attention.output.dense.bias": target_format + ".attention.output.dense.bias",
                "encoder.layer.{}.attention.output.LayerNorm.weight": target_format + ".attention.output.LayerNorm.weight",
                "encoder.layer.{}.attention.output.LayerNorm.bias": target_format + ".attention.output.LayerNorm.bias",
                "encoder.layer.{}.intermediate.dense.weight": target_format + ".intermediate.dense.weight",
                "encoder.layer.{}.intermediate.dense.bias": target_format + ".intermediate.dense.bias",
                "encoder.layer.{}.output.dense.weight": target_format + ".output.dense.weight",
                "encoder.layer.{}.output.dense.bias": target_format + ".output.dense.bias",
                "encoder.layer.{}.output.LayerNorm.weight": target_format + ".output.LayerNorm.weight",
                "encoder.layer.{}.output.LayerNorm.bias": target_format + ".output.LayerNorm.bias",
            }
        }
        
        # Pick the right mapping template based on detected architecture
        # Default to standard if we can't determine
        template_key = "standard"
        
        # Try to autodetect the right template
        if not has_self_attention:
            template_key = "simplified"
        
        template = mapping_templates[template_key]
        print(f"Using {template_key} mapping template")
        
        # Generate mappings for each layer using the template
        for i in range(num_layers):
            for source_pattern, target_pattern in template.items():
                source_key = source_pattern.format(i)
                target_key = target_pattern.format(i)
                
                # Only add if both keys exist in their respective dictionaries
                if source_key in source_keys and target_key in target_keys:
                    mapping[source_key] = target_key
        
        # In case layer structure detection failed, do a direct scan on source layer keys
        # and map them to our layer structure
        if len([k for k in mapping.keys() if "encoder.layer" in k]) == 0:
            print("Direct layer mapping failed. Trying pattern-based matching...")
            
            # Find all encoder layer parameters in source
            for source_key in source_keys:
                if "encoder.layer." in source_key:
                    # Extract layer number
                    parts = source_key.split('.')
                    try:
                        layer_idx = parts[parts.index('layer') + 1]
                        # Replace encoder.layer.X with model.encoder.layers.X
                        target_key = source_key.replace(f"encoder.layer.{layer_idx}", f"model.encoder.layers.{layer_idx}")
                        if target_key in target_keys:
                            mapping[source_key] = target_key
                    except (ValueError, IndexError):
                        continue
    else:
        print("Warning: Could not find encoder layers in model - parameter mapping may be incomplete")
    
    # Add pooler mapping
    mapping["pooler.dense.weight"] = "model.pooler.0.weight"
    mapping["pooler.dense.bias"] = "model.pooler.0.bias"
    
    # Check if we created mapping rules for encoder layers
    encoder_layer_mappings = [k for k in mapping.keys() if "encoder.layer" in k]
    if len(encoder_layer_mappings) == 0:
        print("\nWarning: No encoder layer mappings created!")
        print("Attempting alternative direct mapping approach...")
        
        # Try a direct mapping approach with the correct target layer key
        print("Keys in source state_dict:", list(state_dict.keys())[:10])
        # Based on model inspection, determine the correct target format
        target_layer_format = f"model.encoder.{target_layer_key}"
        
        print(f"Trying direct mapping from 'encoder.layer' to '{target_layer_format}'")
        for source_key, value in state_dict.items():
            if "encoder.layer" in source_key:
                # Convert to our model's format - direct replacement
                target_key = source_key.replace("encoder.layer", target_layer_format)
                if target_key in target_dict and value.shape == target_dict[target_key].shape:
                    mapping[source_key] = target_key
        
        # If we still don't have mappings, try a brute force approach
        if len([k for k in mapping.keys() if "encoder.layer" in k]) == 0:
            print("Still no encoder mappings. Trying brute force parameter mapping...")
            # Find all encoder parameters in source and target based on partial matching
            source_encoder_keys = [k for k in source_keys if "encoder.layer" in k]
            target_encoder_keys = [k for k in target_keys if f"encoder.{target_layer_key}" in k]
            
            # Group by layer and component
            source_groups = {}
            for key in source_encoder_keys:
                parts = key.split('.')
                if len(parts) >= 4:  # encoder.layer.0.component...
                    layer_idx = parts[2]
                    component = parts[3]
                    group_key = f"{layer_idx}_{component}"
                    if group_key not in source_groups:
                        source_groups[group_key] = []
                    source_groups[group_key].append(key)
            
            target_groups = {}
            for key in target_encoder_keys:
                parts = key.split('.')
                if len(parts) >= 5:  # model.encoder.layers.0.component...
                    layer_idx = parts[3]
                    component = parts[4]
                    group_key = f"{layer_idx}_{component}"
                    if group_key not in target_groups:
                        target_groups[group_key] = []
                    target_groups[group_key].append(key)
            
            # Try to match corresponding groups
            for group_key in source_groups:
                if group_key in target_groups:
                    # Match parameters within groups by shape
                    source_params = source_groups[group_key]
                    target_params = target_groups[group_key]
                    
                    # Group by parameter shape
                    source_by_shape = {}
                    for key in source_params:
                        shape = tuple(state_dict[key].shape)
                        if shape not in source_by_shape:
                            source_by_shape[shape] = []
                        source_by_shape[shape].append(key)
                    
                    target_by_shape = {}
                    for key in target_params:
                        shape = tuple(target_dict[key].shape)
                        if shape not in target_by_shape:
                            target_by_shape[shape] = []
                        target_by_shape[shape].append(key)
                    
                    # Match parameters with the same shape
                    for shape in source_by_shape:
                        if shape in target_by_shape:
                            # Simple 1:1 matching if counts match
                            if len(source_by_shape[shape]) == len(target_by_shape[shape]):
                                for s_key, t_key in zip(source_by_shape[shape], target_by_shape[shape]):
                                    mapping[s_key] = t_key
    
    # Print final mapping statistics
    encoder_mappings = len([k for k in mapping.keys() if "encoder.layer" in k])
    print(f"Created {encoder_mappings} encoder layer parameter mappings")
    print(f"Total parameter mappings: {len(mapping)}")
    
    # Map weights using our explicit mapping
    mapped_weights = OrderedDict()
    
    for source_key, value in state_dict.items():
        if source_key in mapping:
            target_key = mapping[source_key]
            if target_key in target_dict:
                # Check shapes match
                if value.shape == target_dict[target_key].shape:
                    mapped_weights[target_key] = value
                else:
                    print(f"Shape mismatch: {source_key} ({value.shape}) -> {target_key} ({target_dict[target_key].shape})")
    
    # Print mapping stats
    mapped_count = len(mapped_weights)
    total_count = len(target_dict)
    mapping_percentage = mapped_count / total_count * 100
    
    print(f"Mapped {mapped_count}/{total_count} parameters ({mapping_percentage:.1f}%)")
    
    # Check for any critical missing keys
    missing_keys = set(target_dict.keys()) - set(mapped_weights.keys())
    
    # Group missing keys by component for clearer reporting
    missing_by_component = {}
    for key in missing_keys:
        parts = key.split(".")
        prefix = ".".join(parts[:3] if len(parts) >= 3 else parts[:2]) 
        missing_by_component[prefix] = missing_by_component.get(prefix, 0) + 1
    
    if missing_keys:
        print(f"Missing {len(missing_keys)} keys")
        print("Missing keys by component:")
        for comp, count in sorted(missing_by_component.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {comp}: {count}")
    
    # Load the weights with strict=True
    model.load_state_dict(mapped_weights, strict=True)
    print("Successfully loaded weights with strict=True")
    
    # Create diagnostics
    success = mapping_percentage > 95  # Success if we mapped at least 95% of parameters
    
    diagnostics = {
        "success": success,
        "mapped_keys": mapped_count,
        "model_keys": total_count,
        "mapping_percentage": mapping_percentage,
        "missing_keys": missing_keys
    }
    
    if success:
        print("✓ Successfully loaded MiniLM weights")
    else:
        print("⚠ Warning: Some parameters could not be mapped")
    
    return model, diagnostics