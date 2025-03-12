"""
Simplified model weight loading module.

This module provides a streamlined approach to loading pretrained weights
into transformer models without unnecessary complexity.
"""

import os
import torch
from collections import OrderedDict


def load_pretrained_weights(model, weights_path):
    """
    Load pretrained weights into a model.
    
    Args:
        model (nn.Module): Model to load weights into
        weights_path (str): Path to pretrained weights or model name
        
    Returns:
        nn.Module: Model with loaded weights
    """
    # Check if weights_path is a local path
    if not os.path.exists(weights_path):
        # Try using environment variable
        env_path = os.environ.get("LLM_MODELS_PATH")
        if env_path:
            full_path = os.path.join(env_path, weights_path)
            if os.path.exists(full_path):
                weights_path = full_path
            else:
                raise ValueError(f"Weights path not found: {weights_path} or {full_path}")
        else:
            raise ValueError(f"Weights path not found: {weights_path} and LLM_MODELS_PATH not set")
    
    # Determine model file based on common naming conventions
    if os.path.isdir(weights_path):
        # Check for common model file formats
        possible_filenames = [
            "pytorch_model.bin",
            "model.safetensors",  # HuggingFace now commonly uses safetensors
            "model.pt", 
            "model.bin",
            "model.pth",
            "weights.bin",
            "weights.pt"
        ]
        
        for filename in possible_filenames:
            model_file = os.path.join(weights_path, filename)
            if os.path.exists(model_file):
                break
        else:
            # Print the directory contents to help debug
            print(f"Available files in {weights_path}:")
            for file in os.listdir(weights_path):
                print(f"  - {file}")
            raise ValueError(f"Could not find model weights file in {weights_path}")
    else:
        model_file = weights_path
    
    print(f"Loading weights from: {model_file}")
    
    # Check for safetensors format
    if model_file.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            from collections import OrderedDict
            
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
    
    # Check if we need to extract from a container format
    if not isinstance(state_dict, OrderedDict) and hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    
    # Determine based on state_dict keys what kind of model this is
    is_sentence_transformers = any("sentence_bert" in key for key in state_dict.keys())
    is_huggingface = any("bert" in key.lower() for key in state_dict.keys())
    
    # Check the key structure to handle special cases (debug information)
    sample_keys = list(state_dict.keys())[:5]
    print(f"State dict contains keys like: {sample_keys}")
    
    # Check model type to apply specific mappings
    model_type = "generic"
    if hasattr(model, "model"):
        if hasattr(model.model, "model_type"):
            model_type = model.model.model_type
    
    print(f"Loading weights for model type: {model_type}")
    
    # Special pre-processing for specific model types
    if 'all-MiniLM' in str(weights_path) or 'MiniLM' in str(weights_path):
        print("Detected MiniLM weights, applying specific mappings...")
        # Explicitly check required parameters in our model
        if hasattr(model, "model"):
            base_model = model.model
            
            # Get all layers from our model for reference
            model_layers = []
            for name, _ in base_model.named_modules():
                model_layers.append(name)
            print(f"Model contains layers like: {model_layers[:5]}")
    elif 'ModernBERT' in str(weights_path):
        print("Detected ModernBERT weights, applying specific mappings...")
        # Explicitly check required parameters in our model
        if hasattr(model, "model"):
            base_model = model.model
            
            # Get all layers from our model for reference
            model_layers = []
            for name, _ in base_model.named_modules():
                model_layers.append(name)
            print(f"Model contains layers like: {model_layers[:5]}")
            
            # ModernBERT has a significantly different architecture
            print("ModernBERT model has an incompatible architecture with our current implementation.")
            print("A complete specialized loader would be required to properly map these weights.")
    
    # Special pre-processing for common model structure mismatches
    if hasattr(model, "model") and any("encoder.layer" in k for k in state_dict.keys()):
        print("Pre-processing weight keys for layers/layer naming difference...")
        processed_state_dict = OrderedDict()
        
        for key, value in state_dict.items():
            processed_key = key
            # Handle encoder.layer vs encoder.layers difference
            if "encoder.layer" in key:
                processed_key = key.replace("encoder.layer", "encoder.layers")
            
            # Add model. prefix if needed for SentenceEncoder
            if hasattr(model, "model") and not processed_key.startswith("model."):
                processed_key = "model." + processed_key
                
            processed_state_dict[processed_key] = value
            
            # If it's the original key, keep that too as a fallback
            if processed_key != key:
                processed_state_dict[key] = value
                
        state_dict = processed_state_dict
    
    # Simplified mapping function 
    mapped_weights = map_weights(state_dict, model, 
                                is_sentence_transformers=is_sentence_transformers,
                                is_huggingface=is_huggingface)
    
    # Load mapped weights with strict=True
    model.load_state_dict(mapped_weights, strict=True)
    print("Successfully loaded weights with strict=True")
    
    # Report success and any potential issues
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped_weights.keys())
    missing_keys = model_keys - mapped_keys
    unexpected_keys = mapped_keys - model_keys
    
    # Add detailed diagnostics
    print(f"Model has {len(model_keys)} parameters, loaded {len(mapped_keys)} parameters")
    
    if missing_keys:
        # Get counts by parameter type for better diagnostics
        key_prefixes = {}
        for key in missing_keys:
            # Extract prefix (e.g., "model.encoder.layers.0")
            parts = key.split('.')
            prefix = '.'.join(parts[:3] if len(parts) >= 3 else parts[:2])
            key_prefixes[prefix] = key_prefixes.get(prefix, 0) + 1
            
        print(f"Missing keys: {len(missing_keys)} parameters")
        print(f"Missing keys by component:")
        for prefix, count in sorted(key_prefixes.items(), key=lambda x: x[1], reverse=True):
            if count > 0:  # Only show components with missing keys
                print(f"  - {prefix}: {count} parameters")
                
        # Check if missing keys are critical
        critical_components = ['embeddings', 'pooler', 'encoder.layers.0']
        critical_missing = False
        for comp in critical_components:
            if any(comp in key for key in missing_keys):
                critical_missing = True
                print(f"WARNING: Missing keys in critical component: {comp}")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)} parameters")
        # Show some examples of unexpected keys
        print(f"Examples: {list(unexpected_keys)[:5]}")
    
    # Evaluate loading success
    success = len(missing_keys) == 0 or (len(missing_keys) / len(model_keys) < 0.05)
    
    if success:
        if len(missing_keys) == 0:
            print(f"✓ Successfully loaded all pretrained weights")
        else:
            print(f"✓ Successfully loaded pretrained weights (missing {len(missing_keys)} non-critical parameters)")
    else:
        print(f"⚠ Warning: Model loaded with {len(missing_keys)} missing parameters - model may not work correctly")
    
    # Return a tuple with model and diagnostics info
    diagnostics = {
        "success": success,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "model_keys": len(model_keys),
        "mapped_keys": len(mapped_keys),
    }
    
    return model, diagnostics


def map_weights(state_dict, model, is_sentence_transformers=False, is_huggingface=False):
    """
    Map weights from pretrained models to the target model.
    
    This is a simplified mapping approach that handles the most common cases
    without excessive complexity.
    
    Args:
        state_dict (OrderedDict): Source state dictionary
        model (nn.Module): Target model
        is_sentence_transformers (bool): If True, source is sentence-transformers format
        is_huggingface (bool): If True, source is HuggingFace format
        
    Returns:
        OrderedDict: Mapped weights dictionary
    """
    mapped_weights = OrderedDict()
    target_dict = model.state_dict()
    
    # Auto-detect format if not specified
    if not is_sentence_transformers and not is_huggingface:
        # Check if this looks like a sentence-transformers model (highest priority)
        if any(k.startswith("0.") for k in state_dict.keys()):
            is_sentence_transformers = True
        # Check if this looks like a HuggingFace model
        elif any(k.startswith(("bert.", "roberta.")) or "encoder.layer" in k for k in state_dict.keys()):
            is_huggingface = True
    
    # Print detected model format for diagnostics
    if is_sentence_transformers:
        print("Detected sentence-transformers format")
    elif is_huggingface:
        print("Detected HuggingFace format")
    else:
        print("Using direct weight mapping")
    
    # Create lookup table for our model's parameter shapes
    target_shapes = {k: v.shape for k, v in target_dict.items()}
    
    # Handle sentence-transformers format
    if is_sentence_transformers:
        # Most sentence-transformers models use "0." prefix for the base model
        for key, value in state_dict.items():
            # Example mapping: "0.bert.embeddings.word_embeddings.weight" -> "model.embeddings.word_embeddings.weight"
            if key.startswith("0."):
                new_key = "model." + key[2:]
                if "bert" in new_key:
                    new_key = new_key.replace("bert.", "")
                
                if new_key in target_dict:
                    mapped_weights[new_key] = value
                    
    # Handle HuggingFace format
    elif is_huggingface:
        for key, value in state_dict.items():
            # First, clean up prefixes like "bert." 
            if key.startswith("bert."):
                cleaned_key = key[5:]  # Remove "bert."
            else:
                cleaned_key = key
            
            # Map encoder layers - our code uses layers instead of layer
            if "encoder.layer." in cleaned_key:
                # Convert "encoder.layer.0.attention" -> "encoder.layers.0.attention"
                cleaned_key = cleaned_key.replace("encoder.layer.", "encoder.layers.")
            
            # Map pooler
            if "pooler.dense.weight" in cleaned_key:
                cleaned_key = "pooler.0.weight"  # Map to sequential pooler
            if "pooler.dense.bias" in cleaned_key:
                cleaned_key = "pooler.0.bias"
                
            # Add model prefix for SentenceEncoder (our architecture)
            if hasattr(model, "model"):
                target_key = "model." + cleaned_key
                if target_key in target_dict:
                    mapped_weights[target_key] = value
                    continue
            
            # Try direct match
            if cleaned_key in target_dict:
                mapped_weights[cleaned_key] = value
    
    # For direct matching (simpler cases)
    else:
        for key, value in state_dict.items():
            # Try direct match first
            if key in target_dict:
                mapped_weights[key] = value
                continue
                
            # Try with model prefix
            model_key = "model." + key
            if model_key in target_dict:
                mapped_weights[model_key] = value
                continue
    
    # This is where we'll add a simplified fix for our model architecture
    # Specifically handling the MiniLM/ModernBERT model key mapping
    
    # Special case for our encoder layers naming
    if len(mapped_weights) < len(target_dict) * 0.8:  # If we're missing a significant number of weights
        # Check if this looks like a HuggingFace transformer model
        if any("encoder.layer" in k for k in state_dict.keys()):
            print("Attempting to map encoder.layer to encoder.layers...")
            
            for key, value in state_dict.items():
                if "encoder.layer" in key:
                    # Convert key from HuggingFace format to our format
                    new_key = key.replace("encoder.layer", "encoder.layers")
                    
                    # Add model. prefix for SentenceEncoder
                    if hasattr(model, "model") and not new_key.startswith("model."):
                        new_key = "model." + new_key
                        
                    # Check if the transformed key exists in our model
                    if new_key in target_dict:
                        mapped_weights[new_key] = value
    
    print(f"Mapped {len(mapped_weights)}/{len(target_dict)} parameters ({len(mapped_weights)/len(target_dict)*100:.1f}%)")
    return mapped_weights