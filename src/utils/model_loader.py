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
    
    # Simplified mapping function 
    mapped_weights = map_weights(state_dict, model, 
                                is_sentence_transformers=is_sentence_transformers,
                                is_huggingface=is_huggingface)
    
    # Load mapped weights
    model.load_state_dict(mapped_weights, strict=False)
    
    # Report success and any potential issues
    missing_keys = set(model.state_dict().keys()) - set(mapped_weights.keys())
    unexpected_keys = set(mapped_weights.keys()) - set(model.state_dict().keys())
    
    if missing_keys or unexpected_keys:
        print(f"Loaded pretrained weights with issues:")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"Successfully loaded pretrained weights into custom model")
    
    return model


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
                
            # Map encoder layers
            if "encoder.layer." in cleaned_key:
                # Convert "encoder.layer.0.attention" -> "encoder.layers.0.attention"
                cleaned_key = cleaned_key.replace("encoder.layer.", "encoder.layers.")
            
            # Map pooler
            if "pooler.dense.weight" in cleaned_key:
                cleaned_key = "pooler.0.weight"  # Map to sequential pooler
            if "pooler.dense.bias" in cleaned_key:
                cleaned_key = "pooler.0.bias"
                
            # Add model prefix for SentenceEncoder
            if hasattr(model, "model"):
                target_key = "model." + cleaned_key
                if target_key in target_dict:
                    mapped_weights[target_key] = value
            else:
                if cleaned_key in target_dict:
                    mapped_weights[cleaned_key] = value
    
    # For direct matching (simpler cases)
    else:
        for key, value in state_dict.items():
            if key in target_dict:
                mapped_weights[key] = value
            elif "model." + key in target_dict:
                mapped_weights["model." + key] = value
                
    return mapped_weights