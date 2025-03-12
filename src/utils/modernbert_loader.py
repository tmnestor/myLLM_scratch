"""
Specialized loader for ModernBERT model weights.

This module provides a dedicated loader for ModernBERT models that correctly maps
the weights from the ModernBERT format to our custom model architecture.
"""

import os
import torch
from collections import OrderedDict
from safetensors import safe_open


def load_modernbert_weights(model, weights_path):
    """
    Load ModernBERT weights into our custom model architecture.
    
    Args:
        model: Our ModernBERTForSentenceEmbedding model
        weights_path: Path to the ModernBERT weights (directory or file)
        
    Returns:
        tuple: (model with loaded weights, diagnostics)
    """
    # Resolve weights path
    resolved_path = weights_path
    
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
        # ModernBERT typically uses safetensors format
        possible_filenames = [
            "model.safetensors",
            "pytorch_model.bin",
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
    
    print(f"Loading ModernBERT weights from: {model_file}")
    
    # Load weights based on file format
    if model_file.endswith('.safetensors'):
        try:
            # Load weights directly from safetensors
            state_dict = OrderedDict()
            with safe_open(model_file, framework="pt") as f:
                # Get list of all keys for reference
                all_keys = list(f.keys())
                
                # Load weights
                for key in all_keys:
                    state_dict[key] = f.get_tensor(key)
            
            print(f"Loaded {len(state_dict)} tensors from safetensors file")
            
            # Print some sample keys to understand the structure
            print(f"Sample source keys: {list(state_dict.keys())[:5]}")
            
            # Print keys related to embeddings and layers to help debugging
            embedding_keys = [k for k in state_dict.keys() if "embedding" in k.lower()]
            position_keys = [k for k in state_dict.keys() if "pos" in k.lower()]
            layer_keys = [k for k in state_dict.keys() if "layer" in k.lower()][:5]
            print(f"Embedding keys: {embedding_keys}")
            print(f"Position-related keys: {position_keys}")
            print(f"Sample layer keys: {layer_keys}")
            
            # Map weights correctly
            mapped_weights = map_modernbert_weights(state_dict, model)
            
            # Load the mapped weights
            missing_keys, unexpected_keys = model.load_state_dict(mapped_weights, strict=False)
            
            # Report statistics
            model_keys = set(model.state_dict().keys())
            mapped_keys = set(mapped_weights.keys())
            missing_keys = model_keys - mapped_keys
            
            print(f"Mapped {len(mapped_keys)}/{len(model_keys)} parameters ({len(mapped_keys)/len(model_keys)*100:.1f}%)")
            
            if missing_keys:
                print(f"Missing {len(missing_keys)} parameters")
                
                # Group missing keys by component for better diagnostics
                missing_by_component = {}
                for key in missing_keys:
                    # Extract prefix (e.g., "model.embeddings")
                    parts = key.split('.')
                    if len(parts) >= 3:
                        prefix = parts[0] + '.' + parts[1]
                        if len(parts) >= 4:
                            prefix += '.' + parts[2]
                    else:
                        prefix = key
                    
                    if prefix not in missing_by_component:
                        missing_by_component[prefix] = 0
                    missing_by_component[prefix] += 1
                
                # Print missing keys by component
                for component, count in sorted(missing_by_component.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {component}: {count} parameters")
            
            # Create diagnostics
            success = len(mapped_keys) > 0.9 * len(model_keys)
            
            diagnostics = {
                "success": success,
                "model_keys": len(model_keys),
                "mapped_keys": len(mapped_keys),
                "missing_keys": missing_keys
            }
            
            if success:
                print("✓ Successfully loaded ModernBERT weights")
            else:
                print("⚠ Warning: Some parameters could not be mapped")
            
            return model, diagnostics
            
        except ImportError:
            print("safetensors package not found, trying to load with torch...")
            state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    else:
        # Regular PyTorch format
        state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    
    # For PyTorch format (fallback)
    mapped_weights = map_modernbert_weights(state_dict, model)
    
    # Load the mapped weights
    missing_keys, unexpected_keys = model.load_state_dict(mapped_weights, strict=False)
    
    # Report statistics
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped_weights.keys())
    missing_keys = model_keys - mapped_keys
    
    print(f"Mapped {len(mapped_keys)}/{len(model_keys)} parameters ({len(mapped_keys)/len(model_keys)*100:.1f}%)")
    
    # Create diagnostics
    success = len(mapped_keys) > 0.9 * len(model_keys)
    
    diagnostics = {
        "success": success,
        "model_keys": len(model_keys),
        "mapped_keys": len(mapped_keys),
        "missing_keys": missing_keys
    }
    
    if success:
        print("✓ Successfully loaded ModernBERT weights")
    else:
        print("⚠ Warning: Some parameters could not be mapped")
    
    return model, diagnostics


def map_modernbert_weights(state_dict, model):
    """
    Map ModernBERT weights to our custom model structure.
    
    Args:
        state_dict: Source state dict from ModernBERT
        model: Our ModernBERTForSentenceEmbedding model
        
    Returns:
        OrderedDict: Mapped weights
    """
    target_dict = model.state_dict()
    mapped_weights = OrderedDict()
    
    # Analyze key patterns in source
    source_keys = list(state_dict.keys())
    
    # Define direct mappings for known parameters
    direct_mappings = {
        # Embeddings
        'model.embeddings.tok_embeddings.weight': 'model.embeddings.tok_embeddings.weight',
        'model.embeddings.norm.weight': 'model.embeddings.norm.weight',
        
        # Position embeddings might have different names - try various possibilities
        'model.embeddings.pos_embeddings.weight': 'model.embeddings.position_embeddings.weight',
        'model.embeddings.position_embeddings.weight': 'model.embeddings.position_embeddings.weight',
        'model.pos_embeddings.weight': 'model.embeddings.position_embeddings.weight',
        
        # Pooler
        'head.dense.weight': 'model.pooler.dense.weight',
        'head.norm.weight': 'model.pooler.norm.weight',
        
        # Final norm
        'model.final_norm.weight': 'model.encoder.final_norm.weight',
        
        # Layers structure
        'model.layers': 'model.encoder.layers',
    }
    
    # Apply direct mappings
    for source_key, target_key in direct_mappings.items():
        if source_key in state_dict and target_key in target_dict:
            if state_dict[source_key].shape == target_dict[target_key].shape:
                mapped_weights[target_key] = state_dict[source_key]
                print(f"Mapped {source_key} -> {target_key}")
    
    # Try to find position embeddings by various patterns
    position_embedding_target = 'model.embeddings.position_embeddings.weight'
    if position_embedding_target not in mapped_weights and position_embedding_target in target_dict:
        # 1. Try common variations of position embedding names
        possible_pos_keys = [
            'model.embeddings.pos_embeddings.weight',
            'model.embeddings.position_embeddings.weight',
            'model.pos_embeddings.weight',
            'embeddings.position_embeddings.weight',
            'embeddings.pos_embeddings.weight',
            'wpe.weight'  # GPT-style position embeddings
        ]
        
        for pos_key in possible_pos_keys:
            if pos_key in state_dict and state_dict[pos_key].shape == target_dict[position_embedding_target].shape:
                mapped_weights[position_embedding_target] = state_dict[pos_key]
                print(f"Found position embeddings: {pos_key} -> {position_embedding_target}")
                break
                
        # 2. If still not found, search for any key containing position/pos and embedding
        if position_embedding_target not in mapped_weights:
            for source_key in source_keys:
                if ('pos' in source_key.lower() or 'position' in source_key.lower()) and 'embedding' in source_key.lower():
                    if source_key in state_dict and state_dict[source_key].shape == target_dict[position_embedding_target].shape:
                        mapped_weights[position_embedding_target] = state_dict[source_key]
                        print(f"Found position embeddings: {source_key} -> {position_embedding_target}")
                        break
    
    # Handle encoder layers mappings dynamically
    # Extract the number of layers in our model
    num_layers = 0
    for key in target_dict.keys():
        if 'model.encoder.layers.' in key:
            layer_idx = int(key.split('.')[3])  # Extract layer index
            num_layers = max(num_layers, layer_idx + 1)
    
    print(f"Target model has {num_layers} layers")
    
    # Check the model.layers structure specifically
    if 'model.encoder.layers' in target_dict and 'model.layers' in state_dict:
        # Map the entire layers object if shapes match
        if state_dict['model.layers'].shape == target_dict['model.encoder.layers'].shape:
            mapped_weights['model.encoder.layers'] = state_dict['model.layers']
            print(f"Mapped entire layers structure: model.layers -> model.encoder.layers")
    
    # Define layer-level mappings
    for layer_idx in range(num_layers):
        source_prefix = f'model.layers.{layer_idx}'
        target_prefix = f'model.encoder.layers.{layer_idx}'
        
        # Define mappings for each layer component
        layer_mappings = {
            # Attention
            f'{source_prefix}.attn.Wqkv.weight': f'{target_prefix}.attn.Wqkv.weight',
            f'{source_prefix}.attn.Wo.weight': f'{target_prefix}.attn.Wo.weight',
            
            # Attention norm
            f'{source_prefix}.attn_norm.weight': f'{target_prefix}.attn_norm.weight',
            
            # MLP
            f'{source_prefix}.mlp.Wi.weight': f'{target_prefix}.mlp.Wi.weight',
            f'{source_prefix}.mlp.Wo.weight': f'{target_prefix}.mlp.Wo.weight',
            
            # MLP norm
            f'{source_prefix}.mlp_norm.weight': f'{target_prefix}.mlp_norm.weight',
        }
        
        # Apply layer mappings
        for source_key, target_key in layer_mappings.items():
            if source_key in state_dict and target_key in target_dict:
                # Match shapes - no resizing needed with correct architecture
                if state_dict[source_key].shape == target_dict[target_key].shape:
                    mapped_weights[target_key] = state_dict[source_key]
                else:
                    print(f"Shape mismatch for {source_key}: {state_dict[source_key].shape} vs {target_dict[target_key].shape}")
    
    # Final pass - check for any remaining key mismatches with the same shape
    # This helps find parameters that are just named differently
    if len(mapped_weights) < len(target_dict):
        missing_target_keys = set(target_dict.keys()) - set(mapped_weights.keys())
        
        # Special handling for rotary embedding buffers - these aren't learned weights,
        # but rather geometric sequences that can be generated on-the-fly
        rotary_keys = [k for k in missing_target_keys if 'rotary_emb.inv_freq' in k]
        if rotary_keys:
            print(f"Initializing {len(rotary_keys)} rotary embedding buffers")
            
            # Initialize rotary embedding buffers directly from model
            for key in rotary_keys:
                mapped_weights[key] = target_dict[key].clone()
            
            # Update missing keys
            missing_target_keys = missing_target_keys - set(rotary_keys)
        
        # For each remaining missing parameter, look for any source parameter with matching shape
        for target_key in missing_target_keys:
            target_shape = target_dict[target_key].shape
            
            # Skip complex structures like full modules
            if len(target_shape) == 0:  # It's a module, not a tensor
                continue
                
            # Find all source keys with matching shape
            matching_source_keys = [
                k for k in source_keys 
                if k not in [direct_mappings.get(k, '') for k in direct_mappings]  # Not already mapped
                and state_dict[k].shape == target_shape
            ]
            
            # If we found a single matching key, it's likely the right one
            if len(matching_source_keys) == 1:
                source_key = matching_source_keys[0]
                mapped_weights[target_key] = state_dict[source_key]
                print(f"Shape-matched: {source_key} -> {target_key}")
            
            # If we found multiple, use heuristics based on similar names
            elif len(matching_source_keys) > 1:
                best_match = None
                best_score = 0
                
                # Simple name similarity heuristic
                for source_key in matching_source_keys:
                    # Calculate similarity as number of shared substrings
                    source_parts = source_key.split('.')
                    target_parts = target_key.split('.')
                    
                    score = sum(1 for s in source_parts if any(s in t for t in target_parts))
                    
                    if score > best_score:
                        best_score = score
                        best_match = source_key
                
                if best_match and best_score > 0:
                    mapped_weights[target_key] = state_dict[best_match]
                    print(f"Best shape & name match: {best_match} -> {target_key}")
    
    # Summary
    print(f"Mapped {len(mapped_weights)}/{len(target_dict)} parameters ({len(mapped_weights)/len(target_dict)*100:.1f}%)")
    
    return mapped_weights