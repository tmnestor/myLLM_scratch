import torch
import numpy as np
import re
import inspect
from src.models.modernbert import create_modernbert_model
from src.utils import get_tokenizer_for_model
import os

def get_model_param_count(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def find_modules_by_attributes(model, attr_patterns):
    """Search for modules with specific attributes in their methods or variables"""
    matches = []
    for name, module in model.named_modules():
        for attr_name, pattern in attr_patterns.items():
            # Check if the module has the attribute
            if hasattr(module, attr_name):
                attr_value = getattr(module, attr_name)
                
                # If attribute is a method, get its source code
                if callable(attr_value):
                    try:
                        source = inspect.getsource(attr_value)
                        if re.search(pattern, source):
                            matches.append((name, attr_name, 'method'))
                    except (TypeError, OSError):
                        pass
                # If attribute is a string, check directly
                elif isinstance(attr_value, str) and re.search(pattern, attr_value):
                    matches.append((name, attr_name, 'attribute'))
                
    return matches

def main():
    print("Validating ModernBERT implementation...")
    
    # Get tokenizer first
    tokenizer = get_tokenizer_for_model("modernbert")
    
    # Prepare sample input
    sample_texts = [
        "This is a test sentence to compare model outputs.",
        "Let's see if both models produce similar embeddings."
    ]
    
    # Load custom model - handle the case where it returns a tuple
    print("Loading custom ModernBERT implementation...")
    model_result = create_modernbert_model()
    
    if isinstance(model_result, tuple):
        # If it returns a tuple, the first element is likely the model
        custom_model = model_result[0]
    else:
        custom_model = model_result
    
    custom_param_count = get_model_param_count(custom_model)
    print(f"Loaded custom ModernBERT with {custom_param_count:,} parameters")
    
    # Analyze the model architecture by introspection
    print("\nAnalyzing custom ModernBERT architecture:")
    print("-" * 60)
    
    # Get layer count
    layer_count = 0
    for name, _ in custom_model.named_modules():
        if re.search(r'encoder\.layers\.\d+$', name) or re.search(r'encoder\.\d+$', name):
            layer_idx = int(name.split(".")[-1])
            layer_count = max(layer_count, layer_idx + 1)
    
    if layer_count == 0:
        # Try different pattern if the first one didn't work
        for name, _ in custom_model.named_modules():
            if 'layer' in name.lower() and re.search(r'\d+', name):
                matches = re.findall(r'\d+', name)
                if matches:
                    layer_idx = int(matches[0])
                    layer_count = max(layer_count, layer_idx + 1)
    
    # If still not found, default to 12
    if layer_count == 0:
        layer_count = 12
        print("⚠️ Could not detect layer count, defaulting to 12")
    else:
        print(f"✅ Detected {layer_count} transformer layers")
    
    # Extract other architecture details
    hidden_size = None
    num_attention_heads = None
    intermediate_size = None
    
    # Try to get these from model attributes
    if hasattr(custom_model, 'config'):
        if hasattr(custom_model.config, 'hidden_size'):
            hidden_size = custom_model.config.hidden_size
        if hasattr(custom_model.config, 'num_attention_heads'):
            num_attention_heads = custom_model.config.num_attention_heads
        if hasattr(custom_model.config, 'intermediate_size'):
            intermediate_size = custom_model.config.intermediate_size
    
    # If not found in config, try to infer from parameter shapes
    if hidden_size is None:
        # Look for embedding layer or first layer weight
        for name, param in custom_model.named_parameters():
            if ('embeddings' in name and 'weight' in name) or ('encoder.0' in name and 'weight' in name):
                if len(param.shape) == 2:
                    hidden_size = param.shape[1]
                    print(f"✅ Inferred hidden size from {name}: {hidden_size}")
                    break
    
    if num_attention_heads is None:
        # Try to find attention head count from a parameter name or shape
        for name, module in custom_model.named_modules():
            if hasattr(module, 'num_attention_heads'):
                num_attention_heads = module.num_attention_heads
                print(f"✅ Found num_attention_heads attribute: {num_attention_heads}")
                break
            elif hasattr(module, 'num_heads'):
                num_attention_heads = module.num_heads
                print(f"✅ Found num_heads attribute: {num_attention_heads}")
                break
    
    # Use default values if not found
    if hidden_size is None:
        hidden_size = 768
        print("⚠️ Could not detect hidden size, defaulting to 768")
    
    if num_attention_heads is None:
        num_attention_heads = 12
        print("⚠️ Could not detect number of attention heads, defaulting to 12")
    
    if intermediate_size is None:
        intermediate_size = 3072
        print("⚠️ Could not detect intermediate size, defaulting to 3072")
    
    print(f"\nModel architecture parameters:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Number of layers: {layer_count}")
    print(f"  - Number of attention heads: {num_attention_heads}")
    print(f"  - Intermediate size: {intermediate_size}")
    
    print("\nChecking for ModernBERT-specific features...")
    
    # 1. Check for RoPE embeddings (sin/cos buffers)
    rope_evidence = []
    
    # Look for sin/cos buffers
    for name, buf in custom_model.named_buffers():
        if 'sin' in name.lower() or 'cos' in name.lower() or 'freq' in name.lower():
            rope_evidence.append(f"Buffer named '{name}'")
    
    # Look for specific RoPE patterns in parameter names
    for name, _ in custom_model.named_parameters():
        if 'rotary' in name.lower() or 'rope' in name.lower():
            rope_evidence.append(f"Parameter named '{name}'")
    
    # Look for methods that might implement RoPE
    rope_patterns = {
        'forward': r'rotary|rope|apply_rotary|sin_cached|cos_cached',
        'apply_rotary': r'.',
        'rotary': r'.',
    }
    
    rope_methods = find_modules_by_attributes(custom_model, rope_patterns)
    for name, attr, type in rope_methods:
        rope_evidence.append(f"Module '{name}' has {attr} {type}")
    
    # 2. Check for GLU activation patterns
    glu_evidence = []
    
    # Check for GLU activation patterns in parameters
    for name, param in custom_model.named_parameters():
        if any(pattern in name.lower() for pattern in ['gate_proj', 'up_proj', 'down_proj', 'glu', 'gated']):
            glu_evidence.append(f"Parameter named '{name}'")
    
    # Look for methods that might implement GLU
    glu_patterns = {
        'forward': r'chunk\(2|split\(.*2|glu|gate|act\(.*\)\s*\*',
        'glu': r'.',
    }
    
    glu_methods = find_modules_by_attributes(custom_model, glu_patterns)
    for name, attr, type in glu_methods:
        glu_evidence.append(f"Module '{name}' has {attr} {type}")
    
    # 3. Check for pre-normalization pattern
    prenorm_evidence = []
    
    # Methods that might implement pre-normalization
    prenorm_patterns = {
        'forward': r'normalization.*attention|norm.*attn|layernorm.*attention',
    }
    
    prenorm_methods = find_modules_by_attributes(custom_model, prenorm_patterns)
    for name, attr, type in prenorm_methods:
        prenorm_evidence.append(f"Module '{name}' has {attr} {type}")
    
    # Print feature evidence
    print("\n1. Rotary Position Embeddings (RoPE) evidence:")
    if rope_evidence:
        print("✅ Found evidence of RoPE implementation:")
        for evidence in rope_evidence[:3]:  # Limit to first 3 pieces of evidence
            print(f"  - {evidence}")
        if len(rope_evidence) > 3:
            print(f"  - ... and {len(rope_evidence) - 3} more")
    else:
        print("❌ No evidence of RoPE implementation found")
    
    print("\n2. Gated Linear Units (GLU) evidence:")
    if glu_evidence:
        print("✅ Found evidence of GLU implementation:")
        for evidence in glu_evidence[:3]:  # Limit to first 3 pieces of evidence
            print(f"  - {evidence}")
        if len(glu_evidence) > 3:
            print(f"  - ... and {len(glu_evidence) - 3} more")
    else:
        print("❌ No evidence of GLU implementation found")
    
    print("\n3. Pre-normalization evidence:")
    if prenorm_evidence:
        print("✅ Found evidence of Pre-normalization:")
        for evidence in prenorm_evidence[:3]:  # Limit to first 3 pieces of evidence
            print(f"  - {evidence}")
        if len(prenorm_evidence) > 3:
            print(f"  - ... and {len(prenorm_evidence) - 3} more")
    else:
        print("❌ No evidence of Pre-normalization found")
    
    # 4. Check for combined QKV projection
    combined_qkv = False
    for name, param in custom_model.named_parameters():
        if any(pattern in name.lower() for pattern in ['qkv', 'qkv_proj']):
            print("\n4. Combined QKV projection:")
            print(f"✅ Found combined QKV projection: '{name}'")
            combined_qkv = True
            break
    
    if not combined_qkv:
        print("\n4. Combined QKV projection:")
        print("❌ No evidence of combined QKV projection found")
    
    # Encode sample inputs and run model
    print("\nRunning inference test...")
    inputs = tokenizer.encode(sample_texts)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Set model to eval mode
    custom_model.eval()
    
    # Forward pass
    with torch.no_grad():
        custom_outputs = custom_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Handle different return types
        if isinstance(custom_outputs, tuple):
            custom_hidden_states = custom_outputs[0]
            if len(custom_outputs) > 1:
                custom_pooled = custom_outputs[1]
            else:
                custom_pooled = None
        else:
            custom_hidden_states = custom_outputs
            custom_pooled = None
            
    print("\nOutput shape analysis:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Hidden states shape: {custom_hidden_states.shape}")
    if custom_pooled is not None:
        print(f"Pooled output shape: {custom_pooled.shape}")
    
    # Perform embedding analysis
    print("\nEmbedding analysis:")
    if custom_hidden_states is not None:
        # For vector-like output (shape is [batch_size, embedding_dim])
        if len(custom_hidden_states.shape) == 2:
            sentence_embeddings = custom_hidden_states
        # For sequence outputs (shape is [batch_size, seq_len, embedding_dim])
        elif len(custom_hidden_states.shape) == 3:
            # Get sentence embeddings (mean pooling)
            sentence_embeddings = custom_hidden_states.mean(dim=1)
        else:
            print("⚠️ Unexpected output shape, cannot calculate embeddings")
            return
        
        # Normalize along the last dimension
        normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=-1)
        
        # Calculate similarity between the two sentences
        if normalized_embeddings.shape[0] >= 2:
            sim = torch.nn.functional.cosine_similarity(
                normalized_embeddings[0:1], normalized_embeddings[1:2], dim=-1
            )
            print(f"Similarity between test sentences: {sim.item():.4f}")
    
    # Weight loading validation
    print("\nWeight loading validation:")
    # Check if model parameter count is reasonable
    expected_param_count = layer_count * hidden_size * hidden_size * 12  # Rough estimate
    if custom_param_count > expected_param_count * 0.8:
        print("✅ Parameter count seems reasonable for the architecture")
    else:
        print("⚠️ Parameter count seems low - weights may not be fully loaded")
        
    # Check for zero weights
    zero_params = 0
    total_params = 0
    sample_count = 0
    
    # Sample some parameters to check for zeros
    for _, param in custom_model.named_parameters():
        if param.dim() > 1 and sample_count < 5:  # Sample only a few large parameters
            flat = param.flatten()
            if len(flat) > 100:  # Only check larger params
                indices = torch.randint(0, len(flat), (100,))
                samples = flat[indices]
                zero_params += (samples.abs() < 1e-6).sum().item()
                total_params += len(samples)
                sample_count += 1
    
    if total_params > 0:
        zero_ratio = zero_params / total_params
        print(f"Zero weight ratio in sampled parameters: {zero_ratio:.2%}")
        
        if zero_ratio < 0.01:
            print("✅ Very few zero parameters - weights appear to be properly loaded")
        elif zero_ratio < 0.1:
            print("ℹ️ Some zero parameters - weights may be partially loaded or sparsified")
        else:
            print("⚠️ Many zero parameters - weights may not be properly loaded")
    
    # Final architecture verification summary
    print("\nModernBERT architecture verification summary:")
    
    score = 0
    max_score = 4
    
    if rope_evidence:
        print("✅ RoPE (Rotary Position Embeddings): Implemented")
        score += 1
    else:
        print("❌ RoPE (Rotary Position Embeddings): Not detected")
        
    if glu_evidence:
        print("✅ GLU (Gated Linear Unit): Implemented")
        score += 1
    else:
        print("❌ GLU (Gated Linear Unit): Not detected")
        
    if prenorm_evidence:
        print("✅ Pre-Layer Normalization: Implemented")
        score += 1
    else:
        print("❌ Pre-Layer Normalization: Not detected")
        
    if combined_qkv:
        print("✅ Combined QKV Projection: Implemented")
        score += 1
    else:
        print("❌ Combined QKV Projection: Not detected")
    
    print(f"\nOverall ModernBERT compatibility score: {score}/{max_score}")
    
    if score == max_score:
        print("🎉 Your implementation fully matches the ModernBERT architecture!")
    elif score >= max_score/2:
        print("👍 Your implementation has some ModernBERT features but may be missing others.")
    else:
        print("⚠️ Your implementation appears to be missing several key ModernBERT features.")
    
    print("\nRecommendations for further validation:")
    print("1. Test with the same pretrained weights as the original ModernBERT")
    print("2. Compare performance on downstream tasks")
    print("3. If features are missing, check your implementation against ModernBERT-src/")

if __name__ == "__main__":
    main()