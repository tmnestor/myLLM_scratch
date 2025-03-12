#!/usr/bin/env python3
"""
Simple test script to verify the package is working correctly.
"""

import torch
from src.models import create_minilm_model
from src.utils import get_tokenizer_for_model, load_pretrained_weights, load_minilm_weights

def test_sentence_similarity():
    """
    Test sentence similarity using MiniLM model.
    """
    print("Testing sentence similarity with MiniLM-L6 model...")
    
    # Create model and tokenizer
    model = create_minilm_model(num_layers=6)
    tokenizer = get_tokenizer_for_model("minilm-l6")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    # Load weights using specialized MiniLM loader
    print("Loading model weights...")
    model, diagnostics = load_minilm_weights(model, "all-MiniLM-L6-v2")
    
    # Check loading diagnostics
    if not diagnostics['success']:
        print("Warning: Model weights loading had issues, but continuing for testing.")
        print(f"Loaded {diagnostics['mapped_keys']} out of {diagnostics['model_keys']} parameters.")
    
    model.to(device)
    
    # Sample sentences
    texts = [
        "The weather today is beautiful.",
        "It's a lovely sunny day outside.",
        "The stock market crashed yesterday.",
    ]
    
    print("\nGenerating embeddings for test sentences:")
    for text in texts:
        print(f"  - {text}")
    
    # Generate embeddings
    encoded = tokenizer.encode(texts)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        embeddings = model(encoded['input_ids'], encoded['attention_mask'])
    
    # Calculate similarities
    cosine = torch.nn.CosineSimilarity(dim=1)
    
    print("\nCalculating similarities:")
    similarity_1_2 = cosine(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    similarity_1_3 = cosine(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
    similarity_2_3 = cosine(embeddings[1].unsqueeze(0), embeddings[2].unsqueeze(0))
    
    print(f"Similarity between sentences 1 & 2: {similarity_1_2.item():.4f}")
    print(f"Similarity between sentences 1 & 3: {similarity_1_3.item():.4f}")
    print(f"Similarity between sentences 2 & 3: {similarity_2_3.item():.4f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_sentence_similarity()