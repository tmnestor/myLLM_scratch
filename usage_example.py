import torch
from mini_lm_model import MiniLMModel, SentenceTransformer
from tokenizer import MiniLMTokenizer
from model_loader import load_pretrained_weights

def test_custom_implementation():
    # Initialize our custom model with production-friendly max_length
    max_length = 128  # Set a lower max_length for production
    model = SentenceTransformer(max_length=max_length)
    tokenizer = MiniLMTokenizer(max_length=max_length)

    # Option 1: Load pretrained weights
    model = load_pretrained_weights(model)

    # Example sentences (including a longer one to demonstrate truncation)
    sentences = [
        "It is a beautify day",
        "It is lovely and sunny outside",
        "What glorious weather we are having today",
    ]

    # Tokenize the sentences
    encoded_input = tokenizer.encode(sentences)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(encoded_input["input_ids"], encoded_input["attention_mask"])

    # Show results
    print(f"\nInput shapes:")
    print(f"input_ids shape: {encoded_input['input_ids'].shape}")
    print(f"attention_mask shape: {encoded_input['attention_mask'].shape}")
    print(f"Output embedding shape: {embeddings.shape}")

    # Compute similarities
    cos = torch.nn.CosineSimilarity(dim=1)
    sim_1_2 = cos(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    sim_1_3 = cos(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
    print(f"\nSimilarities:")
    print(f"Similarity between short sentences (1 & 2): {sim_1_2.item():.4f}")
    print(f"Similarity between short and long sentences (1 & 3): {sim_1_3.item():.4f}")

if __name__ == "__main__":
    test_custom_implementation()
