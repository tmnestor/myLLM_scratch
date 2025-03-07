import torch
import argparse
from mini_lm_model import SentenceTransformer
from tokenizer import MiniLMTokenizer
from model_loader import load_pretrained_weights


def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description="Test MiniLM implementation with different models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-MiniLM-L6-v2",
        choices=["paraphrase-MiniLM-L3-v2", "all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
        help="Model to use for embedding generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum sequence length for input tokens"
    )
    return parser.parse_args()


def test_custom_implementation(model_name="all-MiniLM-L6-v2", max_length=64):
    
    # Determine model parameters based on selected model
    if model_name == "all-MiniLM-L12-v2":
        num_hidden_layers = 12
    elif model_name in ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]:
        num_hidden_layers = 6
    else:  # Default for L3 models
        num_hidden_layers = 3
    
    print(f"Using model: {model_name} with {num_hidden_layers} layers and max_length={max_length}")

    # Initialize our custom model instance with appropriate layer count
    base_model = SentenceTransformer(max_length=max_length, num_hidden_layers=num_hidden_layers)
    tokenizer = MiniLMTokenizer(model_name=model_name, max_length=max_length)

    # Load pretrained weights, and assign to a new variable for clarity
    loaded_model = load_pretrained_weights(base_model, model_name=model_name)

    # Then use loaded_model in place of model:
    # Example sentences (including a longer one to demonstrate truncation)
    sentences = [
        "It is a beautify day",
        "It is lovely and sunny outside",
        "What glorious weather we are having today",
    ]

    encoded_input = tokenizer.encode(sentences)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    loaded_model.to(device)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        embeddings = loaded_model(
            encoded_input["input_ids"], encoded_input["attention_mask"]
        )

    print("\nInput shapes:")
    print(f"input_ids shape: {encoded_input['input_ids'].shape}")
    print(f"attention_mask shape: {encoded_input['attention_mask'].shape}")
    print(f"Output embedding shape: {embeddings.shape}")

    cos = torch.nn.CosineSimilarity(dim=1)
    sim_1_2 = cos(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    sim_1_3 = cos(embeddings[0].unsqueeze(0), embeddings[2].unsqueeze(0))
    print("\nSimilarities:")
    print(f"Similarity between short sentences (1 & 2): {sim_1_2.item():.4f}")
    print(f"Similarity between short and long sentences (1 & 3): {sim_1_3.item():.4f}")


if __name__ == "__main__":
    args = parse_arguments()
    test_custom_implementation(model_name=args.model, max_length=args.max_length)
