import torch
import argparse
from mini_lm_model import SentenceTransformer
from tokenizer import TransformerTokenizer
from model_loader import load_pretrained_weights


def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Test transformer model implementations with different models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=[
            "paraphrase-MiniLM-L3-v2",
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "modernbert",
        ],
        help="Model to use for embedding generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum sequence length for input tokens",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["minilm", "modernbert"],
        help="Model architecture type (if not specified, will be inferred from model name)",
    )
    return parser.parse_args()


def test_custom_implementation(model_name="all-MiniLM-L6-v2", max_length=64, model_type=None):
    # Determine model type if not provided
    if model_type is None:
        if "MiniLM" in model_name:
            model_type = "minilm"
        elif model_name.lower() == "modernbert":
            model_type = "modernbert"
        else:
            raise ValueError(f"Cannot infer model_type from model_name: {model_name}. Please specify model_type explicitly.")
    
    # Determine model parameters based on selected model and model type
    if model_type.lower() == "minilm":
        if model_name == "all-MiniLM-L12-v2":
            num_hidden_layers = 12
        elif model_name in ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]:
            num_hidden_layers = 6
        else:  # Default for L3 models
            num_hidden_layers = 3
            
        # Default settings for MiniLM models
        hidden_size = 384
        intermediate_size = 1536
        
    elif model_type.lower() == "modernbert":
        # ModernBERT uses different architecture parameters
        num_hidden_layers = 12  # Standard BERT size
        hidden_size = 768  # Standard BERT size
        intermediate_size = 3072  # Standard BERT size
        # Adjust max_length for ModernBERT if not explicitly set
        if max_length == 64:  # If using the default
            max_length = 512  # ModernBERT typically supports longer sequences
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(
        f"Using {model_type} model: {model_name} with {num_hidden_layers} layers and max_length={max_length}"
    )

    # Initialize our custom model instance with appropriate configuration
    base_model = SentenceTransformer(
        model_type=model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        max_length=max_length
    )
    
    # Use the unified tokenizer class for both model types
    tokenizer = TransformerTokenizer(
        model_name=model_name, 
        model_type=model_type,
        max_length=max_length
    )

    # Load pretrained weights, and assign to a new variable for clarity
    loaded_model = load_pretrained_weights(base_model, model_name=model_name, model_type=model_type)

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
    test_custom_implementation(
        model_name=args.model, 
        max_length=args.max_length,
        model_type=args.model_type
    )
