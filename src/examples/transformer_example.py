"""
Simplified example for using the transformer models.

This script demonstrates how to use the transformer models for both
text similarity and classification tasks with a clean, straightforward
interface.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import os
import sys

# Add the parent directory to sys.path to allow importing src as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models import create_minilm_model
from src.models import (
    ModernBERTForSentenceEmbedding,
    create_modernbert_model,
)
from src.utils import (
    get_tokenizer_for_model,
    load_minilm_weights,
    load_modernbert_weights,
    load_pretrained_weights,
)


def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description="Transformer models example")
    parser.add_argument(
        "--mode",
        type=str,
        default="similarity",
        choices=["similarity", "classification"],
        help="Mode to run example in",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="minilm-l6",
        choices=["minilm-l3", "minilm-l6", "minilm-l12", "modernbert"],
        help="Model to use",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to model weights (overrides model choice)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum sequence length (default depends on model)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to data directory for classification",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training/inference"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs for classification",
    )
    return parser.parse_args()


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""

    def __init__(self, texts, labels, tokenizer):
        """
        Initialize dataset with texts and labels.

        Args:
            texts (list): List of text strings
            labels (list): List of integer labels
            tokenizer (Tokenizer): Tokenizer for text processing
        """
        self.texts = texts
        self.labels = labels

        # Pre-encode all texts
        self.encodings = tokenizer.encode(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get pre-encoded tensors for this index
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }


class TextClassifier(nn.Module):
    """Text classifier using sentence embeddings."""

    def __init__(self, encoder, num_classes):
        """
        Initialize classifier with encoder and classification head.

        Args:
            encoder (SentenceEncoder or ModernBERTForSentenceEmbedding): Encoder for generating sentence embeddings
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.encoder = encoder

        # Handle different encoder types
        if hasattr(encoder, "hidden_size"):
            # ModernBERT has hidden_size at the top level
            self.hidden_dim = encoder.hidden_size
        elif hasattr(encoder, "model") and hasattr(encoder.model, "hidden_size"):
            # SentenceEncoder has model.hidden_size
            self.hidden_dim = encoder.model.hidden_size
        else:
            # Default to 768 if not found (most common embedding size)
            print("Warning: Could not determine embedding size, defaulting to 768")
            self.hidden_dim = 768

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        # Get sentence embeddings
        embeddings = self.encoder(input_ids, attention_mask)
        # Pass through classification head
        logits = self.classifier(embeddings)
        return logits


def text_similarity_example(model, tokenizer):
    """
    Run a text similarity example.

    Args:
        model (SentenceEncoder): Model for generating sentence embeddings
        tokenizer (Tokenizer): Tokenizer for processing text
    """
    # Example sentences
    sentences = [
        "It is a beautiful day",
        "It is lovely and sunny outside",
        "What glorious weather we are having today",
        "The weather is terrible and rainy",
    ]

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Move model to device
    model.to(device)

    # Tokenize sentences
    encoded_input = tokenizer.encode(sentences)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(encoded_input["input_ids"], encoded_input["attention_mask"])

    # Calculate similarities
    cos = torch.nn.CosineSimilarity(dim=1)

    print("\nSimilarities:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cos(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            print(f"Similarity between sentences {i + 1} & {j + 1}: {sim.item():.4f}")
            print(f"  - '{sentences[i]}'")
            print(f"  - '{sentences[j]}'")

    # Find most similar pair
    max_sim = 0
    max_pair = (0, 0)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cos(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            if sim.item() > max_sim:
                max_sim = sim.item()
                max_pair = (i, j)

    print("\nMost similar pair:")
    print(f"  - '{sentences[max_pair[0]]}'")
    print(f"  - '{sentences[max_pair[1]]}'")
    print(f"  Similarity: {max_sim:.4f}")


def load_classification_data(data_path):
    """
    Load classification datasets from CSV files.

    Args:
        data_path (str): Path to directory containing data files

    Returns:
        tuple: (train_df, val_df, test_df, class_names)
    """
    # Get absolute path to data directory
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)

    # Check if files exist
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "val.csv")
    test_path = os.path.join(data_path, "test.csv")

    # Force recreation of sample data to use our expanded dataset
    if True or not all(
        os.path.exists(path) for path in [train_path, val_path, test_path]
    ):
        # Create sample data if it doesn't exist
        create_sample_data(data_path)

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Extract class names
    all_categories = pd.concat([train_df, val_df, test_df])["category"].unique()
    class_names = sorted(all_categories)

    return train_df, val_df, test_df, class_names


def create_sample_data(data_path):
    """
    Create sample classification data for demonstration.

    Args:
        data_path (str): Directory to create data files in
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Create larger sample data
    texts = [
        # Business examples (20)
        "Stock markets plunge on fears of global recession",
        "Government announces new economic stimulus package",
        "Company reports record quarterly profits",
        "Merger between two major corporations announced",
        "New CEO appointed to lead struggling retail chain",
        "Stock price falls following disappointing earnings report",
        "Investors concerned about market volatility",
        "Start-up secures multi-million dollar funding round",
        "Bank introduces new financial services for small businesses",
        "Economic indicators suggest strong growth ahead",
        "Trade negotiations continue between major economies",
        "Company launches IPO on stock exchange",
        "Retail sales figures show consumer confidence rising",
        "Central bank announces interest rate decision",
        "Business leaders meet to discuss economic cooperation",
        "Financial analysts predict market correction",
        "New tax regulations impact corporate profits",
        "Company expands operations to international markets",
        "Economic forum discusses global financial challenges",
        "Industry report shows changing consumer trends",
        # Technology examples (20)
        "Tech company launches revolutionary AI assistant",
        "Latest smartphone sales exceed expectations",
        "Software update fixes critical security vulnerabilities",
        "New virtual reality headset hits the market",
        "Tech giant unveils next generation of processors",
        "Researchers develop breakthrough in quantum computing",
        "Electric vehicle manufacturer increases production capacity",
        "Social media platform introduces new privacy features",
        "Streaming service adds interactive content options",
        "Cloud computing provider expands data center network",
        "Robotics company demonstrates advanced automation system",
        "Tech startup develops innovative payment solution",
        "New programming language gains popularity among developers",
        "Companies invest in 6G wireless technology research",
        "Artificial intelligence system beats human experts at complex game",
        "Cybersecurity firm identifies new type of malware attack",
        "Tech conference showcases future consumer gadgets",
        "Smartphone manufacturer unveils foldable screen technology",
        "Voice recognition software improves multilingual capabilities",
        "Tech industry leaders discuss ethical AI development",
        # Sports examples (20)
        "The football team won the championship last night",
        "Tennis player reaches semifinals after tough match",
        "Olympic committee announces host city for next games",
        "Basketball player signs record-breaking contract",
        "Local team advances to national tournament finals",
        "Athlete breaks long-standing world record",
        "Coach fired after disappointing season performance",
        "Injury concerns for star player ahead of crucial match",
        "Sports league announces rule changes for next season",
        "Team overcomes deficit to win in dramatic comeback",
        "Young athlete named rookie of the year",
        "Historic rivalry renewed in upcoming championship match",
        "Sports venue undergoes major renovation",
        "International competition draws record viewership",
        "Player announces retirement after illustrious career",
        "Team secures sponsorship deal with major brand",
        "Athlete speaks out on social issues affecting sports",
        "Underdog team causes major upset in tournament",
        "Sports federation investigates allegations of misconduct",
        "New technology introduced to improve referee decisions",
        # Science examples (20)
        "Space agency successfully launches new satellite",
        "Scientists discover new species in Amazon rainforest",
        "Research team makes breakthrough in quantum computing",
        "New medical treatment shows promising results in trials",
        "Astronomers observe unusual celestial phenomenon",
        "Climate study reveals accelerating environmental changes",
        "Research team sequences genome of endangered species",
        "Archeologists uncover ancient civilization artifacts",
        "Scientists develop new renewable energy technology",
        "Medical researchers identify potential cancer treatment",
        "Study reveals insights into human cognitive development",
        "Marine biologists document previously unknown deep-sea creatures",
        "Laboratory demonstrates successful nuclear fusion experiment",
        "Geologists predict seismic activity patterns",
        "Environmental scientists monitor pollution levels",
        "Researchers create advanced materials with unique properties",
        "Neuroscientists map previously unknown brain functions",
        "Space telescope captures images of distant galaxy formation",
        "Botanists discover plant species with medicinal properties",
        "Scientific consortium publishes climate change projections",
    ]

    # Create matching categories list
    categories = (
        ["Business"] * 20 + ["Technology"] * 20 + ["Sports"] * 20 + ["Science"] * 20
    )

    # Convert categories to numeric labels
    unique_categories = sorted(set(categories))
    labels = [unique_categories.index(cat) for cat in categories]

    # Create DataFrame
    df = pd.DataFrame({"text": texts, "category": categories, "label": labels})

    # Split data
    train = df.sample(frac=0.7, random_state=42)
    temp = df.drop(train.index)
    val = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(val.index)

    # Save files
    train.to_csv(os.path.join(data_path, "train.csv"), index=False)
    val.to_csv(os.path.join(data_path, "val.csv"), index=False)
    test.to_csv(os.path.join(data_path, "test.csv"), index=False)

    print(f"Created sample data in {data_path} with {len(df)} examples")


def text_classification_example(model, tokenizer, data_path, batch_size=16, epochs=10):
    """
    Run a text classification example.

    Args:
        model (SentenceEncoder): Model for generating sentence embeddings
        tokenizer (Tokenizer): Tokenizer for processing text
        data_path (str): Path to data directory
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
    """
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load data
    train_df, val_df, test_df, class_names = load_classification_data(data_path)
    num_classes = len(class_names)

    print("Loaded datasets:")
    print(f"  - Training: {len(train_df)} examples")
    print(f"  - Validation: {len(val_df)} examples")
    print(f"  - Test: {len(test_df)} examples")
    print(f"Classes: {class_names}")

    # First, try eval-only to see how well the pretrained model already performs
    with torch.no_grad():
        # Create simple classifier head for evaluation
        eval_classifier = TextClassifier(model, num_classes)
        eval_classifier.to(device)
        eval_classifier.eval()

        # Create a simple dataset for testing pretrained performance
        test_examples = [
            "Stock markets plunge due to economic instability",
            "Scientists discover new planet in nearby solar system",
            "Team scores winning goal in championship match",
            "New smartphone features cutting-edge processor technology",
        ]
        test_labels = [0, 1, 2, 3]  # Business, Science, Sports, Technology

        # Encode and predict
        encoded = tokenizer.encode(test_examples)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = eval_classifier(encoded["input_ids"], encoded["attention_mask"])
        _, predictions = torch.max(outputs, 1)

        # Calculate accuracy
        correct = (predictions == torch.tensor(test_labels).to(device)).sum().item()
        print(
            f"\nPretrained model accuracy on sample examples: {correct / len(test_examples) * 100:.2f}%"
        )
        for i, (text, pred, true) in enumerate(
            zip(test_examples, predictions, test_labels)
        ):
            print(f'Example {i + 1}: "{text}"')
            print(f"  True class: {class_names[true]}")
            print(f"  Predicted: {class_names[pred.item()]}")

    # Create classifier for training with enhanced head
    classifier = TextClassifier(model, num_classes)

    # Modify the classifier head with the hidden_dim we already determined
    classifier.classifier = nn.Sequential(
        nn.Linear(classifier.hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )

    classifier.to(device)

    # Create datasets
    train_dataset = TextClassificationDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer
    )

    val_dataset = TextClassificationDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer
    )

    test_dataset = TextClassificationDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train model with learning rate scheduler
    criterion = nn.CrossEntropyLoss()

    # Use a higher learning rate (5x) for ModernBERT fine-tuning
    base_lr = 1e-4 if isinstance(model, ModernBERTForSentenceEmbedding) else 2e-5
    print(f"Using base learning rate: {base_lr}")

    # Group parameters: higher learning rate for classifier layer
    classifier_parameters = [
        p for n, p in classifier.named_parameters() if "classifier" in n
    ]
    encoder_parameters = [
        p for n, p in classifier.named_parameters() if "classifier" not in n
    ]

    # Use parameter groups with different learning rates
    optimizer = optim.AdamW(
        [
            {"params": encoder_parameters, "lr": base_lr},
            {
                "params": classifier_parameters,
                "lr": base_lr * 5,
            },  # 5x learning rate for classifier
        ],
        weight_decay=0.001,
    )

    # Warm up then linear decay scheduler works well for transformer fine-tuning
    num_warmup_steps = len(train_loader) * 1  # 1 epoch of warmup
    num_training_steps = len(train_loader) * epochs

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("Training model...")
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = classifier(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update lr every batch (common in warm-up schedulers)

            # Calculate metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = classifier(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Print metrics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"  Train loss: {train_loss / len(train_loader):.4f}, accuracy: {100 * train_correct / train_total:.2f}%"
        )
        print(
            f"  Val loss: {val_loss / len(val_loader):.4f}, accuracy: {100 * val_correct / val_total:.2f}%"
        )
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")

    # Evaluate on test set
    classifier.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = classifier(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"\nTest accuracy: {test_accuracy:.2f}%")

    # Example inference
    examples = [
        "New trade deal signed between two major economies",
        "Athlete breaks world record at championship event",
        "Researchers publish findings on new energy source",
        "Latest software update introduces innovative features",
    ]

    print("\nExample inference:")
    classifier.eval()

    with torch.no_grad():
        encoded = tokenizer.encode(examples)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = classifier(encoded["input_ids"], encoded["attention_mask"])
        _, predictions = torch.max(outputs, 1)

    for text, pred in zip(examples, predictions):
        print(f"Text: {text}")
        print(f"Predicted class: {class_names[pred.item()]}")
        print("---")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create model based on arguments
    if args.model == "minilm-l3":
        model = create_minilm_model(num_layers=3, max_length=args.max_length)
        weights_path = args.weights_path or "paraphrase-MiniLM-L3-v2"
        max_length = args.max_length or 128

    elif args.model == "minilm-l6":
        model = create_minilm_model(num_layers=6, max_length=args.max_length)
        weights_path = args.weights_path or "all-MiniLM-L6-v2"
        max_length = args.max_length or 128

    elif args.model == "minilm-l12":
        model = create_minilm_model(num_layers=12, max_length=args.max_length)
        weights_path = args.weights_path or "all-MiniLM-L12-v2"
        max_length = args.max_length or 128

    elif args.model == "modernbert":
        model = create_modernbert_model(max_length=args.max_length, num_layers=12)
        weights_path = args.weights_path or "ModernBERT-base"
        max_length = args.max_length or 512

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Get tokenizer
    tokenizer = get_tokenizer_for_model(args.model, max_length=max_length)

    # Load weights using the appropriate loader
    if "minilm" in args.model.lower():
        print(f"Using specialized MiniLM loader for {args.model}")
        model, diagnostics = load_minilm_weights(model, weights_path)
    elif "modernbert" in args.model.lower():
        print(f"Using specialized ModernBERT loader for {args.model}")
        model, diagnostics = load_modernbert_weights(model, weights_path)
    else:
        model, diagnostics = load_pretrained_weights(model, weights_path)

    # Handle loading diagnostics
    if not diagnostics["success"]:
        print(
            "Warning: Model weights loading had issues. Some functionality may be limited."
        )
        print(
            f"Loaded {diagnostics['mapped_keys']} out of {diagnostics['model_keys']} parameters."
        )

    # Continue only if loading was at least partially successful
    if diagnostics.get("mapped_keys", 0) == 0:
        raise RuntimeError("Failed to load any model weights. Cannot continue.")

    # Run example
    if args.mode == "similarity":
        text_similarity_example(model, tokenizer)
    else:
        text_classification_example(
            model,
            tokenizer,
            args.data_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
