# Standard library imports
import argparse
import os
import sys

# Add the parent directory to sys.path to allow importing src as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Local application imports
import src.models.minilm.model
import src.models.modernbert.model
from src.utils import (
    get_tokenizer_for_model,
    load_minilm_weights,
    load_modernbert_weights,
)

# Get the model creation functions directly from their modules
create_minilm_model = src.models.minilm.model.create_minilm_model
create_modernbert_model = src.models.modernbert.model.create_modernbert_model


class SentenceClassifier(nn.Module):
    """
    Text classification model that uses sentence embeddings from transformer models.
    Takes sentence embeddings from a pretrained model and adds a classification head.
    """

    def __init__(self, base_model, num_classes):
        """
        Initialize the sentence classifier with a base transformer model
        and a classification head.

        Args:
            base_model (nn.Module): Pretrained sentence transformer model
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.base_model = base_model

        # Get hidden dimension from the base model
        # Handle different model architectures
        if hasattr(base_model, "hidden_size"):
            self.hidden_dim = base_model.hidden_size
        elif hasattr(base_model, "model") and hasattr(base_model.model, "hidden_size"):
            self.hidden_dim = base_model.model.hidden_size
        elif hasattr(base_model, "model") and hasattr(base_model.model, "pooler"):
            if isinstance(base_model.model.pooler, dict):
                self.hidden_dim = base_model.model.pooler["dense"].out_features
            elif hasattr(base_model.model.pooler, "dense"):
                self.hidden_dim = base_model.model.pooler.dense.out_features
            else:
                # Default to 768 if we can't determine
                print("Warning: Could not determine hidden size, defaulting to 768")
                self.hidden_dim = 768
        else:
            # Default size for most models
            print("Warning: Could not determine hidden size, defaulting to 768")
            self.hidden_dim = 768

        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),  # Smaller intermediate size
            nn.ReLU(),
            nn.Dropout(0.2),  # Higher dropout for better regularization
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token ids of shape [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_length]

        Returns:
            torch.Tensor: Logits for each class of shape [batch_size, num_classes]
        """
        # Get sentence embeddings from base model
        embeddings = self.base_model(input_ids, attention_mask)
        # Pass through classification head
        logits = self.classifier(embeddings)
        return logits


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks.
    Tokenizes text and returns tensors suitable for the model.
    """

    def __init__(self, texts, labels, tokenizer, max_length=None):
        """
        Initialize dataset with texts and labels.

        Args:
            texts (list): List of text strings
            labels (list): List of integer labels
            tokenizer (Tokenizer): Tokenizer for text processing
            max_length (int, optional): Maximum sequence length. If None, uses tokenizer's default.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.max_length

        # Pre-encode all texts to ensure consistent tensor sizes
        self.encodings = self.tokenizer.encode(texts)

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


def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Text classification example using transformer models"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,  # Increased for more training time
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,  # Increased learning rate for better training
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset files",
    )
    parser.add_argument(
        "--generate_data",
        action="store_true",
        help="Generate new dataset files before training",
    )
    return parser.parse_args()


def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5):
    """
    Train the classification model.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization

    Returns:
        model (nn.Module): Trained model
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Group parameters for better optimization - higher learning rate for classifier
    encoder_params = list(model.base_model.parameters())
    classifier_params = list(model.classifier.parameters())

    # Use parameter groups with different learning rates
    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": learning_rate},
            {
                "params": classifier_params,
                "lr": learning_rate * 5,
            },  # 5x learning rate for classifier
        ],
        weight_decay=0.01,
    )  # Add weight decay for regularization

    # Learning rate scheduler with warmup then linear decay
    num_warmup_steps = len(train_loader)  # 1 epoch of warmup
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

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
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
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update lr with scheduler

            # Calculate metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Print epoch metrics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"  Train loss: {train_loss / len(train_loader):.4f}, "
            f"accuracy: {100 * train_correct / train_total:.2f}%"
        )
        print(
            f"  Val loss: {val_loss / len(val_loader):.4f}, "
            f"accuracy: {100 * val_correct / val_total:.2f}%"
        )
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")

    return model


def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate the classification model on test data.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to evaluate on
        class_names (list, optional): List of class names for display

    Returns:
        float: Test accuracy
    """
    model.eval()
    test_correct = 0
    test_total = 0

    # Store predictions and true labels for confusion matrix
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # If class names are provided, print per-class metrics
    if class_names:
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        true_labels = np.array(all_labels)

        # Calculate per-class accuracy
        for i, class_name in enumerate(class_names):
            class_mask = true_labels == i
            if np.sum(class_mask) > 0:
                class_accuracy = (
                    100 * np.sum((predictions == i) & class_mask) / np.sum(class_mask)
                )
                print(f"Class '{class_name}' Accuracy: {class_accuracy:.2f}%")

    return test_accuracy


def create_sample_data(data_path):
    """
    Create sample classification data for demonstration.

    Args:
        data_path (str): Directory to create data files in
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Create sample data with examples from 4 categories
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

    # Split data: 70% train, 15% val, 15% test
    train = df.sample(frac=0.7, random_state=42)
    temp = df.drop(train.index)
    val = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(val.index)

    # Save files
    train.to_csv(os.path.join(data_path, "train.csv"), index=False)
    val.to_csv(os.path.join(data_path, "val.csv"), index=False)
    test.to_csv(os.path.join(data_path, "test.csv"), index=False)

    print(f"Created sample data in {data_path} with {len(df)} examples")


def load_datasets(data_dir):
    """
    Load datasets from CSV files.

    Args:
        data_dir (str): Directory containing the dataset files

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Get absolute path to data directory
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)

    # Check if data files exist
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        print(f"Dataset files not found in {data_dir}. Generating sample data...")
        create_sample_data(data_dir)

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def text_classification_example(
    model_name="all-MiniLM-L6-v2",
    max_length=64,
    model_type=None,
    batch_size=16,
    epochs=5,
    learning_rate=2e-5,
    data_dir="data",
    generate_data=False,
):
    """
    Example of using transformer models for multi-class text classification.

    Args:
        model_name (str): Name of the pretrained model to use
        max_length (int): Maximum sequence length
        model_type (str, optional): Model architecture type
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        data_dir (str): Directory containing the dataset files
        generate_data (bool): Generate new dataset files before training
    """
    # Generate datasets if requested
    if generate_data:
        # Use the create_sample_data function to generate datasets
        # This uses the same data creation logic seen in transformer_example.py
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
        create_sample_data(data_path)
        print(f"Generated sample data in {data_path}")

    # Determine model type if not provided
    if model_type is None:
        if "MiniLM" in model_name:
            model_type = "minilm"
        elif model_name.lower() == "modernbert":
            model_type = "modernbert"
        else:
            raise ValueError(
                f"Cannot infer model_type from model_name: {model_name}. Please specify model_type explicitly."
            )

    # Determine model parameters based on selected model and model type
    if model_type.lower() == "minilm":
        if model_name == "all-MiniLM-L12-v2":
            num_hidden_layers = 12
        elif model_name in ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]:
            num_hidden_layers = 6
        else:  # Default for L3 models
            num_hidden_layers = 3

        # Default is 6 layers for MiniLM

    elif model_type.lower() == "modernbert":
        # ModernBERT uses different architecture parameters
        num_hidden_layers = 12  # Standard BERT size
        # Adjust max_length for ModernBERT if not explicitly set
        if max_length == 64:  # If using the default
            max_length = 512  # ModernBERT typically supports longer sequences
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(
        f"Using {model_type} model: {model_name} with {num_hidden_layers} layers and max_length={max_length}"
    )

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Initialize tokenizer
    tokenizer = get_tokenizer_for_model(model_name, max_length=max_length)

    # Initialize base model based on model type
    if model_type.lower() == "minilm":
        base_model = create_minilm_model(
            num_layers=num_hidden_layers, max_length=max_length
        )
    elif model_type.lower() == "modernbert":
        base_model = create_modernbert_model(
            num_layers=num_hidden_layers, max_length=max_length
        )

    # Load pretrained weights based on model type
    if model_type.lower() == "minilm":
        base_model, _ = load_minilm_weights(base_model, weights_path=model_name)
        print("Used specialized MiniLM loader")
    elif model_type.lower() == "modernbert":
        # ModernBERT requires special path handling
        if model_name.lower() == "modernbert":
            model_path = "ModernBERT-base"  # Use standard model name
        else:
            model_path = model_name
        base_model, _ = load_modernbert_weights(base_model, weights_path=model_path)
        print("Used specialized ModernBERT loader")
    else:
        # Use specialized loader based on model type
        if "minilm" in model_name.lower():
            base_model, _ = load_minilm_weights(base_model, weights_path=model_name)
            print("Used specialized MiniLM loader for generic model")
        elif "modernbert" in model_name.lower():
            base_model, _ = load_modernbert_weights(base_model, weights_path=model_name)
            print("Used specialized ModernBERT loader for generic model")
        else:
            raise ValueError(
                f"Unsupported model: {model_name}. Must be MiniLM or ModernBERT variant."
            )

    # Load datasets
    train_df, val_df, test_df = load_datasets(data_dir)

    # Print dataset information
    print(f"Loaded {len(train_df)} training examples")
    print(f"Loaded {len(val_df)} validation examples")
    print(f"Loaded {len(test_df)} test examples")

    # Define class names
    class_names = ["Sports", "Business", "Science", "Technology"]
    num_classes = len(class_names)

    # Create classifier model
    classifier = SentenceClassifier(base_model, num_classes)
    classifier.to(device)

    # Create datasets and dataloaders
    train_dataset = TextClassificationDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length
    )

    val_dataset = TextClassificationDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length
    )

    test_dataset = TextClassificationDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train the model
    print("Starting training...")
    classifier = train_model(
        classifier,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    evaluate_model(classifier, test_loader, device, class_names)

    # Example of inference with new texts
    print("\nExample inference:")
    example_texts = [
        "New trade deal signed between two major economies",
        "Athlete breaks world record at championship event",
        "Researchers publish findings on new renewable energy source",
        "Latest software update introduces innovative features",
    ]

    # Tokenize examples
    encoded_examples = tokenizer.encode(example_texts)

    # Move to device
    encoded_examples = {k: v.to(device) for k, v in encoded_examples.items()}

    # Get predictions
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(
            encoded_examples["input_ids"], encoded_examples["attention_mask"]
        )
        _, predictions = torch.max(outputs, 1)

    # Show results
    for i, text in enumerate(example_texts):
        predicted_class = class_names[predictions[i].item()]
        print(f"Text: {text}")
        print(f"Predicted class: {predicted_class}")
        print("-" * 30)


if __name__ == "__main__":
    args = parse_arguments()
    text_classification_example(
        model_name=args.model,
        max_length=args.max_length,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        generate_data=args.generate_data,
    )
