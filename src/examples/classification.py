import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to sys.path to allow importing src as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import SentenceEncoder, create_minilm_model, create_modernbert_model
from src.utils import Tokenizer, get_tokenizer_for_model, load_pretrained_weights


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
        self.hidden_dim = self.base_model.model.pooler['dense'].out_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
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
    
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Initialize dataset with texts and labels.
        
        Args:
            texts (list): List of text strings
            labels (list): List of integer labels
            tokenizer (TransformerTokenizer): Tokenizer for text processing
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-encode all texts to ensure consistent tensor sizes
        self.encodings = self.tokenizer.encode(texts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Get pre-encoded tensors for this index
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
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
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Print epoch metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {100 * train_correct / train_total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {100 * val_correct / val_total:.2f}%")
        print("-" * 50)
    
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
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
            class_mask = (true_labels == i)
            if np.sum(class_mask) > 0:
                class_accuracy = 100 * np.sum((predictions == i) & class_mask) / np.sum(class_mask)
                print(f"Class '{class_name}' Accuracy: {class_accuracy:.2f}%")
    
    return test_accuracy


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
        print("Dataset files not found. Please run dataset_generator.py first.")
        print("  python dataset_generator.py")
        raise FileNotFoundError(f"Dataset files not found in {data_dir}")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df


def text_classification_example(model_name="all-MiniLM-L6-v2", max_length=64, 
                                model_type=None, batch_size=16, epochs=5, 
                                learning_rate=2e-5, data_dir="data",
                                generate_data=False):
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
        try:
            # Import here to avoid circular import
            from dataset_generator import generate_text_classification_datasets
            generate_text_classification_datasets()
        except ImportError:
            print("Could not import dataset_generator. Please run it separately:")
            print("  python dataset_generator.py")
            
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
    tokenizer = TransformerTokenizer(
        model_name=model_name, 
        model_type=model_type,
        max_length=max_length
    )

    # Initialize base model
    base_model = SentenceTransformer(
        model_type=model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        max_length=max_length
    )
    
    # Load pretrained weights
    base_model = load_pretrained_weights(base_model, model_name=model_name, model_type=model_type)
    
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
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer, max_length
    )
    
    val_dataset = TextClassificationDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer, max_length
    )
    
    test_dataset = TextClassificationDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer, max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train the model
    print("Starting training...")
    classifier = train_model(
        classifier, train_loader, val_loader, device, 
        epochs=epochs, learning_rate=learning_rate
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_accuracy = evaluate_model(classifier, test_loader, device, class_names)
    
    # Example of inference with new texts
    print("\nExample inference:")
    example_texts = [
        "New trade deal signed between two major economies",
        "Athlete breaks world record at championship event",
        "Researchers publish findings on new renewable energy source",
        "Latest software update introduces innovative features"
    ]
    
    # Tokenize examples
    encoded_examples = tokenizer.encode(example_texts)
    
    # Move to device
    encoded_examples = {k: v.to(device) for k, v in encoded_examples.items()}
    
    # Get predictions
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(encoded_examples['input_ids'], 
                             encoded_examples['attention_mask'])
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
        generate_data=args.generate_data
    )