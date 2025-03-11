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
import numpy as np
import pandas as pd
import os
import sys
import os

# Add the parent directory to sys.path to allow importing src as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import TransformerModel, SentenceEncoder, create_minilm_model, create_modernbert_model
from src.utils import Tokenizer, get_tokenizer_for_model, load_pretrained_weights


def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Transformer models example"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="similarity",
        choices=["similarity", "classification"],
        help="Mode to run example in"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="minilm-l6",
        choices=["minilm-l3", "minilm-l6", "minilm-l12", "modernbert"],
        help="Model to use"
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
        help="Path to data directory for classification"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training/inference"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs for classification"
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
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


class TextClassifier(nn.Module):
    """Text classifier using sentence embeddings."""
    
    def __init__(self, encoder, num_classes):
        """
        Initialize classifier with encoder and classification head.
        
        Args:
            encoder (SentenceEncoder): Encoder for generating sentence embeddings
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.model.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
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
        embeddings = model(encoded_input['input_ids'], encoded_input['attention_mask'])
    
    # Calculate similarities
    cos = torch.nn.CosineSimilarity(dim=1)
    
    print("\nSimilarities:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = cos(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            print(f"Similarity between sentences {i+1} & {j+1}: {sim.item():.4f}")
            print(f"  - '{sentences[i]}'")
            print(f"  - '{sentences[j]}'")
    
    # Find most similar pair
    max_sim = 0
    max_pair = (0, 0)
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
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
    
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
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
    
    # Create sample data
    texts = [
        "Stock markets plunge on fears of global recession",
        "Government announces new economic stimulus package",
        "Tech company launches revolutionary AI assistant",
        "Latest smartphone sales exceed expectations",
        "The football team won the championship last night",
        "Tennis player reaches semifinals after tough match",
        "Olympic committee announces host city for next games",
        "Basketball player signs record-breaking contract",
        "Space agency successfully launches new satellite",
        "Scientists discover new species in Amazon rainforest",
        "Research team makes breakthrough in quantum computing",
        "New medical treatment shows promising results in trials",
    ]
    
    categories = ["Business", "Business", "Technology", "Technology", 
                 "Sports", "Sports", "Sports", "Sports",
                 "Science", "Science", "Science", "Science"]
    
    labels = [
        categories.index("Business") if cat == "Business" else
        categories.index("Sports") if cat == "Sports" else
        categories.index("Science") if cat == "Science" else
        categories.index("Technology") for cat in categories
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        "text": texts,
        "category": categories,
        "label": labels
    })
    
    # Split data
    train = df.sample(frac=0.7, random_state=42)
    temp = df.drop(train.index)
    val = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(val.index)
    
    # Save files
    train.to_csv(os.path.join(data_path, "train.csv"), index=False)
    val.to_csv(os.path.join(data_path, "val.csv"), index=False)
    test.to_csv(os.path.join(data_path, "test.csv"), index=False)
    
    print(f"Created sample data in {data_path}")


def text_classification_example(model, tokenizer, data_path, batch_size=16, epochs=5):
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
    
    print(f"Loaded datasets:")
    print(f"  - Training: {len(train_df)} examples")
    print(f"  - Validation: {len(val_df)} examples")
    print(f"  - Test: {len(test_df)} examples")
    print(f"Classes: {class_names}")
    
    # Create classifier
    classifier = TextClassifier(model, num_classes)
    classifier.to(device)
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer
    )
    
    val_dataset = TextClassificationDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer
    )
    
    test_dataset = TextClassificationDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=2e-5)
    
    print("Training model...")
    for epoch in range(epochs):
        # Training phase
        classifier.train()
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
            outputs = classifier(input_ids, attention_mask)
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
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = classifier(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train loss: {train_loss/len(train_loader):.4f}, accuracy: {100*train_correct/train_total:.2f}%")
        print(f"  Val loss: {val_loss/len(val_loader):.4f}, accuracy: {100*val_correct/val_total:.2f}%")
    
    # Evaluate on test set
    classifier.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
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
        "Latest software update introduces innovative features"
    ]
    
    print("\nExample inference:")
    classifier.eval()
    
    with torch.no_grad():
        encoded = tokenizer.encode(examples)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        outputs = classifier(encoded['input_ids'], encoded['attention_mask'])
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
        model = create_modernbert_model(max_length=args.max_length)
        weights_path = args.weights_path or "modernbert"
        max_length = args.max_length or 512
        
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Get tokenizer
    tokenizer = get_tokenizer_for_model(args.model, max_length=max_length)
    
    # Load weights
    model = load_pretrained_weights(model, weights_path)
    
    # Run example
    if args.mode == "similarity":
        text_similarity_example(model, tokenizer)
    else:
        text_classification_example(
            model, tokenizer, args.data_path, 
            batch_size=args.batch_size, epochs=args.epochs
        )


if __name__ == "__main__":
    main()