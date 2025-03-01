# MiniLM Sentence Transformer Implementation

A PyTorch implementation of the MiniLM transformer model for generating sentence embeddings, with support for various model variants including MiniLM-L3-v2 and MiniLM-L6-v2.

## Features

- Custom PyTorch implementation of MiniLM architecture
- Support for loading pretrained weights
- Sentence embedding generation with mean pooling
- Configurable model parameters (layers, attention heads, etc.)
- Batched inference support
- Production-ready with max length truncation
- Cosine similarity computation between sentences

## Requirements

```
torch>=1.7.0
transformers>=4.6.0
sentence-transformers>=2.0.0
numpy>=1.19.0
```

## Project Structure

- `model_components.py` - Core transformer components (embeddings, attention, etc.)
- `mini_lm_model.py` - MiniLM model implementation
- `model.py` - High-level encoder interface
- `tokenizer.py` - Wrapper for HuggingFace tokenizer
- `model_loader.py` - Utilities for loading pretrained weights
- `usage_example.py` - Basic usage example
- `usage_example2.py` - Advanced usage with custom implementation

## Environment Setup

1. Set up the environment variable for model paths:
```bash
export LLM_MODELS_PATH=/path/to/your/models
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage with the high-level encoder:

```python
from model import MiniLML3Encoder
from tokenizer import MiniLMTokenizer

# Initialize
encoder = MiniLML3Encoder()
tokenizer = MiniLMTokenizer(max_length=128)

# Example sentences
sentences = [
    "This is an example sentence",
    "Another sentence for embedding"
]

# Generate embeddings
embeddings = encoder.encode(sentences, tokenizer=tokenizer)
```

Advanced usage with custom implementation:

```python
from mini_lm_model import SentenceTransformer
from tokenizer import MiniLMTokenizer
from model_loader import load_pretrained_weights

# Initialize model
model = SentenceTransformer(max_length=128)
tokenizer = MiniLMTokenizer(max_length=128)

# Load pretrained weights
model = load_pretrained_weights(model)

# Tokenize sentences
encoded_input = tokenizer.encode(sentences)

# Generate embeddings
embeddings = model(encoded_input["input_ids"], encoded_input["attention_mask"])
```

## Model Architecture

The implementation includes:
- Embedding layer (token, position, and type embeddings)
- Multi-head self-attention mechanism
- Feed-forward neural networks
- Layer normalization and dropout
- Mean pooling for sentence embeddings

## License

[Your chosen license]

## Acknowledgments

This implementation is based on the MiniLM architecture from Microsoft Research and the HuggingFace Transformers library.