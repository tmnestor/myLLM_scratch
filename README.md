# MiniLM Sentence Transformer Implementation

A PyTorch implementation of the MiniLM transformer model for generating sentence embeddings, supporting multiple model variants including:

- `paraphrase-MiniLM-L3-v2`
- `all-MiniLM-L6-v2`
- `paraphrase-MiniLM-L6-v2`
- `all-MiniLM-L12-v2`

This implementation provides efficient sentence embedding generation with customizable parameters, making it suitable for both experimentation and production use.

## Features

- Custom PyTorch implementation of the MiniLM architecture
- Support for multiple model variants (L3, L6, and L12)
- Loading pretrained weights from HuggingFace models
- Sentence embedding generation with mean pooling
- Configurable model parameters (layers, attention heads, etc.)
- GPU/MPS acceleration with automatic device detection
- Batched inference support
- Production-ready with configurable max sequence length
- Cosine similarity computation between embeddings

## Requirements

```
torch>=1.7.0
transformers>=4.6.0
numpy>=1.19.0
```

## Architecture Overview

This implementation consists of several interconnected modules that work together to provide a complete sentence embedding solution:

### Core Modules

#### `model_components.py`

Contains the building blocks of the transformer architecture:
- `Embeddings`: Combines token, position, and type embeddings with normalization
- `SelfAttention`: Multi-head self-attention mechanism
- `AttentionOutput`: Processes attention outputs with dense layer and normalization
- `TransformerLayer`: Combines attention and feed-forward networks
- `TransformerEncoder`: Stacks multiple transformer layers

#### `mini_lm_model.py`

Implements the main model classes:
- `MiniLMModel`: Core implementation supporting L3, L6, and L12 variants
- `SentenceTransformer`: Wrapper providing sentence embedding functionality with mean pooling

#### `model_loader.py`

Handles loading pretrained weights:
- Maps weights from HuggingFace models to our custom implementation
- Validates architecture compatibility
- Provides detailed feedback about weight loading process
- Supports all model variants

#### `tokenizer.py`

Provides tokenization functionality:
- Wraps HuggingFace tokenizers for MiniLM models
- Handles encoding and decoding of text
- Supports configurable sequence length


## Core Modules

This implementation consists of several interconnected modules that work together to provide a complete sentence embedding solution:

### model_components.py

Contains the building blocks of the transformer architecture:
- `Embeddings`: Combines token, position, and type embeddings with normalization
- `SelfAttention`: Multi-head self-attention mechanism
- `AttentionOutput`: Processes attention outputs with dense layer and normalization
- `TransformerLayer`: Combines attention and feed-forward networks
- `TransformerEncoder`: Stacks multiple transformer layers

### mini_lm_model.py

Implements the main model classes:
- `MiniLMModel`: Core implementation supporting L3, L6, and L12 variants
- `SentenceTransformer`: Wrapper providing sentence embedding functionality with mean pooling

### model_loader.py

Handles loading pretrained weights:
- Maps weights from HuggingFace models to our custom implementation
- Validates architecture compatibility
- Provides detailed feedback about weight loading process
- Supports all model variants

### tokenizer.py

Provides tokenization functionality:
- Wraps HuggingFace tokenizers for MiniLM models
- Handles encoding and decoding of text
- Supports configurable sequence length

## Module Interaction

1. **Initialization Flow**:
   - `SentenceTransformer` creates a `MiniLMModel` with appropriate parameters
   - `MiniLMModel` initializes `Embeddings` and `TransformerEncoder`
   - `TransformerEncoder` creates the specified number of `TransformerLayer` instances

2. **Weight Loading Flow**:
   - `load_pretrained_weights` loads pretrained weights from local files
   - Maps weights from the pretrained model to the custom architecture
   - Validates shape compatibility and reports any issues

3. **Inference Flow**:
   - Tokenizer converts text to token IDs and attention masks
   - `SentenceTransformer` passes these through the model layers
   - Input embeddings → Transformer layers → Mean pooling → Normalized embeddings

## Environment Setup

1. Set up the environment variable for model paths:
```bash
export LLM_MODELS_PATH=/path/to/your/models
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the model files from HuggingFace to your LLM_MODELS_PATH directory.

## Usage

### Basic Usage

```python
from mini_lm_model import SentenceTransformer
from tokenizer import MiniLMTokenizer
from model_loader import load_pretrained_weights

# Initialize model with appropriate parameters
model = SentenceTransformer(num_hidden_layers=6, max_length=64)
tokenizer = MiniLMTokenizer(model_name="all-MiniLM-L6-v2", max_length=64)

# Load pretrained weights
model = load_pretrained_weights(model, model_name="all-MiniLM-L6-v2")

# Prepare sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "I love natural language processing"
]

# Tokenize sentences
encoded_input = tokenizer.encode(sentences)

# Move to appropriate device (CUDA, MPS, or CPU)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
model.to(device)
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

# Generate embeddings
with torch.no_grad():
    embeddings = model(encoded_input["input_ids"], encoded_input["attention_mask"])

# Use embeddings for similarity, clustering, etc.
```

### Command-Line Usage

The package includes a command-line interface for testing:

```bash
# Using the default model (all-MiniLM-L6-v2) with default max_length (64)
python usage_example.py

# Using a specific model
python usage_example.py --model all-MiniLM-L12-v2

# Using a specific model and max length
python usage_example.py --model paraphrase-MiniLM-L3-v2 --max_length 128
```

## Model Selection Guide

This implementation supports several MiniLM variants:

| Model | Layers | Hidden Size | Parameters | Best For |
|-------|--------|-------------|------------|----------|
| paraphrase-MiniLM-L3-v2 | 3 | 384 | ~22M | Fast inference, low resource usage |
| all-MiniLM-L6-v2 | 6 | 384 | ~33M | Balanced performance (default) |
| paraphrase-MiniLM-L6-v2 | 6 | 384 | ~33M | Better paraphrase quality |
| all-MiniLM-L12-v2 | 12 | 384 | ~55M | Higher quality embeddings |

## Performance Considerations

- Smaller models (L3) provide faster inference with slightly lower quality embeddings
- Larger models (L12) provide higher quality embeddings at the cost of slower inference
- Setting an appropriate `max_length` balances quality and speed (default: 64)
- GPU or MPS acceleration is automatically used when available

## License

[Your chosen license]

## Acknowledgments

This implementation is based on the MiniLM architecture from Microsoft Research and the HuggingFace Transformers library.