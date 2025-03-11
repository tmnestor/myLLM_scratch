# Sentence Transformer Implementation

A clean, modular implementation of transformer models for generating sentence embeddings and text classification.

## Features

- Unified transformer architecture configurable for different model sizes
- Support for MiniLM (L3, L6, L12) and ModernBERT models
- Sentence embedding generation for similarity tasks
- Text classification capabilities
- Modern codebase with clear organization
- Support for HuggingFace models and SafeTensors format

## Project Structure

```
sentence_transformer/
├── src/                   # Main package
│   ├── models/            # Model implementations
│   │   ├── components.py  # Transformer building blocks
│   │   └── transformer.py # Main transformer classes
│   ├── utils/             # Utilities
│   │   ├── model_loader.py # Weight loading functions
│   │   └── tokenizer.py   # Tokenization utilities
│   └── examples/          # Example applications
│       ├── text_similarity.py   # Sentence similarity example
│       ├── classification.py    # Text classification example
│       └── transformer_example.py # Combined example script
├── data/                  # Data directory for examples
├── setup.py               # Package setup file
└── README.md              # This file
```

## Installation

```bash
# Install from source
pip install -e .
```

## Usage

### Sentence Similarity

```bash
# Run text similarity example
python src/examples/transformer_example.py --mode similarity --model minilm-l6
```

### Text Classification

```bash
# Run text classification example
python src/examples/transformer_example.py --mode classification --model minilm-l6 --epochs 5
```

## Model Selection Guide

| Model | Description | Best For |
|-------|-------------|----------|
| minilm-l3 | 3-layer model (fastest) | Resource-constrained environments |
| minilm-l6 | 6-layer model (balanced) | General-purpose use |
| minilm-l12 | 12-layer model (highest quality) | High-quality embeddings |
| modernbert | Full BERT architecture | State-of-the-art performance |

## Code Example

```python
from src.models import create_minilm_model
from src.utils import get_tokenizer_for_model, load_pretrained_weights

# Create model and tokenizer
model = create_minilm_model(num_layers=6)
tokenizer = get_tokenizer_for_model("minilm-l6")

# Load weights
model = load_pretrained_weights(model, "all-MiniLM-L6-v2")

# Generate embeddings
texts = ["This is a sample sentence.", "Another example text."]
encoded = tokenizer.encode(texts)
embeddings = model(encoded['input_ids'], encoded['attention_mask'])

# Calculate similarity
import torch
cosine = torch.nn.CosineSimilarity(dim=1)
similarity = cosine(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
print(f"Similarity: {similarity.item():.4f}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.