from transformers import AutoTokenizer, AutoConfig
import os

class TransformerTokenizer:
    """General purpose tokenizer wrapper that can load any Hugging Face tokenizer.
    Provides a common interface for different model types including MiniLM and ModernBERT.
    """
    
    def __init__(self, model_name=None, model_type=None, max_length=128):
        """Initialize the tokenizer for a specific model type.
        
        Args:
            model_name (str, optional): Model name such as 'paraphrase-MiniLM-L3-v2' or 'modernbert'.
            model_type (str, optional): Model architecture type ('minilm' or 'modernbert').
                If None, will be inferred from model_name.
            max_length (int, optional): Maximum sequence length. Defaults to 128.
            
        Raises:
            ValueError: If model_name or max_length is invalid
            EnvironmentError: If required environment variables are not set
            FileNotFoundError: If the model directory is not found
        """
        # Infer model type if not provided
        if model_type is None:
            if model_name and "MiniLM" in model_name:
                model_type = "minilm"
            elif model_name and model_name.lower() == "modernbert":
                model_type = "modernbert"
            else:
                raise ValueError(f"Cannot infer model_type from model_name: {model_name}. Please specify model_type explicitly.")
                
        # Basic validation
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
            
        # Standardize model type
        model_type = model_type.lower()
        
        # Validate model type and standardize model name
        if model_type == "minilm":
            supported_models = [
                "paraphrase-MiniLM-L3-v2",
                "all-MiniLM-L6-v2", 
                "paraphrase-MiniLM-L6-v2",
                "all-MiniLM-L12-v2"
            ]
            
            if model_name not in supported_models:
                raise ValueError(f"Unsupported MiniLM model: {model_name}. Must be one of {supported_models}")
                
        elif model_type == "modernbert":
            # Always use modernbert-base for consistency
            model_name = "modernbert-base"
            
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'minilm' or 'modernbert'")
            
        # Always use LLM_MODELS_PATH, but adjust folder name for ModernBERT
        env_var_name = "LLM_MODELS_PATH"
        
        # Get model path from environment variable
        model_path = os.environ.get(env_var_name)
        if not model_path:
            raise EnvironmentError(f"{env_var_name} environment variable is not set")
            
        # Form the full model path
        if model_type == "minilm":
            model_folder = model_name.split('/')[-1] if '/' in model_name else model_name
        else:  # modernbert - use correct capitalization for the folder
            model_folder = "ModernBERT-base"
            
        local_model_path = os.path.join(model_path, model_folder)
            
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model folder not found: {local_model_path}")
        
        # Load tokenizer directly (no need for config validation)
        print(f"Loading tokenizer from local path: {local_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        
        # Store max_length and vocab size
        self.max_length = max_length
        self._vocab_size = self.tokenizer.vocab_size
        self.model_type = model_type.lower()
        
        print(f"Loaded {model_type} tokenizer with vocabulary size: {self._vocab_size}")
        print(f"Maximum sequence length set to: {self.max_length}")

    def encode(self, sentences, padding=True, truncation=True, return_tensors="pt"):
        """
        Tokenize sentences and prepare them for the model

        Args:
            sentences: Single sentence or list of sentences
            padding: Whether to pad sequences to the longest in the batch
            truncation: Whether to truncate sequences that exceed max length
            return_tensors: Return PyTorch tensors ('pt') or TensorFlow ('tf')

        Returns:
            Dictionary with input_ids, attention_mask, and token_type_ids (if applicable)
        """
        # Validate input
        if sentences is None:
            raise ValueError("sentences cannot be None")
        if not isinstance(sentences, (str, list)):
            raise ValueError(f"sentences must be a string or list, got {type(sentences)}")
        if isinstance(sentences, list) and not all(isinstance(s, str) for s in sentences):
            raise ValueError("All items in sentences list must be strings")
        if return_tensors not in ["pt", "tf", "np"]:
            raise ValueError(f"return_tensors must be 'pt', 'tf', or 'np', got {return_tensors}")
            
        return self.tokenizer(
            sentences,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token ids back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self):
        """Get the vocabulary size for model initialization"""
        return self._vocab_size
