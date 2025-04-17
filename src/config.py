from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
FEW_SHOT_DIR = DATA_DIR / "few_shot"

# Model configurations
DEFAULT_MODEL = "gpt2"  # Default model to use
MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # Where to cache downloaded models

# Fine-tuning configurations
FINE_TUNE_CONFIG = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": PROJECT_ROOT / "logs",
    "output_dir": PROJECT_ROOT / "fine_tuned_models",
}

# Generation configurations
GENERATION_CONFIG = {
    "max_length": 512,
    "num_return_sequences": 1,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# Create necessary directories
for directory in [INPUT_DIR, OUTPUT_DIR, FEW_SHOT_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 