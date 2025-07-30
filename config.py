import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = BASE_DIR / "model"

# Data paths
CHARTS_DIR = DATA_DIR / "generated_charts"
LABELS_FILE = DATA_DIR / "labels.csv"
TRAIN_LABELS_FILE = DATA_DIR / "train_labels.csv"
VAL_LABELS_FILE = DATA_DIR / "val_labels.csv"

# Model paths
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
FINAL_MODEL_PATH = MODEL_DIR / "final_model.pth"

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "test_size": 0.2,
    "random_seed": 42
}

# Model configuration
MODEL_CONFIG = {
    "num_classes": 3,
    "input_size": (224, 224),
    "pretrained": True,
    "dropout_rates": [0.5, 0.3, 0.2],
    "hidden_sizes": [512, 256]
}

# Data generation configuration
DATA_CONFIG = {
    "samples_per_stock": 30,
    "window_size": 50,
    "future_days": 5,
    "price_threshold": 0.02,
    "period": "2y"
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".png", ".jpg", ".jpeg"],
    "cors_origins": ["*"]  # Change this in production
}

# Stock symbols for data generation
STOCK_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'UBER', 'SNAP', 'ZOOM', 'PYPL', 'SQ', 'ROKU', 'PINS', 'TWTR',
    'CRM', 'ADBE', 'INTC', 'AMD', 'ORCL', 'IBM', 'HPQ', 'DELL',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'COF',
    'DIS', 'BABA', 'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'
]

# Class labels
CLASS_LABELS = {
    0: "SELL",
    1: "BUY", 
    2: "HOLD"
}

# Reverse mapping
LABEL_CLASSES = {v: k for k, v in CLASS_LABELS.items()}

# Image preprocessing parameters
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [DATA_DIR, MODEL_DIR, CHARTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
if __name__ == "__main__":
    create_directories()
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")