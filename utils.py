import torch
import numpy as np
from PIL import Image
import logging
import os
from datetime import datetime
from typing import Optional, Tuple, List
import json

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def validate_image(image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
    """Validate uploaded image"""
    if len(image_bytes) > max_size:
        raise ValueError(f"File size exceeds {max_size} bytes")
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Check if it's a valid image
        image.verify()
        return True
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_stats(stats: dict, filepath: str):
    """Save training statistics to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Training stats saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save training stats: {e}")

def load_training_stats(filepath: str) -> Optional[dict]:
    """Load training statistics from JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load training stats: {e}")
    return None

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    cm = confusion_matrix(targets, predictions)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),  
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def create_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                          epoch: int, loss: float, accuracy: float, 
                          additional_info: dict = None) -> dict:
    """Create a comprehensive model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'model_params': count_parameters(model)
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    return checkpoint

def load_model_checkpoint(filepath: str, model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer = None, 
                         device: torch.device = None) -> dict:
    """Load model checkpoint"""
    if device is None:
        device = get_device()
        
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint

def ensure_directory_exists(directory: str):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(directory, exist_ok=True)

def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0

def cleanup_old_files(directory: str, max_files: int = 10, pattern: str = "*.pth"):
    """Keep only the most recent files in a directory"""
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) > max_files:
        # Sort by modification time
        files.sort(key=os.path.getmtime)
        # Remove oldest files
        for file in files[:-max_files]:
            try:
                os.remove(file)
                logger.info(f"Removed old file: {file}")
            except Exception as e:
                logger.error(f"Failed to remove {file}: {e}")

class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class ProgressTracker:
    """Track training progress"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_times = []
        
    def update(self, train_loss: float, val_loss: float, val_accuracy: float, epoch_time: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        self.epoch_times.append(epoch_time)
    
    def get_summary(self) -> dict:
        return {
            'total_epochs': len(self.train_losses),
            'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'total_training_time': sum(self.epoch_times)
        }

def print_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...] = (3, 224, 224)):
    """Print model summary similar to Keras"""
    device = next(model.parameters()).device
    
    print("="*80)
    print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += params
            trainable_params += trainable
            
            # Get module type
            module_type = module.__class__.__name__
            
            print(f"{name:<25} {module_type:<20} {params:<15,}")
    
    print("="*80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("="*80)