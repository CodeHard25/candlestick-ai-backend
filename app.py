from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import io
import numpy as np
import logging
from datetime import datetime
import os
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Candlestick Chart Prediction API",
    description="AI-powered trading signal prediction from candlestick chart images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Local backend
        "https://*.github.io",    # GitHub Pages
        "https://*.githubpages.com",  # GitHub Pages alternative
        "*"  # Allow all for now - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
transform = None

class CandlestickCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CandlestickCNN, self).__init__()

        # Use ResNet50 as backbone
        self.backbone = resnet50(pretrained=pretrained)

        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the trained model"""
    global model, device, transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = CandlestickCNN(num_classes=2, pretrained=False)
    
    # Load model weights - try enhanced model first, then fallback to basic model
    enhanced_model_path = "enhanced_model/best_enhanced_model.pth"
    basic_model_path = "model/best_model.pth"

    model_loaded = False
    try:
        if os.path.exists(enhanced_model_path):
            checkpoint = torch.load(enhanced_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Enhanced model loaded from {enhanced_model_path}")
            model_loaded = True
        elif os.path.exists(basic_model_path):
            checkpoint = torch.load(basic_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Basic model loaded from {basic_model_path}")
            model_loaded = True
        else:
            logger.warning(f"No model file found. Using untrained model.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using untrained model due to loading error.")

    if not model_loaded:
        logger.warning("Model will need to be trained or uploaded before making predictions.")
    
    model.to(device)
    model.eval()
    
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    logger.info("Model loaded successfully")

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def predict_signal(image_tensor):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Convert to numpy
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            all_probs = probabilities.cpu().numpy()[0]
            
            return predicted_class, confidence_score, all_probs
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Load model on startup
load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Candlestick Chart Prediction API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_trading_signal(file: UploadFile = File(...)):
    """
    Predict trading signal from candlestick chart image
    
    Args:
        file: Uploaded image file (PNG, JPG, JPEG)
    
    Returns:
        JSON response with prediction results
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (max 10MB)
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
    
    try:
        # Read image data
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        predicted_class, confidence_score, all_probs = predict_signal(image_tensor)
        
        # Map prediction to trading signal
        class_mapping = {
            0: "SELL",
            1: "BUY"
        }

        action = class_mapping[predicted_class]

        # Get confidence for each class
        class_confidences = {
            "sell_confidence": float(all_probs[0]),
            "buy_confidence": float(all_probs[1])
        }
        
        # Determine recommendation strength
        if confidence_score >= 0.8:
            strength = "STRONG"
        elif confidence_score >= 0.6:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        # Generate response
        response = {
            "success": True,
            "prediction": {
                "action": action,
                "confidence": round(confidence_score * 100, 2),
                "strength": strength,
                "class_confidences": {
                    "sell": round(class_confidences["sell_confidence"] * 100, 2),
                    "buy": round(class_confidences["buy_confidence"] * 100, 2)
                }
            },
            "metadata": {
                "filename": file.filename,
                "file_size": file.size,
                "content_type": file.content_type,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Prediction made: {action} with {confidence_score:.3f} confidence")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Predict trading signals for multiple images
    
    Args:
        files: List of uploaded image files
    
    Returns:
        JSON response with batch prediction results
    """
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        try:
            # Validate file
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid file type"
                })
                continue
            
            # Read and predict
            image_bytes = await file.read()
            image_tensor = preprocess_image(image_bytes)
            predicted_class, confidence_score, _ = predict_signal(image_tensor)

            # Map prediction
            class_mapping = {0: "SELL", 1: "BUY"}
            action = class_mapping[predicted_class]
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "action": action,
                    "confidence": round(confidence_score * 100, 2)
                }
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "results": results,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Enhanced CandlestickCNN with ResNet50 backbone",
        "num_classes": 2,
        "classes": ["SELL", "BUY"],
        "input_size": "224x224x3",
        "device": str(device),
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    # Run the server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )