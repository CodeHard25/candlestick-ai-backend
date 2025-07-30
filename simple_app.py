"""
Simple FastAPI app for Candlestick AI - Deployment Ready
This version starts without requiring model files
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Candlestick AI API",
    description="AI-powered trading signal prediction from candlestick charts",
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Candlestick AI Backend API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "candlestick-ai-backend",
        "version": "1.0.0"
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the model"""
    return {
        "model_type": "Enhanced CandlestickCNN with ResNet50 backbone",
        "num_classes": 2,
        "classes": ["SELL", "BUY"],
        "input_size": "224x224x3",
        "status": "Model files not loaded - training required",
        "accuracy": "60.63% (achieved in training)",
        "note": "Upload model files or train on server to enable predictions"
    }

@app.post("/predict")
async def predict_signal():
    """Predict trading signal from candlestick chart"""
    return {
        "error": "Model not loaded",
        "message": "Please train the model first using /train endpoint or upload model files",
        "status": "service_ready_model_missing"
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service": "candlestick-ai-backend",
        "status": "running",
        "model_loaded": False,
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info", 
            "predict": "/predict",
            "status": "/status"
        },
        "deployment": {
            "platform": "Render.com",
            "url": "https://candlestick-ai-backend.onrender.com"
        },
        "next_steps": [
            "Train model using enhanced_model_trainer.py",
            "Upload model files to enable predictions",
            "Test prediction endpoint with candlestick images"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "simple_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
