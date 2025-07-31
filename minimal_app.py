"""
Ultra-minimal FastAPI app for testing deployment
"""

from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Candlestick AI Backend is running!", "status": "success"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "candlestick-ai-backend"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
