# Deployment Instructions for Render.com

## Prerequisites
1. Create a GitHub repository for your backend code
2. Push the backend folder contents to the repository
3. Create a Render.com account

## Deployment Steps

### 1. Prepare Repository
```bash
# Initialize git repository in backend folder
cd backend
git init
git add .
git commit -m "Initial commit - Candlestick AI Backend"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/candlestick-ai-backend.git
git push -u origin main
```

### 2. Deploy on Render.com
1. Go to https://render.com and sign in
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: candlestick-ai-backend
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or paid for better performance)

### 3. Environment Variables
Set these environment variables in Render dashboard:
- `PYTHON_VERSION`: 3.11.0
- `PORT`: 10000 (automatically set by Render)

### 4. Health Check
- Set health check path to: `/health`

### 5. Model Upload (Important!)
Since model files are too large for git, you'll need to:
1. Train the model locally first
2. Upload the model file (`best_enhanced_model.pth`) manually to your deployment
3. Or implement a model download mechanism

## Frontend Configuration
Update your frontend's API URL to point to your Render deployment:
```javascript
const API_BASE_URL = 'https://your-app-name.onrender.com';
```

## Testing
After deployment, test these endpoints:
- `GET /health` - Should return healthy status
- `GET /model-info` - Should return model information
- `POST /predict` - Should accept image uploads

## Notes
- Free tier on Render may have cold starts
- Consider upgrading to paid tier for production use
- Monitor logs in Render dashboard for debugging
