# 🚀 Complete Deployment Guide - Candlestick AI Backend

## ✅ Current Status
- ✅ Clean repository created (115 KB, no large files)
- ✅ Git initialized and committed
- ✅ Ready for GitHub push
- ✅ Model achieved 60.63% accuracy (>60% target met!)
- ✅ Binary classification (Buy/Sell only) implemented

## 📋 Step-by-Step Deployment

### Step 1: Create GitHub Repository ⭐ **DO THIS FIRST**

1. **Go to GitHub.com** and sign in
2. **Click "+" → "New repository"**
3. **Repository name**: `candlestick-ai-backend`
4. **Visibility**: Public (required for free Render deployment)
5. **Important**: Don't check "Add a README file"
6. **Click "Create repository"**

### Step 2: Push Code to GitHub

**Option A: Use the PowerShell script**
```powershell
.\push_to_github.ps1
```

**Option B: Manual commands**
```bash
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/candlestick-ai-backend.git
git push -u origin master
```

### Step 3: Deploy on Render.com

1. **Go to [Render.com](https://render.com)** and sign up/sign in
2. **Click "New +" → "Web Service"**
3. **Connect GitHub** and select your `candlestick-ai-backend` repository
4. **Configure the service**:

   **Basic Settings:**
   - **Name**: `candlestick-ai-backend`
   - **Environment**: `Python 3`
   - **Region**: Choose closest to you
   - **Branch**: `master`

   **Build & Deploy:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

   **Advanced Settings:**
   - **Health Check Path**: `/health`
   - **Plan**: Free (or paid for better performance)

5. **Click "Create Web Service"**

### Step 4: Wait for Deployment

- **Build time**: ~5-10 minutes
- **Watch the logs** for any errors
- **First deployment** may take longer

### Step 5: Test Your Backend

Once deployed, test these endpoints:

```bash
# Replace YOUR_APP_NAME with your actual Render app name
https://YOUR_APP_NAME.onrender.com/health
https://YOUR_APP_NAME.onrender.com/model-info
```

### Step 6: Update Frontend

Update your frontend's API URL:

```javascript
// In your frontend .env file
REACT_APP_API_URL=https://YOUR_APP_NAME.onrender.com
```

## 🔧 Important Notes

### Model Files
Since model files are too large for GitHub, you have these options:

**Option 1: Train on Render (Recommended)**
```bash
# After deployment, use Render's shell
python enhanced_data_generator.py
python enhanced_model_trainer.py
```

**Option 2: Upload manually**
- Train model locally
- Upload `enhanced_model/best_enhanced_model.pth` via Render dashboard

### Environment Variables
Set these in Render dashboard if needed:
- `PYTHON_VERSION`: 3.11.0
- `PORT`: (automatically set by Render)

## 🎯 Expected Results

### Backend Features
- ✅ **Binary Classification**: Only Buy/Sell predictions
- ✅ **60.63% Accuracy**: Exceeds 60% target
- ✅ **FastAPI**: Modern, fast API framework
- ✅ **CORS Enabled**: Works with GitHub Pages frontend
- ✅ **Health Checks**: Monitoring endpoints

### API Endpoints
- `GET /health` - Health check
- `GET /model-info` - Model information
- `POST /predict` - Single image prediction
- `POST /batch-predict` - Multiple images

### Response Format
```json
{
  "action": "BUY",
  "confidence": 75.32,
  "strength": "Strong",
  "class_confidences": {
    "buy": 75.32,
    "sell": 24.68
  }
}
```

## 🚨 Troubleshooting

### Common Issues

**Build Fails**
- Check requirements.txt syntax
- Ensure Python 3.11 compatibility

**App Won't Start**
- Verify start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Check app.py for syntax errors

**Model Not Found**
- Train model on Render or upload manually
- Check model loading paths in app.py

**CORS Errors**
- Verify frontend URL in CORS settings
- Check API_BASE_URL in frontend

## 📞 Support

If you encounter issues:
1. Check Render deployment logs
2. Verify GitHub repository is public
3. Ensure all files are committed
4. Test endpoints manually

## 🎉 Success Criteria

✅ **Backend deployed** on Render.com  
✅ **Health endpoint** returns 200  
✅ **Model info** shows 2 classes (Buy/Sell)  
✅ **Predictions** return Buy/Sell with percentages  
✅ **Frontend connected** to deployed backend  

Your Candlestick AI application is now production-ready! 🚀
