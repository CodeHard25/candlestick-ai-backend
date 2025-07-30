# PowerShell Script to Push Clean Repository to GitHub
# Run this AFTER creating your GitHub repository

Write-Host "🚀 Pushing Candlestick AI Backend to GitHub..." -ForegroundColor Green

# Prompt for GitHub username
$username = Read-Host "Enter your GitHub username"

# Construct repository URL
$repoUrl = "https://github.com/$username/candlestick-ai-backend.git"

Write-Host "📡 Repository URL: $repoUrl" -ForegroundColor Yellow

# Add remote origin
Write-Host "🔗 Adding remote origin..." -ForegroundColor Yellow
git remote add origin $repoUrl

# Push to GitHub
Write-Host "📤 Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin master

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ SUCCESS! Repository pushed to GitHub!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Your repository is now available at:" -ForegroundColor Cyan
    Write-Host "   $repoUrl" -ForegroundColor White
    Write-Host ""
    Write-Host "🚀 Next: Deploy on Render.com" -ForegroundColor Cyan
    Write-Host "   1. Go to https://render.com" -ForegroundColor White
    Write-Host "   2. Click 'New +' → 'Web Service'" -ForegroundColor White
    Write-Host "   3. Connect your GitHub repository" -ForegroundColor White
    Write-Host "   4. Use these settings:" -ForegroundColor White
    Write-Host "      - Build Command: pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host "      - Start Command: uvicorn app:app --host 0.0.0.0 --port `$PORT" -ForegroundColor Gray
    Write-Host "      - Health Check Path: /health" -ForegroundColor Gray
} else {
    Write-Host "❌ Error pushing to GitHub. Please check:" -ForegroundColor Red
    Write-Host "   1. Repository exists on GitHub" -ForegroundColor White
    Write-Host "   2. You have push permissions" -ForegroundColor White
    Write-Host "   3. GitHub username is correct" -ForegroundColor White
}
