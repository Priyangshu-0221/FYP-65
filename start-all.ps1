# Full Stack Startup Script for FYP-65

Write-Host "================================" -ForegroundColor Cyan
Write-Host "AI Career Guidance System" -ForegroundColor Cyan
Write-Host "Full Stack Startup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install Backend Dependencies
Write-Host "[1/3] Installing backend dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Backend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install backend dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Starting Servers..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 2: Start Backend Server
Write-Host "[2/3] Starting Backend API Server..." -ForegroundColor Yellow
Write-Host "Backend will run at: http://localhost:8000" -ForegroundColor Green
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Gray
Write-Host ""

Start-Process powershell -ArgumentList "-Command", "cd '$PWD' ; python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload" -NoNewWindow

# Wait for backend to start
Start-Sleep -Seconds 3

# Step 3: Start Frontend Server
Write-Host "[3/3] Starting Frontend Dev Server..." -ForegroundColor Yellow
Write-Host "Frontend will run at: http://localhost:3000" -ForegroundColor Green
Write-Host ""

cd frontend-next
npm run dev

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "All servers running!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✓ Backend API:  http://localhost:8000" -ForegroundColor Green
Write-Host "✓ Frontend:     http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "Open the frontend in your browser to start testing!" -ForegroundColor Cyan
