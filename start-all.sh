#!/bin/bash

echo "================================"
echo "AI Career Guidance System"
echo "Full Stack Startup"
echo "================================"
echo ""

# Step 1: Install Backend Dependencies
echo "[1/3] Installing backend dependencies..."
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo "✓ Backend dependencies installed"
else
    echo "✗ Failed to install backend dependencies"
    exit 1
fi

echo ""
echo "================================"
echo "Starting Servers..."
echo "================================"
echo ""

# Step 2: Start Backend Server
echo "[2/3] Starting Backend API Server..."
echo "Backend will run at: http://localhost:8080"
echo "Starting in background..."
echo ""

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Step 3: Start Frontend Server
echo "[3/3] Starting Frontend Dev Server..."
echo "Frontend will run at: http://localhost:3000"
echo ""

cd frontend-next
npm run dev
