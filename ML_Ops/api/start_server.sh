#!/bin/bash

# Crop Disease Classification API Server Startup Script

echo "🚀 Starting Crop Disease Classification API Server..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r ../requirements.txt

# Check if model exists
if [ ! -f "../models/resnet50_best.pth" ]; then
    echo "⚠️  Warning: Model file not found at ../models/resnet50_best.pth"
    echo "   Please ensure you have trained a model first using trainer_notebook.py"
    echo "   The server will start but predictions may not be accurate without a trained model."
fi

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🔍 Interactive API testing at: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
