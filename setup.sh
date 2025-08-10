#!/bin/bash

# Setup script for Customer Segmentation Flask API

echo "Setting up Customer Segmentation Flask API..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create mock model for testing
echo "Creating mock model for testing..."
cd models
python ../create_mock_model.py
cd ..

echo "Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the Flask app: python app.py"
echo "3. Test the API: python test_api.py"
echo ""
echo "The API will be available at http://localhost:5000"
