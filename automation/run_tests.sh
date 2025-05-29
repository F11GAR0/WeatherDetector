#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the absolute path of the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# Ensure data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found at $DATA_DIR"
    exit 1
fi

# Build the Docker image
print_status "Building Docker image..."
docker build -t weather-detector "$PROJECT_ROOT"

# Train the model
print_status "Training the model..."
docker run -v "$DATA_DIR:/data" weather-detector python wd.py

# Run the tests
print_status "Running tests..."
docker run -v "$DATA_DIR:/data" weather-detector python -m pytest tests/test_weather_detector.py -v

print_status "All tests completed successfully!" 