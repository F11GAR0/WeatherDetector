#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[INFO] Starting dataset download...${NC}"

# Check if kaggle.json exists
if [ ! -f "kaggle.json" ]; then
    echo -e "${YELLOW}[ERROR] kaggle.json not found!${NC}"
    echo "Please place your kaggle.json file in the project root directory."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Build Docker image if it doesn't exist
if ! docker image inspect weather-detector >/dev/null 2>&1; then
    echo -e "${YELLOW}[INFO] Building Docker image...${NC}"
    docker build -t weather-detector .
fi

# Run the download script in Docker
echo -e "${YELLOW}[INFO] Running dataset download in Docker...${NC}"
docker run --rm \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/kaggle.json:/root/.kaggle/kaggle.json" \
    weather-detector python download_dataset.py

# Check if download was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Dataset downloaded successfully!${NC}"
else
    echo -e "${YELLOW}[ERROR] Dataset download failed!${NC}"
    exit 1
fi 