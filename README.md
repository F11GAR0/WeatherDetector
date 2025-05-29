# Weather Detector

A PyTorch-based weather detection model that classifies weather conditions from images.

## Project Structure
```
weather_detector/
├── src/
│   ├── config.py      # Configuration parameters
│   ├── dataset.py     # Dataset loading and processing
│   └── detector.py    # Model and training implementation
├── wd.py             # Main entry point
├── requirements.txt   # Project dependencies
└── Dockerfile        # Container configuration
```

## Setup

### Local Development
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add kaggle.json config to root directory.

### Docker
Build and run using Docker:
```bash
docker build -t weather_detector .
docker run --rm weather_detector
```

## Dataset
The project uses the weather dataset from Kaggle (muthuj7/weather-dataset). The dataset is automatically downloaded when running the application.

## Model Architecture
- Input: RGB images (28x28 pixels)
- Architecture: Simple feed-forward neural network
- Output: Binary classification

## Training
The model is trained with:
- Adam optimizer
- BCE loss
- Binary accuracy metric
- 80/20 train/test split 