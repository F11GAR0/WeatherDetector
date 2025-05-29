# 🌤️ Weather Detector

A powerful PyTorch-based weather detection model that predicts precipitation types with 98.6% accuracy! 🎯

## 📋 Features

- 🔍 Accurate weather prediction (98.6% accuracy)
- 🎯 Three-class classification (No precipitation, Rain, Snow)
- 📊 Comprehensive weather feature analysis
- 🐳 Docker support for easy deployment
- 🧪 Automated testing pipeline
- 📈 Training visualization and metrics
- 📥 Automated dataset download

## 🏗️ Project Structure
```
weather_detector/
├── wd/                    # Main package directory
│   ├── src/              # Source code
│   │   ├── config.py     # Configuration parameters
│   │   ├── dataset.py    # Dataset loading and processing
│   │   └── detector.py   # Model and training implementation
│   ├── tests/            # Test suite
│   │   └── test_weather_detector.py
│   ├── wd.py            # Main entry point
│   ├── predict.py       # Prediction script
│   └── requirements.txt # Project dependencies
├── automation/           # Automation scripts
│   ├── run_tests.sh     # Automated testing script
│   └── download_dataset.sh # Dataset download automation
├── data/                # Data directory
├── Dockerfile          # Container configuration
└── download_dataset.py # Dataset download script
```

## 🚀 Setup

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

### 🐳 Docker
Build and run using Docker:
```bash
docker build -t weather_detector .
docker run --rm weather_detector
```

### 📥 Dataset Download
You can download the dataset using the automated script:
```bash
./automation/download_dataset.sh
```

The script will:
- Check for kaggle.json configuration
- Create data directory if needed
- Build Docker image if not exists
- Download dataset using Docker container into ./data directory
- Provide clear feedback about the process

## 📊 Dataset
The project uses the weather dataset from Kaggle (muthuj7/weather-dataset). The dataset is automatically downloaded when running the application.

## 🧠 Model Architecture
- Input: Weather features (temperature, humidity, wind speed, etc.)
- Architecture: Feed-forward neural network
- Output: Three-class classification (No precipitation, Rain, Snow)
- Features:
  - Temperature
  - Apparent temperature
  - Humidity
  - Wind speed
  - Wind bearing
  - Visibility
  - Pressure
  - Hour
  - Month

## 🎯 Training
The model is trained with:
- Adam optimizer
- Cross Entropy Loss
- Accuracy metric
- 80/20 train/test split
- 20 epochs with early stopping
- Learning rate: 0.001

## 🧪 Testing
Run the automated test suite:
```bash
./automation/run_tests.sh
```

The test suite verifies:
- Model prediction functionality
- Output format validation
- Probability distribution checks
- Invalid input handling

## 📈 Performance
- Training Accuracy: 98.6%
- Test Accuracy: 98.6%
- Loss: 0.0347

## 🤝 Contributing
Feel free to submit issues and enhancement requests!

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details. 