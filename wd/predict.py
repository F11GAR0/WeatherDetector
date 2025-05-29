#!/usr/bin/python

import logging
import sys
from src.detector import WeatherDetectorTrainer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        # Initialize the trainer
        logging.info("Initializing Weather Detector")
        trainer = WeatherDetectorTrainer()
        
        # Load the trained model
        logging.info("Loading trained model")
        if not trainer.load_model():
            logging.error("Failed to load model. Please train the model first using wd.py")
            sys.exit(1)
        
        # Example weather data
        weather_data = {
            'temperature': 15.0,        # Temperature in Celsius
            'apparent_temperature': 14.0,  # Apparent temperature in Celsius
            'humidity': 0.75,           # Humidity (0-1)
            'wind_speed': 10.0,         # Wind speed in km/h
            'wind_bearing': 180.0,      # Wind bearing in degrees
            'visibility': 10.0,         # Visibility in km
            'pressure': 1013.0,         # Pressure in millibars
            'hour': 14,                 # Hour of day (0-23)
            'month': 6                  # Month (1-12)
        }
        
        # Make prediction
        predicted_class, probabilities = trainer.predict_from_dict(weather_data)
        
        # Map class numbers to precipitation types
        precipitation_types = {
            0: "No precipitation",
            1: "Rain",
            2: "Snow"
        }
        
        # Print results
        logging.info("\nWeather Prediction Results:")
        logging.info(f"Predicted precipitation type: {precipitation_types[predicted_class]}")
        logging.info("\nProbability distribution:")
        for i, prob in enumerate(probabilities):
            logging.info(f"{precipitation_types[i]}: {prob.item():.2%}")
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 