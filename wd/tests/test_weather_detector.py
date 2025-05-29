import pytest
import torch
from src.detector import WeatherDetectorTrainer

@pytest.fixture
def trainer():
    """Fixture to create and load a trained model."""
    trainer = WeatherDetectorTrainer()
    trainer.load_model()
    return trainer

def test_model_prediction(trainer):
    """
    Test the core functionality of the weather prediction model:
    1. Model can make predictions
    2. Predictions return valid classes
    3. Probability distribution is valid
    4. Model handles input data correctly
    """
    # Test case 1: Normal weather conditions
    weather_data = {
        'temperature': 15.0,
        'apparent_temperature': 14.0,
        'humidity': 0.75,
        'wind_speed': 10.0,
        'wind_bearing': 180.0,
        'visibility': 10.0,
        'pressure': 1013.0,
        'hour': 14,
        'month': 6
    }
    
    # Test prediction functionality
    predicted_class, probabilities = trainer.predict_from_dict(weather_data)
    
    # Verify prediction output format
    assert isinstance(predicted_class, int)
    assert predicted_class in [0, 1, 2]  # Valid class range
    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == (3,)  # Three classes
    
    # Verify probability distribution
    assert abs(torch.sum(probabilities) - 1.0) < 1e-6  # Sums to 1
    assert all(0 <= p <= 1 for p in probabilities)  # All probabilities between 0 and 1
    
    # Test case 2: Invalid input handling
    with pytest.raises(KeyError):
        invalid_data = {
            'temperature': 15.0,
            'humidity': 0.75
        }
        trainer.predict_from_dict(invalid_data) 