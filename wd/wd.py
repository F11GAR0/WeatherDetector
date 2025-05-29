#!/usr/bin/python

import logging
import sys
from src.detector import WeatherDetectorTrainer
import torch
import matplotlib.pyplot as plt

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
        logging.info("Initializing Weather Detector Trainer")
        trainer = WeatherDetectorTrainer()
        
        logging.info("Starting model training")
        history = trainer.train()
        
        logging.info("Plotting training history")
        trainer.plot_training_history(history)
        
        # Final evaluation
        logging.info("Performing final evaluation")
        trainer.model.eval()
        test_features, test_labels = next(iter(trainer.test_dataloader))
        out = trainer.model(test_features)
        loss = trainer.loss_fn(out.squeeze(), test_labels)
        trainer.metric.update(out.squeeze(), test_labels)
        accuracy = trainer.metric.compute()
        
        logging.info(f"Final Evaluation Results:")
        logging.info(f"Loss: {float(loss):.4f}")
        logging.info(f"Accuracy: {float(accuracy):.4f}")
        
        # Save the trained model
        logging.info("Saving trained model")
        trainer.save_model()
        
        # Visualize sample image
        logging.info("Visualizing sample features")
        train_features, train_labels = next(iter(trainer.train_dataloader))
        features = train_features[0].squeeze()
        label = train_labels[0]
        
        # Create feature names for the plot
        feature_names = [
            'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
            'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
            'Pressure (millibars)', 'Hour', 'Month'
        ]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, features.cpu().numpy())
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Sample Weather Features (Label: {label})')
        plt.tight_layout()
        plt.show()
        logging.info(f"Sample features label: {label}")
        
        # Visualize model architecture
        logging.info("Visualizing model architecture")
        trainer.visualize_model(test_features)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 