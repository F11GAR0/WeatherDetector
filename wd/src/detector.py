import torch
from torchvision import transforms
from torcheval.metrics import MulticlassAccuracy
from torchviz import make_dot
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from modelsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from .config import *
from .dataset import WeatherCSVDataset

class WeatherDetectorModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class WeatherDetectorTrainer:
    def __init__(self):
        self.device = DEVICE
        self.input_size = 9  # Number of features in our CSV
        self.hidden_size = 64
        self.num_classes = 3  # Number of precipitation types (rain, snow, none)
        
        self.model = WeatherDetectorModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes
        ).to(self.device)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = MulticlassAccuracy(num_classes=self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Create model directory if it doesn't exist
        self.model_dir = os.path.join(DATASET_STORAGE, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize data loaders
        self._setup_data_loaders()
        
    def _setup_data_loaders(self):
        # Create dataset
        dataset = WeatherCSVDataset(csv_path=os.path.join(DATASET_STORAGE, 'weatherHistory.csv'))
        
        # Split dataset into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    def train_epoch(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                self.metric.update(pred, y)
                
        test_loss /= num_batches
        metric_value = self.metric.compute()
        
        print(f"Test Error: \n Accuracy: {(100*metric_value):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, metric_value

    def train(self):
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.train_epoch()
            cur_loss, cur_metric = self.test()
            history['loss'].append(cur_loss)
            history['accuracy'].append(cur_metric)
            
        return history

    def plot_training_history(self, history):
        acc = history['accuracy']
        loss = history['loss']
        epochs = range(1, len(acc) + 1)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='red')
        ax1.plot(epochs, loss, color='red', label='Training loss')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='blue')
        ax2.plot(epochs, acc, color='blue', label='Training acc')

        plt.title('Training History')
        plt.legend()
        
        # Save plot as PNG
        output_dir = os.path.join(DATASET_STORAGE, 'output')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        
        plt.show()

    def visualize_model(self, features):
        # Create output directory if it doesn't exist
        output_dir = os.path.join(DATASET_STORAGE, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the graph
        dot = make_dot(self.model(features), params=dict(self.model.named_parameters()))
        
        # Save the graph to a file in the output directory
        output_path = os.path.join(output_dir, "model_architecture")
        dot.render(output_path, format="png", cleanup=True)
        logging.info(f"Model architecture saved to {output_path}.png")

    def predict(self, features):
        """
        Make a weather prediction using the trained model.
        
        Args:
            features (torch.Tensor): A tensor of shape (1, 9) containing:
                - Temperature (C)
                - Apparent Temperature (C)
                - Humidity
                - Wind Speed (km/h)
                - Wind Bearing (degrees)
                - Visibility (km)
                - Pressure (millibars)
                - Hour
                - Month
                
        Returns:
            tuple: (predicted_class, probabilities)
                - predicted_class (int): 0 for no precipitation, 1 for rain, 2 for snow
                - probabilities (torch.Tensor): Probability distribution over classes
        """
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
        return predicted_class.item(), probabilities[0]

    def predict_from_dict(self, weather_data):
        """
        Make a weather prediction from a dictionary of weather features.
        
        Args:
            weather_data (dict): Dictionary containing weather features with keys:
                'temperature', 'apparent_temperature', 'humidity', 'wind_speed',
                'wind_bearing', 'visibility', 'pressure', 'hour', 'month'
                
        Returns:
            tuple: (predicted_class, probabilities)
        """
        # Convert dictionary to tensor
        features = torch.tensor([
            weather_data['temperature'],
            weather_data['apparent_temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['wind_bearing'],
            weather_data['visibility'],
            weather_data['pressure'],
            weather_data['hour'],
            weather_data['month']
        ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        return self.predict(features)

    def save_model(self, filename='weather_detector_model.pth'):
        """
        Save the trained model to a file.
        
        Args:
            filename (str): Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes
        }, model_path)
        logging.info(f"Model saved to {model_path}")

    def load_model(self, filename='weather_detector_model.pth'):
        """
        Load a trained model from a file.
        
        Args:
            filename (str): Name of the file to load the model from
        """
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            logging.warning(f"No saved model found at {model_path}")
            return False
            
        checkpoint = torch.load(model_path)
        self.model = WeatherDetectorModel(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")
        return True
