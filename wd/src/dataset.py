import os
import kagglehub
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class WeatherCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        self.data_dir = os.path.dirname(csv_path)
        
        try:
            # Read CSV file
            self.df = pd.read_csv(csv_path)
            
            # Preprocess data
            self._preprocess_data()
            
            logger.info(f"CSV Dataset loaded successfully with {len(self.df)} samples")
            
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {str(e)}")
            raise RuntimeError(f"Failed to load CSV dataset: {str(e)}")
    
    def _preprocess_data(self):
        # Convert date to datetime and extract relevant features
        self.df['Formatted Date'] = pd.to_datetime(self.df['Formatted Date'], utc=True, errors='coerce')
        # Drop rows where date could not be parsed
        self.df = self.df.dropna(subset=['Formatted Date'])
        self.df['Hour'] = self.df['Formatted Date'].dt.hour
        self.df['Month'] = self.df['Formatted Date'].dt.month
        
        # Select features for training
        feature_columns = [
            'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
            'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
            'Pressure (millibars)', 'Hour', 'Month'
        ]
        
        # Handle categorical variables
        self.df['Precip Type'] = self.df['Precip Type'].fillna('none')
        precip_encoder = LabelEncoder()
        self.df['Precip Type'] = precip_encoder.fit_transform(self.df['Precip Type'])
        
        # Prepare features and target
        self.features = self.df[feature_columns].values
        self.target = self.df['Precip Type'].values
        
        # Scale features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.target = torch.LongTensor(self.target)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, target

# Keep the original image dataset class for reference
class WeatherDataset:
    def __init__(self, transform=None):
        self.data_dir = "/data/weather-dataset"
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            if not os.path.exists(self.data_dir):
                logger.info("Dataset not found locally. Please download it manually from Kaggle and place it in the /data directory.")
                raise RuntimeError("Dataset not found. Please download it manually from Kaggle.")
            
            self.dataset = ImageFolder(root=self.data_dir, transform=self.transform)
            logger.info(f"Dataset loaded successfully with {len(self.dataset)} images")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]