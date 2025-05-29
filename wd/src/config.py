import torch

# Data paths
DATASET_STORAGE = '/data'

# Model parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
