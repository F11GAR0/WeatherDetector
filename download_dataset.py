import os
import kaggle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset():
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Download the dataset
        logger.info("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            "muthuj7/weather-dataset",
            path="data",
            unzip=True
        )
        logger.info("Dataset downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_dataset() 