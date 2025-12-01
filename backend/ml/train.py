"""
Training script for AQI prediction model
This is optional - models are trained on-the-fly in the API
"""
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ml.model import AQIPredictor
from aqi_service import AQIService


def train_model_for_city(city: str, save_path: str = None):
    """
    Train a model for a specific city and save it
    
    Args:
        city: City name
        save_path: Path to save the model
    """
    # Fetch history
    history_data = AQIService.get_history_aqi(city, days=7)
    
    if not history_data or not history_data.get("history"):
        print(f"Could not fetch history for {city}")
        return False
    
    history = history_data["history"]
    
    # Create and train model
    predictor = AQIPredictor(model_type="linear")
    
    if predictor.train(history):
        # Default save path if not provided
        if save_path is None:
            save_path = os.path.join(os.path.dirname(__file__), "aqi_model.pkl")
        predictor.save(save_path)
        print(f"Model trained and saved for {city}")
        return True
    else:
        print(f"Failed to train model for {city}")
        return False


if __name__ == "__main__":
    # Example: Train model for a default city
    train_model_for_city("Delhi")

