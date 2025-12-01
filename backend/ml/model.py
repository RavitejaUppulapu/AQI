"""
Enhanced ML Model for AQI Prediction with feature engineering and metrics
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AQIPredictor:
    """Enhanced AQI Prediction Model with feature engineering"""
    
    def __init__(self, model_type: str = "linear"):
        """
        Initialize predictor
        
        Args:
            model_type: "linear" or "random_forest"
        """
        self.model_type = model_type
        if model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
    
    def prepare_features(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert history data into features for training with feature engineering
        
        Args:
            history: List of dicts with 'day' and 'aqi' keys
            
        Returns:
            X (features): array of engineered features
            y (target): array of AQI values
        """
        if not history:
            return np.array([]).reshape(-1, 1), np.array([])
        
        aqi_values = [item.get("aqi", 0) for item in history]
        n = len(aqi_values)
        
        if n == 0:
            return np.array([]).reshape(-1, 1), np.array([])
        
        # Feature engineering
        features = []
        
        # 1. Day index (trend)
        day_indices = np.arange(1, n + 1).reshape(-1, 1)
        features.append(day_indices)
        
        # 2. Moving average (3-day)
        if n >= 3:
            moving_avg = []
            for i in range(n):
                window_start = max(0, i - 2)
                window_end = i + 1
                window_avg = np.mean(aqi_values[window_start:window_end])
                moving_avg.append(window_avg)
            features.append(np.array(moving_avg).reshape(-1, 1))
        else:
            features.append(np.array(aqi_values).reshape(-1, 1))
        
        # 3. Lag features (previous day AQI)
        lag1 = [aqi_values[0]] + aqi_values[:-1]
        features.append(np.array(lag1).reshape(-1, 1))
        
        # 4. Change from previous day
        change = [0] + [aqi_values[i] - aqi_values[i-1] for i in range(1, n)]
        features.append(np.array(change).reshape(-1, 1))
        
        # 5. Rolling variance (if enough data)
        if n >= 3:
            rolling_var = []
            for i in range(n):
                window_start = max(0, i - 2)
                window_end = i + 1
                window_var = np.var(aqi_values[window_start:window_end])
                rolling_var.append(window_var if not np.isnan(window_var) else 0)
            features.append(np.array(rolling_var).reshape(-1, 1))
        else:
            features.append(np.zeros((n, 1)))
        
        # Combine all features
        X = np.hstack(features)
        y = np.array(aqi_values)
        
        self.feature_names = ["day_index", "moving_avg_3d", "lag1", "change", "rolling_var"]
        
        return X, y
    
    def train(self, history: List[Dict]) -> bool:
        """
        Train the model on historical data
        
        Args:
            history: List of historical AQI data points
            
        Returns:
            True if training successful
        """
        try:
            if len(history) < 3:
                logger.warning("Not enough data points for training (need at least 3)")
                return False
            
            X, y = self.prepare_features(history)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("Empty feature or target arrays")
                return False
            
            # Train model
            self.model.fit(X, y)
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            self.training_metrics = {
                "r2_score": float(r2_score(y, y_pred)),
                "mae": float(mean_absolute_error(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "training_samples": len(y)
            }
            
            self.is_trained = True
            logger.info(f"Model trained successfully. R²={self.training_metrics['r2_score']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return False
    
    def predict(self, history: List[Dict]) -> Optional[Dict]:
        """
        Predict next day AQI with confidence score
        
        Args:
            history: List of historical AQI data points
            
        Returns:
            Dictionary with prediction, confidence, and metrics or None if error
        """
        try:
            if not self.is_trained:
                # Train on the fly if not already trained
                if not self.train(history):
                    return None
            
            # Prepare features for prediction
            X, _ = self.prepare_features(history)
            
            if len(X) == 0:
                return None
            
            # Create prediction features for next day
            last_features = X[-1].copy()
            n = len(history)
            
            # Update features for next day
            # Day index
            last_features[0] = n + 1
            
            # Moving average (predict using current moving avg)
            if n >= 2:
                last_features[1] = np.mean([h.get("aqi", 0) for h in history[-2:]])
            
            # Lag1 (current AQI)
            last_features[2] = history[-1].get("aqi", 0)
            
            # Change (predict small change based on recent trend)
            if n >= 2:
                recent_change = history[-1].get("aqi", 0) - history[-2].get("aqi", 0)
                last_features[3] = recent_change * 0.5  # Damped change
            else:
                last_features[3] = 0
            
            # Rolling variance (use current)
            if n >= 2:
                last_features[4] = np.var([h.get("aqi", 0) for h in history[-2:]])
            
            # Make prediction
            X_pred = last_features.reshape(1, -1)
            prediction = self.model.predict(X_pred)[0]
            
            # Calculate confidence based on model metrics
            r2 = self.training_metrics.get("r2_score", 0.5)
            mae = self.training_metrics.get("mae", 10)
            
            # Confidence: higher R² and lower MAE = higher confidence
            confidence = min(1.0, max(0.0, r2 * 0.8 + (1 - min(mae / 50, 1)) * 0.2))
            
            # Ensure prediction is non-negative
            predicted_aqi = max(0, float(prediction))
            
            return {
                "prediction": predicted_aqi,
                "confidence": confidence,
                "metrics": self.training_metrics
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            return None
    
    def save(self, filepath: str):
        """Save model to file"""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            joblib.dump({
                "model": self.model,
                "model_type": self.model_type,
                "is_trained": self.is_trained,
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "trained_at": datetime.now().isoformat()
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
    
    def load(self, filepath: str):
        """Load model from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            data = joblib.load(filepath)
            self.model = data["model"]
            self.model_type = data.get("model_type", "linear")
            self.is_trained = data.get("is_trained", False)
            self.feature_names = data.get("feature_names", [])
            self.training_metrics = data.get("training_metrics", {})
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False


def get_health_message(aqi: float) -> str:
    """
    Get health message based on AQI value
    
    Args:
        aqi: AQI value
        
    Returns:
        Health message string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
