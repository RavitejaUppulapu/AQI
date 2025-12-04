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
            model_type: "linear", "random_forest" or "auto"
        """
        # model_type describes the *strategy* to use.
        # When set to "auto", the predictor will train multiple models
        # (LinearRegression and RandomForest) and automatically select
        # the one with the best metrics (primarily highest R², then lowest MAE).
        self.model_type = model_type

        # Keep separate instances for linear and random forest models
        self.linear_model = LinearRegression()
        self.random_forest_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )

        # This will point to the *selected* model after training
        self.model = self.linear_model
        self.selected_model_type = model_type
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

            def _fit_and_score(model, name: str):
                """Train a single model and compute metrics."""
                model.fit(X, y)
                y_pred = model.predict(X)
                metrics = {
                    "r2_score": float(r2_score(y, y_pred)),
                    "mae": float(mean_absolute_error(y, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                    "training_samples": len(y),
                    "model_name": name,
                }
                return metrics

            if self.model_type in ("linear", "random_forest"):
                # Single-model path (backwards compatible)
                if self.model_type == "linear":
                    self.model = self.linear_model
                else:
                    self.model = self.random_forest_model

                metrics = _fit_and_score(self.model, self.model_type)
                self.training_metrics = metrics
                self.selected_model_type = self.model_type
                self.is_trained = True
                logger.info(
                    "Model (%s) trained successfully. R²=%.3f",
                    self.selected_model_type,
                    self.training_metrics["r2_score"],
                )
                return True

            # Auto model selection: train LinearRegression and RandomForest,
            # then pick the best one based on R² (and MAE as a tiebreaker).
            linear_metrics = _fit_and_score(self.linear_model, "linear")
            rf_metrics = _fit_and_score(self.random_forest_model, "random_forest")

            # Choose best model: higher R² is better; if close, prefer lower MAE
            candidates = [linear_metrics, rf_metrics]
            candidates_sorted = sorted(
                candidates,
                key=lambda m: (m["r2_score"], -m["mae"]),  # higher r2, lower mae
                reverse=True,
            )
            best = candidates_sorted[0]

            if best["model_name"] == "linear":
                self.model = self.linear_model
            else:
                self.model = self.random_forest_model

            self.training_metrics = best
            self.selected_model_type = best["model_name"]
            self.is_trained = True
            logger.info(
                "Auto-selected '%s' model. R²=%.3f, MAE=%.3f",
                self.selected_model_type,
                self.training_metrics["r2_score"],
                self.training_metrics["mae"],
            )
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
                "metrics": self.training_metrics,
                "model_used": self.selected_model_type,
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
                "model_type": self.selected_model_type,
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


def backtest_history(history: List[Dict], model_type: str = "auto") -> Optional[Dict]:
    """
    Simple rolling-origin backtest over the provided history.

    For each step i, train on history[:i] and predict AQI for day i,
    then compare with the actual AQI at history[i].

    Args:
        history: List of dicts with at least 'aqi' and 'day'
        model_type: 'linear', 'random_forest', or 'auto'

    Returns:
        Dictionary with aggregated backtest metrics or None if not enough data.
    """
    try:
        if not history or len(history) < 5:
            logger.warning("Not enough history for backtesting (need at least 5 days)")
            return None

        predictions = []
        actuals = []

        # Start from index 3 so we have at least 3 days to train on
        for i in range(3, len(history)):
            train_hist = history[:i]  # use first i days to predict day i
            true_val = history[i].get("aqi", 0)

            predictor = AQIPredictor(model_type=model_type)
            if not predictor.train(train_hist):
                continue

            res = predictor.predict(train_hist)
            if res is None:
                continue

            predictions.append(res["prediction"])
            actuals.append(true_val)

        if not predictions or not actuals:
            logger.warning("Backtest did not produce any valid predictions")
            return None

        y_true = np.array(actuals)
        y_pred = np.array(predictions)

        metrics = {
            "r2_score": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "samples": int(len(y_true)),
        }

        # Basic trend analysis based on backtest
        if len(actuals) >= 2:
            trend_change = (actuals[-1] - actuals[0]) / max(actuals[0], 1) * 100.0
        else:
            trend_change = 0.0

        trend = "Stable"
        if trend_change > 5:
            trend = "Rising"
        elif trend_change < -5:
            trend = "Falling"

        volatility = float(np.var(y_true))

        return {
            "metrics": metrics,
            "trend": trend,
            "trend_change_percent": float(trend_change),
            "volatility": volatility,
        }
    except Exception as e:
        logger.error(f"Error during backtesting: {e}", exc_info=True)
        return None
