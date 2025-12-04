"""
Configuration settings for the application
"""
import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # API Configuration
    AQICN_API_TOKEN: str = os.getenv(
        "AQICN_API_TOKEN",
        "70fa7c8239bbf05b4090b6992f04d1047fc96267"
    )
    AQICN_BASE_URL: str = "https://api.waqi.info/feed/{city}/?token={token}"
    
    # Cache Configuration
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    PREDICTION_CACHE_TTL: int = int(os.getenv("PREDICTION_CACHE_TTL", "60"))  # 1 minute
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # API Configuration
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "10"))
    MAX_HISTORY_DAYS: int = int(os.getenv("MAX_HISTORY_DAYS", "30"))
    MIN_HISTORY_DAYS_FOR_PREDICTION: int = int(os.getenv("MIN_HISTORY_DAYS_FOR_PREDICTION", "3"))
    
    # CORS Configuration
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS",
        "*"
    ).split(",")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ML Configuration
    # linear | random_forest | auto (auto-select best model)
    ML_MODEL_TYPE: str = os.getenv("ML_MODEL_TYPE", "auto")
    ML_MODEL_SAVE_PATH: str = os.getenv("ML_MODEL_SAVE_PATH", "backend/ml/models/")

settings = Settings()



