"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class CurrentAQIResponse(BaseModel):
    """Response model for current AQI endpoint"""
    city: str
    aqi: int = Field(..., ge=0, description="Air Quality Index")
    pm25: float = Field(..., ge=0, description="PM2.5 concentration")
    pm10: float = Field(..., ge=0, description="PM10 concentration")
    o3: float = Field(..., ge=0, description="Ozone concentration")
    no2: float = Field(..., ge=0, description="Nitrogen dioxide concentration")
    co: float = Field(..., ge=0, description="Carbon monoxide concentration")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    wind: Optional[float] = Field(None, ge=0, description="Wind speed in m/s")
    time: str = Field(..., description="Timestamp of the reading")
    health_message: str = Field(..., description="Health category based on AQI")
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Delhi, India",
                "aqi": 90,
                "pm25": 55.0,
                "pm10": 72.0,
                "o3": 33.0,
                "no2": 12.0,
                "co": 1.2,
                "temperature": 29.0,
                "humidity": 58.0,
                "wind": 3.4,
                "time": "2025-01-16T14:00:00",
                "health_message": "Moderate"
            }
        }


class HistoryEntry(BaseModel):
    """Single history entry"""
    day: str = Field(..., description="Date in YYYY-MM-DD format")
    aqi: int = Field(..., ge=0, description="AQI value for the day")


class HistoryResponse(BaseModel):
    """Response model for history endpoint"""
    city: str
    history: List[HistoryEntry]
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Delhi, India",
                "history": [
                    {"day": "2025-01-10", "aqi": 85},
                    {"day": "2025-01-11", "aqi": 90}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    city: str
    predicted_aqi: float = Field(..., ge=0, description="Predicted AQI for next day")
    risk: str = Field(..., description="Risk category")
    health_message: str = Field(..., description="Health message")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Model confidence score")
    model_metrics: Optional[dict] = Field(None, description="Model evaluation metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Delhi, India",
                "predicted_aqi": 102.5,
                "risk": "Moderate",
                "health_message": "Moderate",
                "confidence_score": 0.85,
                "model_metrics": {
                    "r2_score": 0.82,
                    "mae": 8.5
                }
            }
        }


class CompareCitiesRequest(BaseModel):
    """Request model for comparing multiple cities"""
    cities: List[str] = Field(..., min_items=2, max_items=5, description="List of city names to compare")


class CityComparison(BaseModel):
    """Comparison data for a single city"""
    city: str
    current_aqi: int
    predicted_aqi: float
    health_message: str


class CompareCitiesResponse(BaseModel):
    """Response model for city comparison"""
    comparisons: List[CityComparison]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    uptime: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())



