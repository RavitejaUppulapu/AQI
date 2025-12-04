"""
Enhanced FastAPI Backend for Air Pollution Prediction System
"""
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from aqi_service import AQIService
from ml.model import AQIPredictor, get_health_message, backtest_history
from models.schemas import (
    CurrentAQIResponse,
    HistoryResponse,
    PredictionResponse,
    CompareCitiesRequest,
    CompareCitiesResponse,
    CityComparison,
    HealthCheckResponse,
    ErrorResponse,
    ForecastResponse,
    ForecastEntry,
    BacktestResponse,
    BacktestMetrics,
    AlertsResponse,
    AlertItem,
)
from utils.cache import cache
from utils.rate_limiter import rate_limiter
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Air Pollution Prediction API",
    description="Enhanced API for fetching current AQI, history, and predictions with ML",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track uptime
app_start_time = time.time()


def get_client_ip(request: Request) -> str:
    """Extract client IP for rate limiting"""
    if request.client:
        return request.client.host
    return "unknown"


def rate_limit_dependency(request: Request):
    """Dependency for rate limiting"""
    client_ip = get_client_ip(request)
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
            headers={"X-RateLimit-Remaining": "0", "Retry-After": "60"}
        )
    
    return client_ip


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} for {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Air Pollution Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "/current?city=CityName": "Get current AQI data",
            "/history?city=CityName&days=7": "Get AQI history",
            "/predict?city=CityName": "Predict next day AQI",
            "/forecast?city=CityName&days=5": "Multi-day AQI forecast",
            "/compare": "Compare multiple cities",
            "/health": "Health check",
            "/metrics": "API metrics",
            "/model/backtest?city=CityName": "Model backtesting metrics",
            "/alerts?city=CityName": "AQI alerts and warnings",
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_start_time
    return HealthCheckResponse(
        status="healthy",
        version="2.0.0",
        uptime=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "cache_size": cache.size(),
        "uptime_seconds": time.time() - app_start_time,
        "rate_limiting_enabled": settings.RATE_LIMIT_ENABLED,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/current", response_model=CurrentAQIResponse)
async def get_current(
    city: str = Query(..., description="City name to search for", min_length=1),
    client_ip: str = Depends(rate_limit_dependency)
):
    """
    Get current AQI data for a city
    
    Args:
        city: City name
        
    Returns:
        Current AQI data with pollutants, weather, and timestamp
    """
    cache_key = f"current:{city.lower()}"
    
    # Check cache
    cached_data = cache.get(cache_key, ttl=settings.CACHE_TTL)
    if cached_data:
        logger.info(f"Cache hit for {city}")
        return CurrentAQIResponse(**cached_data)
    
    # Fetch from API
    data = await AQIService.get_current_aqi(city)
    
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch AQI data for '{city}'. Please check the city name."
        )
    
    # Add health message
    data["health_message"] = get_health_message(data["aqi"])
    
    # Cache the result
    cache.set(cache_key, data)
    
    return CurrentAQIResponse(**data)


@app.get("/history", response_model=HistoryResponse)
async def get_history(
    city: str = Query(..., description="City name to search for", min_length=1),
    days: int = Query(7, description="Number of days of history", ge=1, le=settings.MAX_HISTORY_DAYS),
    client_ip: str = Depends(rate_limit_dependency)
):
    """
    Get historical AQI data for a city
    
    Args:
        city: City name
        days: Number of days (default 7, max 30)
        
    Returns:
        Historical AQI data with day and AQI values
    """
    cache_key = f"history:{city.lower()}:{days}"
    
    # Check cache
    cached_data = cache.get(cache_key, ttl=settings.CACHE_TTL)
    if cached_data:
        logger.info(f"Cache hit for {city} history ({days} days)")
        return HistoryResponse(**cached_data)
    
    # Fetch from API
    data = await AQIService.get_history_aqi(city, days=days)
    
    if data is None or not data.get("history"):
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch history for '{city}'. Please check the city name."
        )
    
    # Cache the result
    cache.set(cache_key, data)
    
    return HistoryResponse(**data)


@app.get("/predict", response_model=PredictionResponse)
async def get_prediction(
    city: str = Query(..., description="City name to search for", min_length=1),
    client_ip: str = Depends(rate_limit_dependency)
):
    """
    Predict next day AQI for a city
    
    Args:
        city: City name
        
    Returns:
        Predicted AQI with risk level and confidence
    """
    cache_key = f"predict:{city.lower()}"
    
    # Check cache (shorter TTL for predictions)
    cached_data = cache.get(cache_key, ttl=settings.PREDICTION_CACHE_TTL)
    if cached_data:
        logger.info(f"Cache hit for {city} prediction")
        return PredictionResponse(**cached_data)
    
    # Fetch history first
    history_data = await AQIService.get_history_aqi(city, days=7)
    
    if not history_data or not history_data.get("history"):
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch history for '{city}'. Cannot make prediction."
        )
    
    history = history_data["history"]
    
    # Check if we have enough data
    if len(history) < settings.MIN_HISTORY_DAYS_FOR_PREDICTION:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough historical data for prediction (need at least {settings.MIN_HISTORY_DAYS_FOR_PREDICTION} days, got {len(history)})"
        )
    
    # Create predictor and train (can be 'linear', 'random_forest', or 'auto')
    predictor = AQIPredictor(model_type=settings.ML_MODEL_TYPE)
    
    # Train on history
    if not predictor.train(history):
        raise HTTPException(
            status_code=500,
            detail="Failed to train prediction model"
        )
    
    # Make prediction
    prediction_result = predictor.predict(history)
    
    if prediction_result is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate prediction"
        )
    
    # Get risk level
    predicted_aqi = prediction_result["prediction"]
    risk = get_health_message(predicted_aqi)
    
    result = {
        "city": history_data["city"],
        "predicted_aqi": round(predicted_aqi, 1),
        "risk": risk,
        "health_message": risk,
        "confidence_score": round(prediction_result.get("confidence", 0.7), 3),
        "model_metrics": prediction_result.get("metrics", {}),
    }
    
    # Cache the result
    cache.set(cache_key, result)
    
    return PredictionResponse(**result)


@app.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    city: str = Query(..., description="City name to search for", min_length=1),
    days: int = Query(5, ge=1, le=7, description="Number of future days to forecast"),
    client_ip: str = Depends(rate_limit_dependency),
):
    """
    Multi-day AQI forecast for a city.

    Uses the same ML pipeline as `/predict` but iteratively predicts
    the next N days and returns a list of forecast entries.
    """
    # Fetch history (we'll use up to MAX_HISTORY_DAYS for better context)
    history_data = await AQIService.get_history_aqi(city, days=min(7, settings.MAX_HISTORY_DAYS))

    if not history_data or not history_data.get("history"):
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch history for '{city}'. Cannot generate forecast.",
        )

    history = history_data["history"]

    if len(history) < settings.MIN_HISTORY_DAYS_FOR_PREDICTION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough historical data for forecast "
                f"(need at least {settings.MIN_HISTORY_DAYS_FOR_PREDICTION} days, got {len(history)})"
            ),
        )

    # Train predictor (auto-selects best model when configured)
    predictor = AQIPredictor(model_type=settings.ML_MODEL_TYPE)
    if not predictor.train(history):
        raise HTTPException(
            status_code=500,
            detail="Failed to train prediction model for forecast",
        )

    # Iteratively forecast next N days
    from datetime import datetime, timedelta

    forecast_entries: list[ForecastEntry] = []
    temp_history = list(history)

    # Determine starting date from last history entry if possible
    try:
        last_day_str = temp_history[-1]["day"]
        last_date = datetime.strptime(last_day_str, "%Y-%m-%d")
    except Exception:
        last_date = datetime.now()

    for i in range(days):
        res = predictor.predict(temp_history)
        if res is None:
            break
        predicted_aqi = max(0.0, float(res["prediction"]))
        future_date = last_date + timedelta(days=i + 1)
        day_str = future_date.strftime("%Y-%m-%d")

        forecast_entries.append(
            ForecastEntry(
                day=day_str,
                predicted_aqi=round(predicted_aqi, 1),
                health_message=get_health_message(predicted_aqi),
            )
        )

        # Append to history so that next step uses this prediction
        temp_history.append({"day": day_str, "aqi": predicted_aqi})

    if not forecast_entries:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate forecast",
        )

    model_used = predictor.selected_model_type if hasattr(predictor, "selected_model_type") else settings.ML_MODEL_TYPE

    return ForecastResponse(
        city=history_data["city"],
        forecast=forecast_entries,
        model_used=model_used,
        base_date=history[-1]["day"],
    )


@app.post("/compare", response_model=CompareCitiesResponse)
async def compare_cities(
    request: CompareCitiesRequest,
    client_ip: str = Depends(rate_limit_dependency)
):
    """
    Compare AQI data for multiple cities
    
    Args:
        request: Request with list of cities
        
    Returns:
        Comparison data for all cities
    """
    comparisons = []
    
    # Fetch data for all cities in parallel
    import asyncio
    
    async def get_city_data(city_name: str):
        """Get current and predicted AQI for a city"""
        try:
            current_data = await AQIService.get_current_aqi(city_name)
            history_data = await AQIService.get_history_aqi(city_name, days=7)
            
            if not current_data or not history_data:
                return None
            
            # Make prediction
            history = history_data.get("history", [])
            if len(history) >= settings.MIN_HISTORY_DAYS_FOR_PREDICTION:
                predictor = AQIPredictor(model_type=settings.ML_MODEL_TYPE)
                if predictor.train(history):
                    pred_result = predictor.predict(history)
                    predicted_aqi = pred_result["prediction"] if pred_result else current_data["aqi"]
                else:
                    predicted_aqi = current_data["aqi"]
            else:
                predicted_aqi = current_data["aqi"]
            
            return CityComparison(
                city=current_data["city"],
                current_aqi=current_data["aqi"],
                predicted_aqi=round(predicted_aqi, 1),
                health_message=get_health_message(predicted_aqi)
            )
        except Exception as e:
            logger.error(f"Error getting data for {city_name}: {e}")
            return None
    
    # Fetch all cities in parallel
    tasks = [get_city_data(city) for city in request.cities]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    comparisons = [r for r in results if r is not None]
    
    if not comparisons:
        raise HTTPException(
            status_code=404,
            detail="Could not fetch data for any of the specified cities"
        )
    
    return CompareCitiesResponse(comparisons=comparisons)


@app.get("/model/backtest", response_model=BacktestResponse)
async def model_backtest(
    city: str = Query(..., description="City name to backtest for", min_length=1),
    client_ip: str = Depends(rate_limit_dependency),
):
    """
    Backtest the prediction model on historical AQI data.

    Trains the model on progressively growing windows of history and
    evaluates how well it would have predicted past days.
    """
    history_data = await AQIService.get_history_aqi(city, days=min(30, settings.MAX_HISTORY_DAYS))

    if not history_data or not history_data.get("history"):
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch history for '{city}'. Cannot run backtest.",
        )

    history = history_data["history"]
    backtest = backtest_history(history, model_type=settings.ML_MODEL_TYPE)

    if backtest is None:
        raise HTTPException(
            status_code=500,
            detail="Backtest failed or not enough data",
        )

    metrics_data = backtest["metrics"]
    metrics = BacktestMetrics(
        r2_score=metrics_data["r2_score"],
        mae=metrics_data["mae"],
        rmse=metrics_data["rmse"],
        samples=metrics_data["samples"],
    )

    return BacktestResponse(
        city=history_data["city"],
        metrics=metrics,
        trend=backtest.get("trend", "Stable"),
        trend_change_percent=backtest.get("trend_change_percent", 0.0),
        volatility=backtest.get("volatility", 0.0),
    )


@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    city: str = Query(..., description="City name to get alerts for", min_length=1),
    client_ip: str = Depends(rate_limit_dependency),
):
    """
    Generate AQI alerts for a city based on current and predicted AQI.

    Alerts include:
    - Current AQI thresholds (e.g., Unhealthy, Very Unhealthy)
    - Sudden increase warnings when predicted AQI jumps significantly
    """
    current = await AQIService.get_current_aqi(city)
    if current is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch AQI data for '{city}'.",
        )

    current_aqi = current.get("aqi", 0)
    history_data = await AQIService.get_history_aqi(city, days=7)
    if not history_data or not history_data.get("history"):
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch history for '{city}'. Cannot compute alerts.",
        )

    history = history_data["history"]
    predictor = AQIPredictor(model_type=settings.ML_MODEL_TYPE)
    if not predictor.train(history):
        raise HTTPException(
            status_code=500,
            detail="Failed to train model for alerts",
        )

    pred_res = predictor.predict(history)
    if pred_res is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate prediction for alerts",
        )

    predicted_aqi = float(pred_res["prediction"])
    risk_today = get_health_message(current_aqi)
    risk_tomorrow = get_health_message(predicted_aqi)

    alerts: list[AlertItem] = []

    # Current AQI alerts
    if current_aqi > 200:
        alerts.append(
            AlertItem(
                type="current",
                level="critical",
                message=f"Current AQI is {current_aqi} ({risk_today}). Outdoor activity is not recommended.",
            )
        )
    elif current_aqi > 150:
        alerts.append(
            AlertItem(
                type="current",
                level="warning",
                message=f"Current AQI is {current_aqi} ({risk_today}). Sensitive groups should avoid outdoor exertion.",
            )
        )

    # Predicted AQI alerts
    if predicted_aqi > 200:
        alerts.append(
            AlertItem(
                type="forecast",
                level="critical",
                message=f"Predicted AQI for tomorrow is {round(predicted_aqi, 1)} ({risk_tomorrow}). Prepare for very poor air quality.",
            )
        )
    elif predicted_aqi > 150:
        alerts.append(
            AlertItem(
                type="forecast",
                level="warning",
                message=f"Predicted AQI for tomorrow is {round(predicted_aqi, 1)} ({risk_tomorrow}). Consider limiting outdoor activities.",
            )
        )

    # Sudden change warning
    if current_aqi > 0:
        change_percent = (predicted_aqi - current_aqi) / current_aqi * 100.0
        if change_percent >= 20:
            alerts.append(
                AlertItem(
                    type="sudden_change",
                    level="warning",
                    message=f"Predicted AQI is expected to increase by {change_percent:.1f}% tomorrow.",
                )
            )
        elif change_percent <= -20:
            alerts.append(
                AlertItem(
                    type="sudden_change",
                    level="info",
                    message=f"Good news: predicted AQI is expected to decrease by {abs(change_percent):.1f}% tomorrow.",
                )
            )

    return AlertsResponse(
        city=current["city"],
        current_aqi=current_aqi,
        predicted_aqi=round(predicted_aqi, 1),
        risk_today=risk_today,
        risk_tomorrow=risk_tomorrow,
        alerts=alerts,
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Air Pollution Prediction API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
