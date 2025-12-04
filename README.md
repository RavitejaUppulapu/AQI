# Air Pollution Prediction System

A complete end-to-end web application for fetching real-time Air Quality Index (AQI) data and predicting future AQI values for any city using machine learning.

## 🌟 Features

- **Real-time AQI Data**: Fetch current AQI data for any city worldwide
- **Pollutant Monitoring**: Track PM2.5, PM10, O3, NO2, CO levels
- **Weather Data**: View temperature, humidity, and wind speed
- **Historical Trends**: Visualize last 7 days of AQI data with interactive charts
- **ML Predictions**: Predict next-day AQI using Linear Regression
- **Health Alerts**: Color-coded health messages (Good, Moderate, Unhealthy, etc.)
- **Clean UI**: Modern, responsive web interface

## 🏗️ Architecture

```
air_quality_project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── aqi_service.py       # AQICN API integration
│   ├── requirements.txt     # Python dependencies
│   └── ml/
│       ├── model.py         # ML model implementation
│       └── train.py         # Model training script
├── frontend/
│   └── index.html           # Web UI
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Modern web browser

### Installation & Setup

#### Option 1: Direct Installation

1. **Clone or download the project**

2. **Install backend dependencies**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend**

   ```bash
   # From backend directory
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

   # Or using Python directly
   python main.py
   ```

   The backend will be available at `http://localhost:8000`

4. **Open the frontend**

   - Simply open `frontend/index.html` in your web browser
   - Or serve it using a local server:

     ```bash
     # Using Python
     cd frontend
     python -m http.server 8080

     # Then open http://localhost:8080 in your browser
     ```

5. **Test the API**

   Visit `http://localhost:8000/docs` to see the interactive API documentation
   Or test endpoints directly:

   - `http://localhost:8000/current?city=Delhi`
   - `http://localhost:8000/history?city=Delhi`
   - `http://localhost:8000/predict?city=Delhi`
   - `http://localhost:8000/health` - Health check
   - `http://localhost:8000/metrics` - API metrics

## 📡 API Endpoints

### 1. Get Current AQI

**GET** `/current?city={CityName}`

Returns current AQI data for a city with all pollutants and weather data.

**Example:**

```bash
curl "http://localhost:8000/current?city=Delhi"
```

**Response:**

```json
{
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
```

### 2. Get Historical AQI

**GET** `/history?city={CityName}&days={number}`

Returns historical AQI data (default: 7 days, max: 30 days).

**Example:**

```bash
curl "http://localhost:8000/history?city=Delhi&days=7"
```

**Response:**

```json
{
  "city": "Delhi, India",
  "history": [
    {"day": "2025-01-10", "aqi": 85},
    {"day": "2025-01-11", "aqi": 90},
    ...
  ]
}
```

### 3. Predict Next Day AQI

**GET** `/predict?city={CityName}`

Returns predicted AQI for the next day with confidence scores and model metrics.

**Example:**

```bash
curl "http://localhost:8000/predict?city=Delhi"
```

**Response:**

```json
{
  "city": "Delhi, India",
  "predicted_aqi": 102.5,
  "risk": "Moderate",
  "health_message": "Moderate",
  "confidence_score": 0.85,
  "model_metrics": {
    "r2_score": 0.82,
    "mae": 8.5,
    "rmse": 12.3,
    "training_samples": 7
  }
}
```

### 4. Compare Cities

**POST** `/compare`

Compare AQI data for multiple cities (2-5 cities).

**Example:**

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{"cities": ["Delhi", "London", "New York"]}'
```

**Response:**

```json
{
  "comparisons": [
    {
      "city": "Delhi, India",
      "current_aqi": 90,
      "predicted_aqi": 102.5,
      "health_message": "Moderate"
    },
    ...
  ],
  "timestamp": "2025-01-16T14:00:00"
}
```

### 5. Health Check

**GET** `/health`

Check API health and uptime.

**Example:**

```bash
curl "http://localhost:8000/health"
```

### 6. API Metrics

**GET** `/metrics`

Get API performance metrics and cache statistics.

**Example:**

```bash
curl "http://localhost:8000/metrics"
```

## 🧠 Machine Learning Model

The system uses **Linear Regression** or **Random Forest** to predict next-day AQI based on advanced feature engineering.

### Model Details

- **Algorithm**: Linear Regression or Random Forest (configurable)
- **Features**:
  - Day index (trend)
  - 3-day moving average
  - Lag features (previous day AQI)
  - Change from previous day
  - Rolling variance
- **Target**: AQI values
- **Training**: On-the-fly training for each prediction request
- **Model Metrics**: R² score, MAE, RMSE
- **Confidence Scores**: Based on model performance metrics
- **Model Persistence**: Optional model saving with `joblib`

### Feature Engineering

The model uses multiple features to improve prediction accuracy:

1. **Trend**: Day index to capture temporal trends
2. **Moving Average**: 3-day rolling average for smoothing
3. **Lag Features**: Previous day's AQI value
4. **Change**: Day-to-day change in AQI
5. **Variance**: Rolling variance to capture volatility

### Training Script

You can pre-train models using:

```bash
cd backend
python ml/train.py
```

## 🎨 Frontend

The frontend is a single-page application built with:

- **Vanilla JavaScript** (no framework dependencies)
- **Chart.js** for data visualization
- **Modern CSS** with gradients and responsive design

### Features

- Search any city by name
- Real-time AQI display with color-coded health indicators
- Interactive 7-day history chart
- Next-day AQI prediction
- Responsive design for mobile and desktop

## 📊 AQI Health Categories

| AQI Range | Category                       | Health Message                                                        |
| --------- | ------------------------------ | --------------------------------------------------------------------- |
| 0-50      | Good                           | Green - Air quality is satisfactory                                   |
| 51-100    | Moderate                       | Yellow - Acceptable for most people                                   |
| 101-150   | Unhealthy for Sensitive Groups | Orange - Members of sensitive groups may experience health effects    |
| 151-200   | Unhealthy                      | Red - Everyone may begin to experience health effects                 |
| 201-300   | Very Unhealthy                 | Purple - Health alert: everyone may experience serious health effects |
| 300+      | Hazardous                      | Maroon - Health warning of emergency conditions                       |

## 🔧 Configuration

### Environment Variables

The application supports configuration via environment variables. Create a `.env` file (see `.env.example`):

```bash
# API Configuration
AQICN_API_TOKEN=your_token_here
API_TIMEOUT=10

# Cache Configuration
CACHE_TTL=300
PREDICTION_CACHE_TTL=60

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60

# ML Configuration
ML_MODEL_TYPE=linear  # or random_forest

# Logging
LOG_LEVEL=INFO
```

### API Token

The AQICN API token can be configured via environment variable `AQICN_API_TOKEN`. To get your own token:

1. Sign up at [aqicn.org](https://aqicn.org/api/)
2. Get your API token
3. Set it in `.env` file or export as environment variable

### Caching

The backend implements advanced thread-safe caching with TTL and LRU eviction:

- Default cache TTL: 5 minutes
- Prediction cache TTL: 1 minute
- Max cache size: 1000 items
- Automatic cleanup of expired entries

## 🛠️ Development

### Project Structure

```
backend/
├── main.py           # FastAPI routes and endpoints
├── aqi_service.py    # AQICN API client
└── ml/
    ├── model.py      # ML model class
    └── train.py      # Training utilities
```

### Adding New Features

1. **New Endpoint**: Add route in `backend/main.py`
2. **API Integration**: Extend `backend/aqi_service.py`
3. **ML Improvements**: Modify `backend/ml/model.py`

## 📝 Notes

- The AQICN API has rate limits - caching helps reduce API calls
- Historical data availability depends on the API response
- The ML model trains on-the-fly for each prediction to ensure freshness
- Some cities may not have all weather data (temperature, humidity, wind)

## 🐛 Troubleshooting

### Backend not starting

- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r backend/requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

### API errors

- Verify city name is correct (try common names like "Delhi", "London", "New York")
- Check API token is valid in `aqi_service.py`
- Ensure internet connection is active

### Frontend not loading data

- Verify backend is running on `http://localhost:8000`
- Check browser console for errors
- Update `API_BASE_URL` in `frontend/index.html` if backend is on different port

## 🆕 What's New (Version 2.0)

### ✨ Major Improvements

- ✅ **Async/Await**: Non-blocking API calls for better performance
- ✅ **Advanced Caching**: Thread-safe TTL cache with LRU eviction
- ✅ **Rate Limiting**: Protect API from abuse (60 req/min per IP)
- ✅ **Enhanced ML Model**: Feature engineering with 5 features, confidence scores, metrics
- ✅ **New Endpoints**: Compare cities, health check, metrics
- ✅ **Better Error Handling**: Custom exception handlers, detailed error messages
- ✅ **Logging**: Structured logging for debugging and monitoring
- ✅ **Pydantic Models**: Request/response validation with automatic documentation
- ✅ **Frontend Enhancements**: Confidence scores display, model metrics, loading animations
- ✅ **Configuration**: Environment-based configuration management

## 🔒 Security Features

- Rate limiting (configurable)
- Input validation with Pydantic
- CORS configuration
- Error message sanitization
- Request logging for audit trails

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve this project!

## 🙏 Acknowledgments

- AQICN API for providing air quality data
- scikit-learn for ML capabilities
- Chart.js for visualization
- FastAPI for the excellent web framework
- httpx for async HTTP requests

---

**Happy Coding! 🌍✨**
