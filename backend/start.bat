@echo off
REM Startup script for backend server (Windows)

echo Starting Air Pollution Prediction API...
echo Backend will be available at http://localhost:8000
echo API docs will be available at http://localhost:8000/docs
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000



