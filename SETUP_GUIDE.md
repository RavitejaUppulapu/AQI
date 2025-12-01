# Setup Guide - Running on a New Laptop

This guide will help you set up and run the Air Pollution Prediction System on a new laptop.

## 📋 Prerequisites

Before starting, ensure you have:

1. **Python 3.10 or higher** installed

   - Check version: `python --version` or `python3 --version`
   - Download from: https://www.python.org/downloads/

2. **pip** (Python package manager) - usually comes with Python

   - Check: `pip --version` or `pip3 --version`

3. **Modern web browser** (Chrome, Firefox, Edge, Safari)

4. **Internet connection** (required for API calls)

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Project Directory

Open terminal/command prompt and navigate to the project folder:

```bash
cd path/to/air_quality_project
# Example: cd D:\VIRTUSA\training\CP2
```

### Step 2: Install Backend Dependencies

```bash
# Navigate to backend folder
cd backend

# Install all required packages
pip install -r requirements.txt

# If pip doesn't work, try:
pip3 install -r requirements.txt

# On Windows, you might need:
python -m pip install -r requirements.txt
```

**Expected output:** You should see packages being installed (fastapi, uvicorn, httpx, scikit-learn, etc.)

### Step 3: Verify Installation

Check if all packages are installed correctly:

```bash
python -c "import fastapi, uvicorn, httpx, sklearn; print('All packages installed!')"
```

### Step 4: Start the Backend Server

From the `backend` directory, run:

```bash
# Option 1: Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python main.py

# Option 3: Using the startup script (Windows)
start.bat

# Option 3: Using the startup script (Linux/Mac)
chmod +x start.sh
./start.sh
```

**You should see:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Keep this terminal window open!** The server needs to keep running.

### Step 5: Test the Backend

Open a **new terminal window** and test if the API is working:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Or open in browser:
# http://localhost:8000/docs (Interactive API documentation)
# http://localhost:8000/health (Health check)
```

### Step 6: Open the Frontend

You have two options:

#### Option A: Direct File Open (Simplest)

1. Navigate to the `frontend` folder in File Explorer/Finder
2. Double-click `index.html`
3. It will open in your default browser

#### Option B: Using a Local Server (Recommended)

```bash
# Navigate to frontend folder
cd frontend

# Using Python's built-in server
python -m http.server 8080

# Or
python3 -m http.server 8080
```

Then open: `http://localhost:8080` in your browser

### Step 7: Test the Application

1. In the frontend, enter a city name (e.g., "Delhi", "London", "New York")
2. Click "Get AQI Data"
3. You should see:
   - Current AQI data
   - 7-day history chart
   - Predicted AQI for tomorrow

## 🔧 Troubleshooting

### Issue: "python: command not found"

**Solution:**

- Windows: Use `py` instead of `python`
- Mac/Linux: Use `python3` instead of `python`
- Or install Python from https://www.python.org/downloads/

### Issue: "pip: command not found"

**Solution:**

```bash
# Windows
python -m pip install -r requirements.txt

# Mac/Linux
python3 -m pip install -r requirements.txt
```

### Issue: "Port 8000 already in use"

**Solution:**

```bash
# Use a different port
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Then update frontend/index.html:
# Change API_BASE_URL from 'http://localhost:8000' to 'http://localhost:8001'
```

### Issue: "Module not found" errors

**Solution:**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install specific missing package
pip install fastapi uvicorn httpx scikit-learn numpy pandas joblib
```

### Issue: Frontend can't connect to backend

**Check:**

1. Is the backend server running? (Check terminal)
2. Is the port correct? (Default: 8000)
3. Open `frontend/index.html` and check `API_BASE_URL` variable
4. Try accessing `http://localhost:8000/docs` in browser

### Issue: "Connection refused" or "Failed to fetch"

**Solutions:**

1. Make sure backend is running on port 8000
2. Check firewall settings
3. Try `http://127.0.0.1:8000` instead of `localhost:8000`
4. Update `API_BASE_URL` in `frontend/index.html` if using different port

## 🐳 Alternative: Using Docker (If Installed)

If you have Docker installed:

```bash
# From project root directory
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📝 Quick Reference Commands

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. In another terminal, serve frontend (optional)
cd frontend
python -m http.server 8080

# 4. Access:
# Backend API: http://localhost:8000/docs
# Frontend: Open frontend/index.html or http://localhost:8080
```

## ✅ Verification Checklist

- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Backend server running (port 8000)
- [ ] Can access `http://localhost:8000/docs`
- [ ] Frontend opens in browser
- [ ] Can search for a city and see results

## 🆘 Still Having Issues?

1. **Check Python version**: `python --version` (should be 3.10+)
2. **Check if port 8000 is free**:
   - Windows: `netstat -ano | findstr :8000`
   - Mac/Linux: `lsof -i :8000`
3. **Check backend logs** for error messages
4. **Check browser console** (F12) for frontend errors
5. **Verify internet connection** (API needs internet)

## 📞 Common Error Messages

| Error                    | Solution                                                |
| ------------------------ | ------------------------------------------------------- |
| `ModuleNotFoundError`    | Run `pip install -r requirements.txt`                   |
| `Address already in use` | Change port or stop other service on port 8000          |
| `Connection refused`     | Make sure backend is running                            |
| `CORS error`             | Backend CORS is configured, check if backend is running |
| `Failed to fetch`        | Check API_BASE_URL in frontend/index.html               |

---

**That's it! You should now have the application running on your new laptop! 🎉**
