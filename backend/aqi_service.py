"""
AQI Service for fetching data from AQICN API with async support
"""
import httpx
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
except ImportError:
    # Fallback if config not found
    class Settings:
        AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN", "70fa7c8239bbf05b4090b6992f04d1047fc96267")
        AQICN_BASE_URL = "https://api.waqi.info/feed/{city}/?token={token}"
        API_TIMEOUT = 10
    settings = Settings()

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _pm25_to_aqi(pm25: float) -> int:
    """
    Approximate US EPA AQI from PM2.5 (µg/m³).
    This is used for providers that only return PM2.5 concentrations.
    """
    if pm25 is None:
        return 0
    c = float(pm25)
    # Breakpoints for PM2.5 (µg/m³) -> AQI
    if c <= 12.0:
        return int((50.0 / 12.0) * c)
    elif c <= 35.4:
        return int((c - 12.1) * (100 - 51) / (35.4 - 12.1) + 51)
    elif c <= 55.4:
        return int((c - 35.5) * (150 - 101) / (55.4 - 35.5) + 101)
    elif c <= 150.4:
        return int((c - 55.5) * (200 - 151) / (150.4 - 55.5) + 151)
    elif c <= 250.4:
        return int((c - 150.5) * (300 - 201) / (250.4 - 150.5) + 201)
    elif c <= 350.4:
        return int((c - 250.5) * (400 - 301) / (350.4 - 250.5) + 301)
    else:
        return 500


class AQIService:
    """Service to interact with AQICN and fallback AQI providers"""

    @staticmethod
    def _get_url(city: str) -> str:
        """Generate AQICN city URL"""
        return settings.AQICN_BASE_URL.format(city=city, token=settings.AQICN_API_TOKEN)

    @staticmethod
    def _get_geo_url(lat: float, lon: float) -> str:
        """Generate AQICN geo URL"""
        return f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={settings.AQICN_API_TOKEN}"

    # ---------- Primary: AQICN ----------
    @staticmethod
    async def _fetch_aqicn_city(client: httpx.AsyncClient, city: str) -> Optional[Dict]:
        url = AQIService._get_url(city)
        logger.info(f"Fetching AQICN city data for: {city}")
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                logger.warning("AQICN city lookup failed for %s: %s", city, data.get("data"))
                return None
            return AQIService._normalize_aqicn_payload(data.get("data", {}), fallback_city=city)
        except Exception as e:
            logger.error("Error calling AQICN city API for %s: %s", city, e)
            return None

    @staticmethod
    async def _fetch_aqicn_geo(client: httpx.AsyncClient, lat: float, lon: float, city_hint: str) -> Optional[Dict]:
        url = AQIService._get_geo_url(lat, lon)
        logger.info(f"Fetching AQICN geo data for: {city_hint} ({lat},{lon})")
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                logger.warning("AQICN geo lookup failed for %s: %s", city_hint, data.get("data"))
                return None
            return AQIService._normalize_aqicn_payload(data.get("data", {}), fallback_city=city_hint)
        except Exception as e:
            logger.error("Error calling AQICN geo API for %s: %s", city_hint, e)
            return None

    @staticmethod
    def _normalize_aqicn_payload(aq_data: Dict, fallback_city: str) -> Dict:
        """Normalize AQICN payload to unified current AQI schema."""
        aqi = aq_data.get("aqi", 0)
        iaqi = aq_data.get("iaqi", {}) or {}

        def _v(key):
            item = iaqi.get(key)
            return item.get("v") if isinstance(item, dict) else None

        pm25 = _v("pm25") or 0
        pm10 = _v("pm10") or 0
        o3 = _v("o3") or 0
        no2 = _v("no2") or 0
        co = _v("co") or 0
        temperature = _v("t")
        humidity = _v("h")
        wind = _v("w")

        time_data = aq_data.get("time", {}) or {}
        timestamp = time_data.get("iso", "")

        city_name = aq_data.get("city", {}).get("name", fallback_city)

        return {
            "city": city_name,
            "aqi": int(aqi),
            "pm25": float(pm25),
            "pm10": float(pm10),
            "o3": float(o3),
            "no2": float(no2),
            "co": float(co),
            "temperature": float(temperature) if temperature is not None else None,
            "humidity": float(humidity) if humidity is not None else None,
            "wind": float(wind) if wind is not None else None,
            "time": timestamp,
        }

    # ---------- Geocoding (multiple providers) ----------
    @staticmethod
    async def _geocode_city(client: httpx.AsyncClient, city: str) -> Optional[Dict]:
        """
        Geocode city name to lat/lon using multiple providers with fallback.
        Priority: OpenWeather -> Nominatim (OpenStreetMap) -> None
        """
        # Try OpenWeather first (if API key available)
        if settings.OPENWEATHER_API_KEY:
            try:
                url = "https://api.openweathermap.org/geo/1.0/direct"
                params = {"q": city, "limit": 1, "appid": settings.OPENWEATHER_API_KEY}
                logger.info("Geocoding city via OpenWeather: %s", city)
                resp = await client.get(url, params=params, timeout=5.0)
                resp.raise_for_status()
                data = resp.json()
                if data and len(data) > 0:
                    entry = data[0]
                    result = {
                        "lat": float(entry.get("lat")),
                        "lon": float(entry.get("lon")),
                        "name": entry.get("name") or city,
                    }
                    logger.info("OpenWeather geocoding successful for %s: %s,%s", city, result["lat"], result["lon"])
                    return result
            except Exception as e:
                logger.warning("OpenWeather geocoding failed for %s: %s", city, e)
        
        # Fallback: Nominatim (OpenStreetMap) - FREE, no API key needed
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": city,
                "format": "json",
                "limit": 1,
                "addressdetails": 1
            }
            headers = {
                "User-Agent": "AirQualityApp/1.0"  # Required by Nominatim
            }
            logger.info("Geocoding city via Nominatim: %s", city)
            resp = await client.get(url, params=params, headers=headers, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                entry = data[0]
                result = {
                    "lat": float(entry.get("lat")),
                    "lon": float(entry.get("lon")),
                    "name": entry.get("display_name", city).split(",")[0],  # Use first part of display name
                }
                logger.info("Nominatim geocoding successful for %s: %s,%s", city, result["lat"], result["lon"])
                return result
        except Exception as e:
            logger.warning("Nominatim geocoding failed for %s: %s", city, e)
        
        logger.error("All geocoding providers failed for city %s", city)
        return None

    @staticmethod
    async def _fetch_openweather_aqi(
        client: httpx.AsyncClient, lat: float, lon: float, city_hint: str
    ) -> Optional[Dict]:
        """Fetch AQI from OpenWeather Air Pollution API."""
        if not settings.OPENWEATHER_API_KEY:
            return None
        try:
            url = "https://api.openweathermap.org/data/2.5/air_pollution"
            params = {"lat": lat, "lon": lon, "appid": settings.OPENWEATHER_API_KEY}
            logger.info("Fetching OpenWeather AQI for %s (%s,%s)", city_hint, lat, lon)
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("list") or []
            if not items:
                logger.warning("OpenWeather AQI list empty for %s", city_hint)
                return None
            item = items[0]
            main = item.get("main", {}) or {}
            components = item.get("components", {}) or {}

            # OpenWeather AQI index: 1..5. Map to approximate 0-500 scale.
            ow_aqi = int(main.get("aqi", 1))
            ow_aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
            aqi = ow_aqi_map.get(ow_aqi, 100)

            pm25 = components.get("pm2_5", 0.0)
            pm10 = components.get("pm10", 0.0)
            o3 = components.get("o3", 0.0)
            no2 = components.get("no2", 0.0)
            co = components.get("co", 0.0)

            timestamp = ""
            dt_val = item.get("dt")
            if dt_val:
                try:
                    from datetime import datetime as _dt

                    timestamp = _dt.utcfromtimestamp(dt_val).iso8601()
                except Exception:
                    pass

            return {
                "city": city_hint,
                "aqi": int(aqi),
                "pm25": float(pm25),
                "pm10": float(pm10),
                "o3": float(o3),
                "no2": float(no2),
                "co": float(co),
                "temperature": None,
                "humidity": None,
                "wind": None,
                "time": timestamp,
            }
        except Exception as e:
            logger.error("Error calling OpenWeather AQI for %s: %s", city_hint, e)
            return None

    # ---------- Fallback: Open-Meteo ----------
    @staticmethod
    async def _fetch_openmeteo_current(
        client: httpx.AsyncClient, lat: float, lon: float, city_hint: str
    ) -> Optional[Dict]:
        """Fetch latest AQI-like data from Open-Meteo Air Quality API."""
        try:
            url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm10,pm2_5,carbon_monoxide,ozone,nitrogen_dioxide,sulphur_dioxide",
                "past_days": 1,
                "forecast_days": 0,
            }
            logger.info("Fetching Open-Meteo AQI for %s (%s,%s)", city_hint, lat, lon)
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly") or {}
            times = hourly.get("time") or []
            pm25_list = hourly.get("pm2_5") or []
            pm10_list = hourly.get("pm10") or []
            co_list = hourly.get("carbon_monoxide") or []
            o3_list = hourly.get("ozone") or []
            no2_list = hourly.get("nitrogen_dioxide") or []

            if not times or not pm25_list:
                logger.warning("Open-Meteo AQI hourly empty for %s", city_hint)
                return None

            # Use last hour
            idx = len(times) - 1
            pm25 = pm25_list[idx]
            pm10 = pm10_list[idx] if idx < len(pm10_list) else 0
            co = co_list[idx] if idx < len(co_list) else 0
            o3 = o3_list[idx] if idx < len(o3_list) else 0
            no2 = no2_list[idx] if idx < len(no2_list) else 0

            aqi = _pm25_to_aqi(pm25)
            timestamp = times[idx]

            return {
                "city": city_hint,
                "aqi": int(aqi),
                "pm25": float(pm25 or 0),
                "pm10": float(pm10 or 0),
                "o3": float(o3 or 0),
                "no2": float(no2 or 0),
                "co": float(co or 0),
                "temperature": None,
                "humidity": None,
                "wind": None,
                "time": timestamp,
            }
        except Exception as e:
            logger.error("Error calling Open-Meteo AQI for %s: %s", city_hint, e)
            return None

    @staticmethod
    async def _fetch_openmeteo_history(
        client: httpx.AsyncClient, lat: float, lon: float, city_hint: str, days: int
    ) -> Optional[Dict]:
        """Fetch multi-day history from Open-Meteo and aggregate to daily AQI."""
        try:
            url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm2_5",
                "past_days": days,
                "forecast_days": 0,
            }
            logger.info("Fetching Open-Meteo history for %s (%s,%s)", city_hint, lat, lon)
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly") or {}
            times = hourly.get("time") or []
            pm25_list = hourly.get("pm2_5") or []
            if not times or not pm25_list:
                logger.warning("Open-Meteo history empty for %s", city_hint)
                return None

            from collections import defaultdict

            by_day = defaultdict(list)
            for t, v in zip(times, pm25_list):
                if v is None:
                    continue
                day = t.split("T")[0]
                by_day[day].append(v)

            history = []
            # Sort days chronologically
            for day in sorted(by_day.keys())[-days:]:
                vals = by_day[day]
                if not vals:
                    continue
                avg_pm25 = sum(vals) / len(vals)
                aqi = _pm25_to_aqi(avg_pm25)
                history.append({"day": day, "aqi": int(aqi)})

            if not history:
                return None

            return {"city": city_hint, "history": history}
        except Exception as e:
            logger.error("Error calling Open-Meteo history for %s: %s", city_hint, e)
            return None

    # ---------- Public methods with fallback chain ----------
    @staticmethod
    async def get_current_aqi(city: str) -> Optional[Dict]:
        """
        Fetch current AQI data for a city (async) using multi-provider fallback.

        Priority:
          1) AQICN by city name
          2) AQICN by geo (lat/lon via OpenWeather geocoding)
          3) OpenWeather Air Pollution API
          4) Open-Meteo Air Quality API
        """
        try:
            async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
                # 1) AQICN by city name
                result = await AQIService._fetch_aqicn_city(client, city)
                if result:
                    return result

                # Need coordinates for remaining providers
                geo = await AQIService._geocode_city(client, city)
                if not geo:
                    logger.error("Unable to determine coordinates for city %s", city)
                    return None
                lat = geo["lat"]
                lon = geo["lon"]
                city_name = geo["name"] or city

                # 2) AQICN by geo
                result = await AQIService._fetch_aqicn_geo(client, lat, lon, city_name)
                if result:
                    return result

                # 3) OpenWeather
                result = await AQIService._fetch_openweather_aqi(
                    client, lat, lon, city_name
                )
                if result:
                    return result

                # 4) Open-Meteo
                result = await AQIService._fetch_openmeteo_current(
                    client, lat, lon, city_name
                )
                if result:
                    return result

                logger.error("All AQI providers failed for city %s", city)
                return None
        except httpx.TimeoutException:
            logger.error("Timeout while fetching AQI data for %s", city)
            return None
        except Exception as e:
            logger.error("Error in get_current_aqi for %s: %s", city, e, exc_info=True)
            return None

    @staticmethod
    async def get_history_aqi(city: str, days: int = 7) -> Optional[Dict]:
        """
        Fetch historical AQI data for a city (async) using multi-provider fallback.

        Priority:
          1) AQICN (forecast/time series if available)
          2) Open-Meteo (past_days) aggregated to daily AQI (via PM2.5)
        """
        try:
            async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
                # 1) Attempt AQICN history via city endpoint
                try:
                    url = AQIService._get_url(city)
                    logger.info("Fetching %s days history from AQICN for %s", days, city)
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("status") == "ok":
                        aq_data = data.get("data", {}) or {}
                        city_name = aq_data.get("city", {}).get("name", city)

                        forecast = aq_data.get("forecast", {}) or {}
                        daily = forecast.get("daily", {}) or {}
                        history: List[Dict] = []

                        if daily.get("pm25"):
                            pm25_forecast = daily.get("pm25") or []
                            for day_data in pm25_forecast[:days]:
                                day_str = day_data.get("day", "")
                                avg_aqi = day_data.get("avg", day_data.get("max", 0))
                                history.append({"day": day_str, "aqi": int(avg_aqi)})

                        if history:
                            return {"city": city_name, "history": history[:days]}
                except Exception as e:
                    logger.warning("AQICN history fetch failed for %s: %s", city, e)

                # 2) Fallback to Open-Meteo aggregated history
                geo = await AQIService._geocode_city(client, city)
                if not geo:
                    logger.error("Unable to determine coordinates for history of %s", city)
                    return None
                lat = geo["lat"]
                lon = geo["lon"]
                city_name = geo["name"] or city

                result = await AQIService._fetch_openmeteo_history(
                    client, lat, lon, city_name, days
                )
                if result:
                    # Ensure we only return the last `days` items
                    result["history"] = result["history"][-days:]
                    return result

                logger.error("All history providers failed for city %s", city)
                return None
        except httpx.TimeoutException:
            logger.error("Timeout while fetching history for %s", city)
            return None
        except Exception as e:
            logger.error("Error in get_history_aqi for %s: %s", city, e, exc_info=True)
            return None
