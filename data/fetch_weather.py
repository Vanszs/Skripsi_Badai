import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class WeatherFetcher:
    """
    Fetches real weather data from OpenWeatherMap One Call API 3.0.
    """
    def __init__(self, api_key, lat, lon):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.base_url = "https://api.openweathermap.org/data/3.0/onecall"

    def fetch_all_data(self):
        """
        Fetches 'current', 'hourly', 'daily', and 'alerts' data.
        Returns a dictionary or raises an exception on failure.
        """
        if not self.api_key:
            raise ValueError("API Key is missing. Please provide a valid OpenWeatherMap API Key.")

        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "exclude": "minutely",
            "units": "metric"  # pressure in hPa, wind in m/s
        }

    def fetch_timestamp(self, dt):
        """
        Fetches weather data for a specific timestamp (Historical).
        Endpoint: /onecall/timemachine
        """
        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "dt": int(dt),
            "appid": self.api_key,
            "units": "metric"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching timestamp {dt}: {e}")
            return None






