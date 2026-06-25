# Day 13 Mini-Project — CLI Weather-Stub (uses modules + requests)
# Fetches weather data from Open-Meteo (free, no API key).
# Falls back to mock data if network is unavailable.

import json, sys
from datetime import date

try:
    import requests
    NETWORK = True
except ImportError:
    NETWORK = False

WMO_CODES = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    45:"Foggy",61:"Light rain",63:"Moderate rain",65:"Heavy rain",
    71:"Light snow",80:"Rain showers",95:"Thunderstorm"
}

def get_weather_mock():
    return {"city":"Dhaka (mock)","temp_c":30,"wind_kmh":12,"condition":"Partly cloudy"}

def get_weather(lat, lon, city="Location"):
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&current_weather=true")
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()["current_weather"]
        code = int(data["weathercode"])
        return {
            "city":      city,
            "temp_c":    data["temperature"],
            "wind_kmh":  data["windspeed"],
            "condition": WMO_CODES.get(code, f"Code {code}"),
        }
    except Exception as e:
        print(f"Network error ({e}), using mock data.")
        return get_weather_mock()

def show(w):
    print(f"\n{'='*40}")
    print(f"  City       : {w['city']}")
    print(f"  Date       : {date.today()}")
    print(f"  Temperature: {w['temp_c']} °C")
    print(f"  Wind       : {w['wind_kmh']} km/h")
    print(f"  Condition  : {w['condition']}")
    print(f"{'='*40}")

print("=== Weather CLI ===")
print("1. Dhaka  2. London  3. Tokyo  4. Custom  5. Mock only")
ch = input("Choice: ").strip()

locations = {
    "1": (23.72, 90.41, "Dhaka, Bangladesh"),
    "2": (51.51, -0.13, "London, UK"),
    "3": (35.68, 139.69,"Tokyo, Japan"),
}
if ch in locations:
    lat, lon, city = locations[ch]
    show(get_weather(lat, lon, city) if NETWORK else get_weather_mock())
elif ch == "4":
    lat  = float(input("Latitude: "))
    lon  = float(input("Longitude: "))
    city = input("City name: ")
    show(get_weather(lat, lon, city) if NETWORK else get_weather_mock())
else:
    show(get_weather_mock())
