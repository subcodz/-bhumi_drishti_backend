import os
import time
import urllib.request
import json
import ssl
from datetime import datetime, timezone
from typing import Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# IMD (India Meteorological Department) Configuration
IMD_API_URL = os.getenv("IMD_API_URL", "https://mausam.imd.gov.in/api/v1/rainfall")
IMD_API_KEY = os.getenv("IMD_API_KEY", "")

# Key IMD Regional Meteorological Stations in North Eastern Region (NER)
NER_STATE_STATIONS: Dict[str, Dict[str, Any]] = {
    "AS": {"name": "Assam", "station": "RMC Guwahati (Borjhar)", "lat": 26.14, "lon": 91.73},
    "AR": {"name": "Arunachal Pradesh", "station": "IMD Itanagar", "lat": 27.10, "lon": 93.62},
    "MN": {"name": "Manipur", "station": "IMD Imphal (Tulihal)", "lat": 24.81, "lon": 93.94},
    "ML": {"name": "Meghalaya", "station": "IMD Shillong (Barapani)", "lat": 25.57, "lon": 91.88},
    "MZ": {"name": "Mizoram", "station": "IMD Aizawl (Lengpui)", "lat": 23.73, "lon": 92.72},
    "NL": {"name": "Nagaland", "station": "IMD Kohima", "lat": 25.67, "lon": 94.10},
    "TR": {"name": "Tripura", "station": "IMD Agartala (Singerbhil)", "lat": 23.83, "lon": 91.28},
}

MET_SYMBOL_MAP = {
    "clearsky_day": "Sunny / Clear",
    "clearsky_night": "Clear Night",
    "fair_day": "Mainly Sunny",
    "fair_night": "Mainly Clear",
    "partlycloudy_day": "Partly Cloudy",
    "partlycloudy_night": "Partly Cloudy",
    "cloudy": "Overcast",
    "fog": "Misty / Fog",
    "lightrain": "Light Rain",
    "rain": "Rain Showers",
    "heavyrain": "Heavy Monsoon Rain",
    "lightrainshowers_day": "Passing Showers",
    "lightrainshowers_night": "Passing Showers",
    "rainshowers_day": "Rain Showers",
    "rainshowers_night": "Rain Showers",
    "heavysnow": "Heavy Snow",
    "snow": "Snow",
    "lightsnow": "Light Snow",
    "thunderstorm": "Thunderstorms",
}

WMO_CODE_MAP = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Monsoon Rain",
    80: "Slight Rain Showers",
    81: "Moderate Showers",
    82: "Violent Rain Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Slight Hail",
    99: "Thunderstorm with Heavy Hail",
}

_ner_weather_cache: Dict[str, Dict[str, Any]] = {}
_ner_cache_timestamp: float = 0.0
CACHE_TTL_SECONDS = 900.0  # 15 minutes


def deg_to_compass(deg: float) -> str:
    val = int((deg / 22.5) + 0.5)
    arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return arr[(val % 16)]


def generate_advisory(r1: float, r24: float, temp: float, condition: str) -> str:
    if r24 >= 50.0 or r1 >= 15.0:
        return f"Red Alert: Severe downpour ({r24:.1f}mm in 24h) — high landslide & flash flood triggering potential on hill cuttings"
    if r24 >= 20.0 or r1 >= 5.0:
        return f"Orange Alert: Substantial rainfall ({r24:.1f}mm in 24h) — caution on steep slope road passes and river approaches"
    if r24 >= 5.0 or r1 >= 1.0:
        return f"Yellow Alert: Active rain showers ({r24:.1f}mm in 24h) — roadway ponding and wet hill slopes observed"
    return f"Advisory: Stable atmospheric conditions ({temp:.1f}°C, {condition}) — routine corridor monitoring active"


def _fetch_single_state_met_no(code: str) -> Tuple[str, Dict[str, Any]]:
    info = NER_STATE_STATIONS[code]
    lat, lon = info["lat"], info["lon"]
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {"User-Agent": "BhumiDrishtiRoadRisk/1.0 (contact: subhajitbaidya@ner-transport.gov.in)"}

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=7, context=ctx) as resp:
        data = json.loads(resp.read().decode())
        ts = data.get("properties", {}).get("timeseries", [])
        if not ts:
            raise ValueError(f"No timeseries for {code}")

        ts0 = ts[0]["data"]
        instant = ts0.get("instant", {}).get("details", {})
        temp = float(instant.get("air_temperature", 24.0))
        hum = int(instant.get("relative_humidity", 85))
        w_speed_ms = float(instant.get("wind_speed", 3.5))
        w_speed_kmh = round(w_speed_ms * 3.6, 1)
        w_dir_deg = float(instant.get("wind_from_direction", 210.0))
        w_dir = deg_to_compass(w_dir_deg)

        next_1h = ts0.get("next_1_hours", {})
        next_6h = ts0.get("next_6_hours", {})
        r_now = float(next_1h.get("details", {}).get("precipitation_amount", 0.0))
        r_6h = float(next_6h.get("details", {}).get("precipitation_amount", r_now * 4))

        r_24 = round(float(sum(
            item["data"].get("next_1_hours", {}).get("details", {}).get("precipitation_amount", 0.0)
            for item in ts[:24]
        )), 1)

        r_72 = round(float(sum(
            item["data"].get("next_1_hours", {}).get("details", {}).get("precipitation_amount", 0.0)
            for item in ts[:72]
        )), 1)
        if r_72 == 0.0 and r_24 > 0:
            r_72 = round(r_24 * 1.8, 1)

        sym_code = next_1h.get("summary", {}).get("symbol_code", "clearsky_day")
        condition = MET_SYMBOL_MAP.get(sym_code)
        if not condition:
            base_sym = sym_code.replace("_day", "").replace("_night", "")
            condition = MET_SYMBOL_MAP.get(base_sym, sym_code.replace("_", " ").title())

        advisory = generate_advisory(r_now, r_24, temp, condition)

        return code, {
            "state_code": code,
            "state_name": info["name"],
            "station_name": info["station"],
            "temperature_c": round(temp, 1),
            "relative_humidity_pct": hum,
            "rainfall_1h_mm": round(r_now, 1),
            "rainfall_6h_mm": round(r_6h, 1),
            "rainfall_24h_mm": r_24,
            "rainfall_72h_mm": r_72,
            "wind_speed_kmh": w_speed_kmh,
            "wind_direction": w_dir,
            "weather_condition": condition,
            "wmo_code": 1,
            "advisory": advisory,
            "source": "live_met_norway",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }


def fetch_all_ner_live_weather(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Fetches real-time meteorological observations for all 7 North Eastern states.
    Primary provider: MET Norway (open-access ECMWF global model, unthrottled on cloud IPs).
    Secondary provider: Open-Meteo multi-coordinate batch.
    Fallback: Seasonal baseline.
    Results cached for 15 minutes.
    """
    global _ner_weather_cache, _ner_cache_timestamp
    now_ts = time.time()

    if not force_refresh and _ner_weather_cache and (now_ts - _ner_cache_timestamp < CACHE_TTL_SECONDS):
        return _ner_weather_cache

    last_err = None

    # --- 1. PRIMARY: MET Norway Parallel Multi-Station Fetch ---
    try:
        with ThreadPoolExecutor(max_workers=7) as executor:
            results = list(executor.map(_fetch_single_state_met_no, NER_STATE_STATIONS.keys()))
        new_cache = {code: data for code, data in results}
        if len(new_cache) == len(NER_STATE_STATIONS):
            _ner_weather_cache = new_cache
            _ner_cache_timestamp = now_ts
            return _ner_weather_cache
    except Exception as e_met:
        last_err = f"MET Norway err: {e_met}"
        print(f"MET Norway fetch error: {e_met}")

    # --- 2. SECONDARY: Open-Meteo Multi-Coordinate Batch ---
    try:
        codes = list(NER_STATE_STATIONS.keys())
        lats = ",".join(str(NER_STATE_STATIONS[c]["lat"]) for c in codes)
        lons = ",".join(str(NER_STATE_STATIONS[c]["lon"]) for c in codes)
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lats}&longitude={lons}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code"
            f"&hourly=precipitation&past_days=3&forecast_days=1"
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(om_url, headers={"User-Agent": "RoadRiskNER-Backend/0.5"})
        with urllib.request.urlopen(req, timeout=8, context=ctx) as res:
            om_data = json.loads(res.read().decode())
            results_list = om_data if isinstance(om_data, list) else [om_data]
            new_cache = {}
            for i, code in enumerate(codes):
                st_info = NER_STATE_STATIONS[code]
                item = results_list[i] if i < len(results_list) else {}
                curr = item.get("current", {})
                hourly = item.get("hourly", {}).get("precipitation", [])
                temp = float(curr.get("temperature_2m", 24.0))
                humidity = int(curr.get("relative_humidity_2m", 85))
                r_now = float(curr.get("precipitation", 0.0))
                w_speed = float(curr.get("wind_speed_10m", 15.0))
                w_dir = deg_to_compass(float(curr.get("wind_direction_10m", 210.0)))
                w_code = int(curr.get("weather_code", 2))
                condition = WMO_CODE_MAP.get(w_code, "Partly Cloudy")
                r6 = round(float(sum(hourly[-6:])), 1) if len(hourly) >= 6 else round(r_now * 4, 1)
                r24 = round(float(sum(hourly[-24:])), 1) if len(hourly) >= 24 else round(r_now * 10, 1)
                r72 = round(float(sum(hourly[-72:])), 1) if len(hourly) >= 72 else round(r_now * 20, 1)
                advisory = generate_advisory(r_now, r24, temp, condition)
                new_cache[code] = {
                    "state_code": code,
                    "state_name": st_info["name"],
                    "station_name": st_info["station"],
                    "temperature_c": round(temp, 1),
                    "relative_humidity_pct": humidity,
                    "rainfall_1h_mm": round(r_now, 1),
                    "rainfall_6h_mm": r6,
                    "rainfall_24h_mm": r24,
                    "rainfall_72h_mm": r72,
                    "wind_speed_kmh": round(w_speed, 1),
                    "wind_direction": w_dir,
                    "weather_condition": condition,
                    "wmo_code": w_code,
                    "advisory": advisory,
                    "source": "live_open_meteo",
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            _ner_weather_cache = new_cache
            _ner_cache_timestamp = now_ts
            return _ner_weather_cache
    except Exception as e_om:
        last_err = f"Open-Meteo err: {e_om}"
        print(f"Open-Meteo batch fetch error: {e_om}")

    if _ner_weather_cache and not force_refresh:
        return _ner_weather_cache

    # --- 3. TERTIARY: Seasonal Fallback ---
    month = datetime.now(timezone.utc).month
    is_monsoon = 5 <= month <= 9
    fallback = {}
    for code, info in NER_STATE_STATIONS.items():
        fallback[code] = {
            "state_code": code,
            "state_name": info["name"],
            "station_name": info["station"],
            "temperature_c": 24.0,
            "relative_humidity_pct": 88,
            "rainfall_1h_mm": 1.5 if is_monsoon else 0.0,
            "rainfall_6h_mm": 8.0 if is_monsoon else 0.0,
            "rainfall_24h_mm": 22.0 if is_monsoon else 1.0,
            "rainfall_72h_mm": 45.0 if is_monsoon else 3.0,
            "wind_speed_kmh": 14.0,
            "wind_direction": "SW",
            "weather_condition": "Monsoon Showers" if is_monsoon else "Mainly Clear",
            "wmo_code": 61 if is_monsoon else 1,
            "advisory": "Advisory: Seasonal weather observation active across regional stations",
            "source": f"fallback ({last_err})",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    _ner_weather_cache = fallback
    _ner_cache_timestamp = now_ts - CACHE_TTL_SECONDS + 10
    return _ner_weather_cache


def fetch_imd_weather_for_coordinate(lat: float, lon: float) -> Tuple[float, float, float, float]:
    weather_map = fetch_all_ner_live_weather()
    nearest_code = "AS"
    min_dist_sq = 999999.0
    for code, info in NER_STATE_STATIONS.items():
        d_lat = lat - info["lat"]
        d_lon = lon - info["lon"]
        dist_sq = (d_lat * d_lat) + (d_lon * d_lon)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_code = code

    st_weather = weather_map.get(nearest_code, {})
    r1 = float(st_weather.get("rainfall_1h_mm", 0.0))
    r6 = float(st_weather.get("rainfall_6h_mm", 0.0))
    r24 = float(st_weather.get("rainfall_24h_mm", 0.0))
    r72 = float(st_weather.get("rainfall_72h_mm", 0.0))
    return r1, r6, r24, r72
