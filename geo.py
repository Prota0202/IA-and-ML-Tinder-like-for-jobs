"""
Geocoding and distance utilities for sorting job offers by proximity.

This module provides:
- Geocoding via Google Geocoding API (requires GOOGLE_API_KEY env var)
- Fallback geocoding via OpenStreetMap Nominatim (public, rate-limited)
- Haversine distance calculation
- add_distance_column() to geocode and sort a DataFrame by distance

USAGE:
    Set GOOGLE_API_KEY environment variable for Google Geocoding (preferred).
    If not set, Nominatim (OpenStreetMap) is used as fallback.

    from geo import add_distance_column
    df_sorted = add_distance_column(offers_df, user_place="Bruxelles, 1000")

CACHE:
    Geocoded results are cached to geo_cache.json to avoid repeated API calls.
    The cache file is created automatically if missing.

CAUTION:
    - Nominatim has strict usage limits (1 req/sec, max ~1000/day).
    - Google Geocoding API usage is billed; set quotas in your GCP console.
    - Use responsibly and cache results to minimize API calls.
"""

import os
import json
import math
import time
from typing import Optional, Tuple, List

import requests
import pandas as pd

# User-Agent for Nominatim (required by OSM usage policy)
USER_AGENT = "JobMatcherApp/1.0 (github.com/Prota0202/IA-and-ML-Tinder-like-for-jobs)"

# Timeouts for HTTP requests (seconds)
REQUEST_TIMEOUT = 10

# Nominatim rate limit: 1 request per second (add small buffer)
NOMINATIM_RATE_LIMIT_SECONDS = 1.1


def geocode_google(query: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a location using Google Geocoding API.

    Requires GOOGLE_API_KEY environment variable to be set.

    Args:
        query: Address or place name to geocode.

    Returns:
        (latitude, longitude) tuple, or None if geocoding fails.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": query,
        "key": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "OK" and data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return (location["lat"], location["lng"])
        return None
    except requests.RequestException:
        return None


def geocode_nominatim(query: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a location using OpenStreetMap Nominatim (public API).

    CAUTION: Nominatim has strict usage limits (1 request/second).

    Args:
        query: Address or place name to geocode.

    Returns:
        (latitude, longitude) tuple, or None if geocoding fails.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": USER_AGENT,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data and len(data) > 0:
            return (float(data[0]["lat"]), float(data[0]["lon"]))
        return None
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None


def geocode(query: str) -> Optional[Tuple[float, float]]:
    """
    Unified geocoding function.

    Uses Google Geocoding API if GOOGLE_API_KEY is set,
    otherwise falls back to Nominatim.

    Args:
        query: Address or place name to geocode.

    Returns:
        (latitude, longitude) tuple, or None if geocoding fails.
    """
    if os.getenv("GOOGLE_API_KEY"):
        result = geocode_google(query)
        if result:
            return result
    # Fallback to Nominatim
    return geocode_nominatim(query)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Coordinates of point 1 (degrees).
        lat2, lon2: Coordinates of point 2 (degrees).

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def _load_cache(cache_path: str) -> dict:
    """Load geocache from JSON file, return empty dict if missing/corrupt."""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict, cache_path: str) -> None:
    """Save geocache to JSON file atomically."""
    tmp_path = cache_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, cache_path)
    except OSError:
        # If saving fails, continue without error
        pass


def _get_cached_or_geocode(query: str, cache: dict, use_nominatim_throttle: bool = True) -> Optional[Tuple[float, float]]:
    """
    Get coordinates from cache or geocode and cache result.

    Args:
        query: Location string to geocode.
        cache: Cache dictionary (will be mutated).
        use_nominatim_throttle: If True, add delay for Nominatim rate-limiting.

    Returns:
        (lat, lon) tuple or None.
    """
    if not query or not query.strip():
        return None

    query_key = query.strip().lower()

    if query_key in cache:
        cached = cache[query_key]
        if cached is None:
            return None
        return tuple(cached)

    # Geocode
    result = geocode(query)

    # Throttle for Nominatim (no Google key)
    if use_nominatim_throttle and not os.getenv("GOOGLE_API_KEY"):
        time.sleep(NOMINATIM_RATE_LIMIT_SECONDS)

    # Cache result (None if failed)
    cache[query_key] = list(result) if result else None

    return result


def add_distance_column(
    df: pd.DataFrame,
    user_place: Optional[str] = None,
    place_columns: Optional[List[str]] = None,
    user_latlon: Optional[Tuple[float, float]] = None,
    cache_path: str = "geo_cache.json",
) -> pd.DataFrame:
    """
    Add distance_km column to DataFrame and sort by distance.

    Geocodes user location and offer locations, computes haversine distance,
    and returns DataFrame sorted by distance (NaN distances at end).

    Args:
        df: DataFrame with job offers.
        user_place: User's location string (e.g., "Bruxelles" or "Bruxelles, 1000").
        place_columns: List of column names to build offer location string from.
                       Default: ["lieuxtravaillocalite", "lieuxtravailcodepostal"]
        user_latlon: Optional (lat, lon) tuple for user; skips geocoding if provided.
        cache_path: Path to JSON cache file.

    Returns:
        DataFrame with distance_km column, sorted by distance (NaN at end).
    """
    if df.empty:
        df = df.copy()
        df["distance_km"] = pd.Series(dtype=float)
        return df

    if place_columns is None:
        place_columns = ["lieuxtravaillocalite", "lieuxtravailcodepostal"]

    # Load cache
    cache = _load_cache(cache_path)
    cache_modified = False

    # Geocode user location
    if user_latlon is None and user_place:
        user_latlon = _get_cached_or_geocode(user_place, cache)
        cache_modified = True

    if user_latlon is None:
        # Cannot compute distances without user location
        df = df.copy()
        df["distance_km"] = float("nan")
        return df

    user_lat, user_lon = user_latlon

    # Compute distances for each row
    distances = []
    for _, row in df.iterrows():
        offer_latlon = None

        # Check if lat/lon columns exist in the row
        if "lat" in row and "lon" in row:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                if not (math.isnan(lat) or math.isnan(lon)):
                    offer_latlon = (lat, lon)
            except (ValueError, TypeError):
                pass

        # If no lat/lon, try to build place string from place_columns
        if offer_latlon is None:
            place_parts = []
            for col in place_columns:
                if col in row:
                    val = row[col]
                    # Handle list columns (e.g., lieuxtravaillocalite)
                    if isinstance(val, list):
                        place_parts.extend([str(v) for v in val if v is not None and str(v).strip()])
                    elif pd.notna(val) and str(val).strip():
                        place_parts.append(str(val))

            if place_parts:
                offer_place = ", ".join(place_parts)
                cached_before = len(cache)
                offer_latlon = _get_cached_or_geocode(offer_place, cache)
                if len(cache) > cached_before:
                    cache_modified = True

        # Calculate distance
        if offer_latlon:
            dist = haversine_km(user_lat, user_lon, offer_latlon[0], offer_latlon[1])
            distances.append(dist)
        else:
            distances.append(float("nan"))

    # Save cache if modified
    if cache_modified:
        _save_cache(cache, cache_path)

    # Add distance column and sort
    df = df.copy()
    df["distance_km"] = distances

    # Sort: valid distances first (ascending), NaN at end
    df = df.sort_values(by="distance_km", na_position="last")
    df = df.reset_index(drop=True)

    return df
