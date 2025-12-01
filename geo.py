"""
geo.py - Geocoding helpers (Google Geocoding preferred) + distance calculation + caching.

Usage:
- Set environment variable GOOGLE_API_KEY to use Google Geocoding.
- Optionally set FORCE_GOOGLE=1 to force Google-only (no Nominatim fallback).
- add_distance_column(df, user_place=..., place_columns=[...]) will geocode and add 'distance_km' column.

Notes:
- Respect Google API quotas & billing. Restrict your API key in GCP to reduce risk.
- Cache stored in geo_cache.json by default to avoid repeated network calls.
- For production consider a robust persistent cache (Redis / DB) and proper backoff.
"""
from __future__ import annotations

import os
import json
import tempfile
import math
import time
from typing import Optional, Tuple, Dict, Any, List
import requests
import pandas as pd

USER_AGENT = "IA-ML-Tinder-like/1.0 (contact: your-contact@example.com)"
DEFAULT_CACHE_PATH = "geo_cache.json"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FORCE_GOOGLE_ENV = os.getenv("FORCE_GOOGLE", "")  # set "1" to force Google-only


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    d = json.dumps(data, ensure_ascii=False, indent=2)
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dirn, delete=False, encoding="utf-8") as tmp:
        tmp.write(d)
        tmp.flush()
    try:
        os.replace(tmp.name, path)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            f.write(d)


# ---------------- Google Geocoding ----------------
def geocode_google(query: str, timeout: int = 8) -> Optional[Tuple[float, float]]:
    """
    Geocode a query using Google Geocoding API.
    Returns (lat, lon) or None.
    """
    if not GOOGLE_API_KEY:
        return None
    if not query or not query.strip():
        return None

    base = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": query, "key": GOOGLE_API_KEY}
    try:
        resp = requests.get(base, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        # handle common statuses
        if status == "OK":
            results = data.get("results") or []
            if not results:
                return None
            loc = results[0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
        # ZERO_RESULTS: not found; OVER_QUERY_LIMIT / REQUEST_DENIED: return None (caller may fallback)
        # we avoid raising to keep behavior controlled
        return None
    except Exception:
        return None


# ---------------- Nominatim fallback ----------------
def geocode_nominatim(query: str, timeout: int = 8, max_retries: int = 2) -> Optional[Tuple[float, float]]:
    """Geocode using Nominatim (OpenStreetMap). Note rate limits for public endpoint."""
    if not query or not query.strip():
        return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
    headers = {"User-Agent": USER_AGENT}
    attempt = 0
    while attempt <= max_retries:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            items = resp.json()
            if not items:
                return None
            item = items[0]
            return float(item["lat"]), float(item["lon"])
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(1 + attempt * 2)
                attempt += 1
                continue
            return None
        except Exception:
            return None
    return None


# ---------------- Unified geocode with force option ----------------
def geocode(query: str, force_google: Optional[bool] = None) -> Optional[Tuple[float, float]]:
    """
    Unified geocode function:
    - If force_google True or env FORCE_GOOGLE=1 -> try Google only (return None on failure)
    - Otherwise: try Google (if key present) and fallback to Nominatim
    """
    if not query or not query.strip():
        return None

    # determine force flag
    if force_google is None:
        force_google = bool(FORCE_GOOGLE_ENV and FORCE_GOOGLE_ENV.strip() != "")

    # Try Google if key present
    if GOOGLE_API_KEY:
        coords = geocode_google(query)
        if coords:
            return coords
        if force_google:
            # forced and Google failed => do not fallback
            return None

    # Fallback to Nominatim
    return geocode_nominatim(query)


# ---------------- Distance computation + DataFrame integration ----------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _guess_place_from_row(row: pd.Series, place_columns: List[str]) -> Optional[str]:
    """
    Build a textual place query from a DataFrame row using candidate columns.
    Attempts to unwrap list-like values and simple string encodings like "['Bruxelles']".
    """
    for col in place_columns:
        if col in row and row[col] not in (None, "", pd.NA):
            val = row[col]
            # If already lat/lon present, caller handles numeric columns separately
            # Handle list-like
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return ", ".join(map(str, val))
            # If it's a string that looks like a Python/list repr: "['bruxelles']" -> try to clean
            if isinstance(val, str):
                s = val.strip()
                # Basic cleanup for "['x']" or '["x"]'
                if s.startswith("[") and s.endswith("]"):
                    inner = s[1:-1].strip().strip("'\"")
                    if inner:
                        return inner
                return s
    # fallback: search columns that look like place fields
    for k, v in row.items():
        if v and any(substr in str(k).lower() for substr in ("lieu", "local", "commune", "ville", "adresse", "location", "place")):
            return str(v)
    return None


def add_distance_column(
    df: pd.DataFrame,
    user_place: Optional[str],
    place_columns: Optional[List[str]] = None,
    user_latlon: Optional[Tuple[float, float]] = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_google: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Add 'distance_km' column to df and return df sorted by ascending distance.
    - If force_google True (or env FORCE_GOOGLE=1), geocoding will not fallback to Nominatim.
    - Cache records geocoded strings -> [lat, lon] in cache_path JSON.
    """
    if place_columns is None:
        place_columns = [
            "Lieux de travail_clean",
            "Lieux de travail",
            "Lieux de travail_list",
            "lieuxtravaillocalite",
            "lieux de travail",
            "lieu",
            "commune",
            "localisation",
            "adresse",
            "ville",
            "fields.lieuxtravaillocalite",
        ]

    # load cache
    cache: Dict[str, Tuple[float, float]] = {}
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    cache[k] = (float(v[0]), float(v[1]))
    except Exception:
        cache = {}

    df = df.copy()

    # geocode user if needed
    if user_latlon is None and user_place:
        up = user_place.strip()
        if up in cache:
            user_latlon = cache[up]
        else:
            coords = geocode(up, force_google=force_google)
            if coords:
                cache[up] = coords
                user_latlon = coords

    if not user_latlon:
        df["distance_km"] = pd.NA
        return df

    user_lat, user_lon = user_latlon

    distances: List[Any] = []
    lat_keys = ["latitude", "lat", "y", "coord_lat", "latitud"]
    lon_keys = ["longitude", "lon", "lng", "x", "coord_lon", "longitud"]

    for _, row in df.iterrows():
        lat = None
        lon = None
        for k in lat_keys:
            if k in row and row[k] not in (None, "", pd.NA):
                try:
                    lat = float(row[k])
                    break
                except Exception:
                    lat = None
        for k in lon_keys:
            if k in row and row[k] not in (None, "", pd.NA):
                try:
                    lon = float(row[k])
                    break
                except Exception:
                    lon = None

        if lat is not None and lon is not None:
            try:
                d = haversine_km(user_lat, user_lon, lat, lon)
                distances.append(round(d, 2))
                continue
            except Exception:
                distances.append(pd.NA)
                continue

        place_text = _guess_place_from_row(row, place_columns)
        if not place_text:
            distances.append(pd.NA)
            continue

        pt = place_text.strip()
        if pt in cache:
            latlon = cache[pt]
        else:
            latlon = geocode(pt, force_google=force_google)
            # minor fallback bias: try adding country to help geocoders
            if not latlon and not force_google:
                latlon = geocode(f"{pt}, Belgium", force_google=False)
            if latlon:
                cache[pt] = latlon

        if not latlon:
            distances.append(pd.NA)
            continue

        lat2, lon2 = latlon
        try:
            d = haversine_km(user_lat, user_lon, lat2, lon2)
            distances.append(round(d, 2))
        except Exception:
            distances.append(pd.NA)

    df["distance_km"] = distances

    # persist cache
    try:
        raw_to_save = {k: [v[0], v[1]] for k, v in cache.items()}
        _atomic_write_json(cache_path, raw_to_save)
    except Exception:
        pass

    df_sorted = df.assign(_dist_num=df["distance_km"].apply(lambda x: float(x) if pd.notna(x) else float("inf"))).sort_values(
        "_dist_num"
    ).drop(columns="_dist_num")
    return df_sorted.reset_index(drop=True)