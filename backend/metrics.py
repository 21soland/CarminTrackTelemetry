"""
Metrics Computation for GPS Track Analysis

This module computes derived metrics from raw GPS/IMU data, including
coordinates, distances, speeds, accelerations, and headings.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from . import utils


def latlon_to_xy(lat_series: pd.Series, lon_series: pd.Series, 
                 ref_lat_rad: float, ref_lon_rad: float) -> Tuple[pd.Series, pd.Series]:
    """
    Convert latitude/longitude to local Cartesian coordinates (x, y).
    
    Uses a simple equirectangular projection approximation, suitable for
    small tracks where Earth's curvature can be approximated as flat.
    
    Args:
        lat_series: Series of latitude values in degrees.
        lon_series: Series of longitude values in degrees.
        ref_lat_rad: Reference latitude in radians (typically first valid point).
        ref_lon_rad: Reference longitude in radians (typically first valid point).
        
    Returns:
        Tuple of (x_m, y_m) Series in meters, where x is east and y is north.
    """
    R = 6378137.0  # Earth radius in meters (WGS84)
    lat_rad = np.deg2rad(lat_series)
    lon_rad = np.deg2rad(lon_series)
    
    x = (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad) * R
    y = (lat_rad - ref_lat_rad) * R
    
    return x, y


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Uses the Haversine formula to compute distance along the surface of a sphere.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees.
        lat2, lon2: Latitude and longitude of second point in degrees.
        
    Returns:
        Distance in meters between the two points.
    """
    R = 6371000.0  # Earth radius in meters
    lat1_rad, lon1_rad = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2_rad, lon2_rad = np.deg2rad(lat2), np.deg2rad(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived metrics from raw GPS/IMU data.
    
    Calculates:
    - Elapsed time from start
    - Local Cartesian coordinates (x, y)
    - Segment distances and cumulative distance
    - Speed (from GPS or computed from position changes)
    - Heading (direction of travel)
    - Longitudinal acceleration (forward/backward)
    - Lateral acceleration (sideways, from turning)
    
    Args:
        df: DataFrame with raw GPS/IMU data from extract_time_series().
        
    Returns:
        DataFrame with additional computed columns: elapsed_s, x_m, y_m,
        segment_distance_m, distance_along_m, speed_mps, speed_mph,
        heading_deg, long_accel_mps2, lat_accel_mps2.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Compute elapsed time from first timestamp
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    
    # Get reference point for local coordinate system
    valid_lat = df["lat"].dropna()
    valid_lon = df["lon"].dropna()
    if valid_lat.empty or valid_lon.empty:
        return df
    
    ref_lat = np.deg2rad(valid_lat.iloc[0])
    ref_lon = np.deg2rad(valid_lon.iloc[0])
    
    # Convert to local Cartesian coordinates
    x_m, y_m = latlon_to_xy(df["lat"], df["lon"], ref_lat, ref_lon)
    df["x_m"] = x_m
    df["y_m"] = y_m
    
    # Compute segment distances and cumulative distance
    dx = df["x_m"].diff()
    dy = df["y_m"].diff()
    dt = df["elapsed_s"].diff().replace({0: np.nan})
    segment_dist = np.sqrt(dx**2 + dy**2)
    df["segment_distance_m"] = segment_dist.fillna(0)
    df["distance_along_m"] = df["segment_distance_m"].cumsum().fillna(0)
    
    # Compute speed (fill missing GPS speed with computed speed)
    speed_course = segment_dist / dt
    df["speed_mps"] = df["speed_mps"].fillna(speed_course)
    df["speed_mps"] = df["speed_mps"].ffill().fillna(0)
    df["speed_mph"] = df["speed_mps"] * 2.23694  # Convert m/s to mph
    
    # Compute heading (direction of travel)
    heading = np.arctan2(dx.fillna(0), dy.fillna(0))
    heading = np.unwrap(heading)  # Handle 360° wraparound
    df["heading_deg"] = np.rad2deg(heading)
    
    # Compute longitudinal acceleration (forward/backward)
    long_accel = df["speed_mps"].diff() / dt
    df["long_accel_mps2"] = long_accel.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Compute lateral acceleration (sideways, from turning)
    if len(df) > 1:
        heading_rate = np.gradient(heading, df["elapsed_s"])
        df["lat_accel_mps2"] = df["speed_mps"] * heading_rate
    else:
        df["lat_accel_mps2"] = 0
    
    df["lat_accel_mps2"] = df["lat_accel_mps2"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

