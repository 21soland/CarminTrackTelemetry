"""
Time Series Extraction for GPS Track Analysis

This module extracts time-indexed data from parsed GPS structures, converting
nested dictionaries into flat DataFrames suitable for analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from . import utils


def extract_timestamp_ms(timestep: Dict) -> Optional[float]:
    """
    Extract timestamp in milliseconds from a time step.
    
    Prefers GNSS fix timestamp if available, otherwise falls back to Raw record.
    
    Args:
        timestep: Parsed time step dictionary.
        
    Returns:
        Timestamp in milliseconds as a float, or None if not found.
    """
    # Prefer the GNSS fix timestamp if present
    fix_entries = timestep.get("Fix", {})
    if fix_entries:
        first_fix = fix_entries.get(0, {})
        ts = first_fix.get("UnixTimeMillis")
        if ts:
            return float(ts)
    
    # Fall back to Raw record timestamp
    raw_entries = timestep.get("Raw", {})
    if raw_entries:
        raw = raw_entries.get(0, {})
        ts = raw.get("utcTimeMillis")
        if ts:
            return float(ts)
    
    return None


def extract_fix_fields(fix_entries: Dict[int, Dict]) -> Dict[str, float]:
    """
    Extract and average GNSS fix fields from multiple fix entries.
    
    If multiple fix entries exist in a time step, computes the mean of each
    field (latitude, longitude, altitude, speed, accuracy).
    
    Args:
        fix_entries: Dictionary of fix entries from a time step.
        
    Returns:
        Dictionary with averaged fields: lat, lon, alt, speed_mps, pos_accuracy_m.
    """
    lats, lons, alts, speeds = [], [], [], []
    accuracies = []
    
    for fix in fix_entries.values():
        lats.append(utils.safe_float(fix.get("LatitudeDegrees")))
        lons.append(utils.safe_float(fix.get("LongitudeDegrees")))
        alts.append(utils.safe_float(fix.get("AltitudeMeters")))
        speeds.append(utils.safe_float(fix.get("SpeedMps")))
        accuracies.append(utils.safe_float(fix.get("AccuracyMeters")))
    
    return {
        "lat": utils.mean_or_nan(lats),
        "lon": utils.mean_or_nan(lons),
        "alt": utils.mean_or_nan(alts),
        "speed_mps": utils.mean_or_nan(speeds),
        "pos_accuracy_m": utils.mean_or_nan(accuracies),
    }


def extract_time_series(parsed_data: Dict[int, Dict]) -> pd.DataFrame:
    """
    Flatten nested GNSS/IMU structure into a time-indexed DataFrame.
    
    Converts the hierarchical time step structure into a flat table with one
    row per time step, extracting GNSS fix data and IMU (accelerometer/gyro) data.
    
    Args:
        parsed_data: Dictionary of parsed time steps from load_gps_data().
        
    Returns:
        DataFrame with columns: index, timestamp_ms, timestamp, lat, lon, alt,
        speed_mps, pos_accuracy_m, imu_accel_x/y/z, imu_gyro_x/y/z.
        Sorted by timestamp and with invalid timestamps removed.
    """
    rows = []
    
    for idx in sorted(parsed_data.keys()):
        timestep = parsed_data[idx]
        row = {
            "index": idx,
            "timestamp_ms": extract_timestamp_ms(timestep),
        }
        
        # Extract GNSS Fix data
        fix_entries = timestep.get("Fix", {})
        if fix_entries:
            row.update(extract_fix_fields(fix_entries))
        
        # Extract IMU Accelerometer data (prefer calibrated, fall back to uncalibrated)
        accel_entries = timestep.get("Accel") or timestep.get("UncalAccel")
        if accel_entries:
            accel = accel_entries[0]
            row["imu_accel_x"] = utils.safe_float(accel.get("AccelXMps2") or accel.get("UncalAccelXMps2"))
            row["imu_accel_y"] = utils.safe_float(accel.get("AccelYMps2") or accel.get("UncalAccelYMps2"))
            row["imu_accel_z"] = utils.safe_float(accel.get("AccelZMps2") or accel.get("UncalAccelZMps2"))
        
        # Extract IMU Gyroscope data (prefer calibrated, fall back to uncalibrated)
        gyro_entries = timestep.get("Gyro") or timestep.get("UncalGyro")
        if gyro_entries:
            gyro = gyro_entries[0]
            row["imu_gyro_x"] = utils.safe_float(gyro.get("GyroXRadPerSec") or gyro.get("UncalGyroXRadPerSec"))
            row["imu_gyro_y"] = utils.safe_float(gyro.get("GyroYRadPerSec") or gyro.get("UncalGyroYRadPerSec"))
            row["imu_gyro_z"] = utils.safe_float(gyro.get("GyroZRadPerSec") or gyro.get("UncalGyroZRadPerSec"))
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Clean and sort by timestamp
    df = df.dropna(subset=["timestamp_ms"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

