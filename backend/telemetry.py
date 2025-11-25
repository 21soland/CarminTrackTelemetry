"""
Telemetry and GeoJSON Conversion for GPS Track Analysis

This module converts computed metrics into telemetry records and GeoJSON
formats suitable for visualization and API responses.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from . import utils


def build_telemetry_records(df: pd.DataFrame) -> List[Dict]:
    """
    Convert DataFrame to list of telemetry record dictionaries.
    
    Formats all computed metrics into a standardized telemetry format suitable
    for JSON serialization and frontend consumption.
    
    Args:
        df: DataFrame with computed metrics from compute_derived_metrics().
        
    Returns:
        List of dictionaries, each representing one telemetry sample with
        timestamp, position, speed, acceleration, IMU data, etc.
    """
    records = []
    
    for row in df.itertuples():
        record = {
            "timestamp": row.timestamp.isoformat(),
            "elapsed_s": utils.round_float(row.elapsed_s),
            "lat": utils.preserve_precision(row.lat),
            "lon": utils.preserve_precision(row.lon),
            "alt": utils.preserve_precision(row.alt, digits=2),
            "speed_mps": utils.round_float(row.speed_mps),
            "speed_mph": utils.round_float(row.speed_mph),
            "long_accel_mps2": utils.round_float(row.long_accel_mps2),
            "lat_accel_mps2": utils.round_float(row.lat_accel_mps2),
            "distance_m": utils.preserve_precision(row.distance_along_m),
            "heading_deg": utils.round_float(row.heading_deg, digits=2),
            "imu_accel_x": utils.round_float(getattr(row, "imu_accel_x", np.nan)),
            "imu_accel_y": utils.round_float(getattr(row, "imu_accel_y", np.nan)),
            "imu_accel_z": utils.round_float(getattr(row, "imu_accel_z", np.nan)),
            "imu_gyro_z": utils.round_float(getattr(row, "imu_gyro_z", np.nan)),
            "pos_accuracy_m": utils.round_float(getattr(row, "pos_accuracy_m", np.nan)),
        }
        records.append(record)
    
    return records


def telemetry_to_geojson(telemetry: List[Dict]) -> Dict:
    """
    Convert telemetry records to GeoJSON FeatureCollection.
    
    Creates a LineString feature representing the track path and a Point
    feature marking the start/finish line.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        
    Returns:
        GeoJSON FeatureCollection with:
        - LineString feature: the complete track path
        - Point feature: start/finish marker
        
    Raises:
        ValueError: If no valid coordinates are found in telemetry.
    """
    coordinates = [
        [sample["lon"], sample["lat"]]
        for sample in telemetry
        if sample["lat"] is not None and sample["lon"] is not None
    ]
    
    if not coordinates:
        raise ValueError("No valid coordinates were generated from telemetry.")
    
    line_feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {
            "sampleCount": len(coordinates),
        },
    }
    
    start_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coordinates[0],
        },
        "properties": {"marker": "start_finish"},
    }
    
    return {
        "type": "FeatureCollection",
        "features": [line_feature, start_feature],
    }

