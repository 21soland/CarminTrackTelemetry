"""
Export Functions for GPS Track Analysis

This module provides functions to export session and lap data to various
formats (CSV, JSON) for external analysis or backup.
"""

import csv
import io
from typing import Dict
from . import constants


def export_lap_csv(session: Dict, lap_number: int) -> str:
    """
    Export a single lap's telemetry data to CSV format.
    
    Args:
        session: Session dictionary from build_session_payload().
        lap_number: Lap number to export (1-indexed).
        
    Returns:
        CSV string with lap telemetry data.
        
    Raises:
        ValueError: If lap_number is not found or has no sample range.
    """
    laps = session.get("laps", [])
    telemetry = session.get("telemetry", [])
    
    lap = next((lap for lap in laps if lap["lap_number"] == lap_number), None)
    if not lap:
        raise ValueError(f"Lap {lap_number} not found")
    
    start_idx = lap["start_sample_idx"]
    end_idx = lap["end_sample_idx"]
    
    if start_idx is None or end_idx is None:
        raise ValueError(f"Lap {lap_number} has no sample range")
    
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    # Write header
    writer.writerow([
        "timestamp",
        "lap_number",
        "lap_elapsed_s",
        "lap_distance_m",
        "lat",
        "lon",
        "speed_mph",
        "long_accel_mps2",
        "lat_accel_mps2",
        "imu_accel_x",
        "imu_accel_y",
        "imu_accel_z",
    ])
    
    # Write data rows
    for sample in telemetry[start_idx : end_idx + 1]:
        writer.writerow([
            sample.get("timestamp"),
            sample.get("lap_index"),
            sample.get("lap_elapsed_s"),
            sample.get("lap_distance_m"),
            sample.get("lat"),
            sample.get("lon"),
            sample.get("speed_mph"),
            sample.get("long_accel_mps2"),
            sample.get("lat_accel_mps2"),
            sample.get("imu_accel_x"),
            sample.get("imu_accel_y"),
            sample.get("imu_accel_z"),
        ])
    
    return buffer.getvalue()

