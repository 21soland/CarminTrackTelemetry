"""
Corner Detection for GPS Track Analysis

This module detects corners by identifying segments with high lateral acceleration
and computes corner metrics such as entry/exit speeds, apex, and G-forces.
"""

import numpy as np
from typing import Dict, List
from . import utils


def detect_corners(telemetry: List[Dict], lat_accel_threshold: float = 1.6,
                   min_samples: int = 5) -> List[Dict]:
    """
    Detect corners by identifying segments with high lateral acceleration.
    
    Finds continuous segments where lateral acceleration exceeds the threshold,
    identifying them as corners. For each corner, computes entry/exit speeds,
    minimum speed (apex), maximum lateral G-force, and duration.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        lat_accel_threshold: Minimum lateral acceleration (m/s²) to consider
                            a corner. Default 1.6.
        min_samples: Minimum number of consecutive samples required to
                    constitute a corner. Default 5.
        
    Returns:
        List of corner dictionaries, each containing corner_id, lap_number,
        entry_speed_mph, exit_speed_mph, min_speed_mph, max_lat_g, duration_s,
        geometry (list of [lon, lat] coordinates), and apex (lat, lon).
    """
    corners = []
    idx = 0
    
    while idx < len(telemetry):
        sample = telemetry[idx]
        lat_accel = abs(sample.get("lat_accel_mps2") or 0.0)
        
        if lat_accel < lat_accel_threshold:
            idx += 1
            continue
        
        # Found start of corner - find end
        start_idx = idx
        while idx < len(telemetry) and abs(telemetry[idx].get("lat_accel_mps2") or 0.0) >= lat_accel_threshold:
            idx += 1
        
        end_idx = min(idx - 1, len(telemetry) - 1)
        
        # Check minimum length requirement
        if end_idx - start_idx + 1 < min_samples:
            continue
        
        # Extract corner segment
        segment = telemetry[start_idx : end_idx + 1]
        speeds = [sample.get("speed_mph") or 0.0 for sample in segment]
        lat_accels = [abs(sample.get("lat_accel_mps2") or 0.0) for sample in segment]
        lap_number = segment[0].get("lap_index")
        
        # Find apex (point of minimum speed)
        apex_idx = int(np.argmin(speeds))
        apex_sample = segment[apex_idx]
        
        corners.append({
            "corner_id": len(corners) + 1,
            "lap_number": lap_number,
            "entry_speed_mph": utils.round_float(speeds[0], 1),
            "exit_speed_mph": utils.round_float(speeds[-1], 1),
            "min_speed_mph": utils.round_float(min(speeds), 1),
            "max_lat_g": utils.round_float(max(lat_accels) / 9.80665, 3),  # Convert m/s² to G
            "duration_s": utils.round_float(
                (segment[-1].get("elapsed_s") or 0.0) - (segment[0].get("elapsed_s") or 0.0),
                3,
            ),
            "geometry": [
                [sample["lon"], sample["lat"]]
                for sample in segment
                if sample.get("lat") is not None and sample.get("lon") is not None
            ],
            "apex": {
                "lat": apex_sample.get("lat"),
                "lon": apex_sample.get("lon"),
            },
        })
    
    return corners

