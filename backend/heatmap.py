"""
Heatmap Generation for GPS Track Analysis

This module generates heatmap data for brake, throttle, and lateral acceleration
visualizations by normalizing intensity values across the session.
"""

from typing import Dict, List
from . import utils


def build_heatmap_layers(telemetry: List[Dict]) -> Dict:
    """
    Build heatmap data for brake, throttle, and lateral acceleration.
    
    Creates point data with intensity values normalized to the maximum
    observed value for each metric. Used for heatmap visualization.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        
    Returns:
        Dictionary with three keys:
        - "brake": List of points with negative longitudinal acceleration
        - "throttle": List of points with positive longitudinal acceleration
        - "lateral": List of points with lateral acceleration
        Each point contains lat, lon, intensity (0-1), and value.
    """
    # Find maximum values for normalization
    max_brake = 0.0
    max_throttle = 0.0
    max_lat = 0.0
    
    for sample in telemetry:
        long_accel = sample.get("long_accel_mps2") or 0.0
        lat_accel = sample.get("lat_accel_mps2") or 0.0
        
        if long_accel < 0:
            max_brake = max(max_brake, abs(long_accel))
        else:
            max_throttle = max(max_throttle, long_accel)
        
        max_lat = max(max_lat, abs(lat_accel))
    
    def build_points(filter_fn, max_value):
        """Helper to build heatmap points for a given metric."""
        points = []
        if max_value <= 0:
            return points
        
        for sample in telemetry:
            lat = sample.get("lat")
            lon = sample.get("lon")
            value = filter_fn(sample)
            
            if lat is None or lon is None or value == 0:
                continue
            
            intensity = min(abs(value) / max_value, 1.0)
            points.append({
                "lat": lat,
                "lon": lon,
                "intensity": utils.round_float(intensity, 3),
                "value": utils.round_float(value, 3),
            })
        
        return points
    
    return {
        "brake": build_points(lambda s: min(s.get("long_accel_mps2") or 0.0, 0.0), max_brake),
        "throttle": build_points(lambda s: max(s.get("long_accel_mps2") or 0.0, 0.0), max_throttle),
        "lateral": build_points(lambda s: s.get("lat_accel_mps2") or 0.0, max_lat),
    }

