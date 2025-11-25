"""
Session Builder for GPS Track Analysis

This module orchestrates the complete analysis pipeline, combining all
processing steps to build a complete session payload with all analysis results.
"""

from pathlib import Path
from typing import Dict
from . import constants
from . import data_loading
from . import time_series
from . import metrics
from . import telemetry
from . import lap_analysis
from . import corner_detection
from . import heatmap


def build_session_payload(data_file: Path = constants.DEFAULT_DATA_FILE) -> Dict:
    """
    Build complete session payload with all analysis results.
    
    Main entry point that orchestrates the entire analysis pipeline:
    1. Loads and parses GPS data
    2. Extracts time series
    3. Computes derived metrics
    4. Builds telemetry records
    5. Detects laps
    6. Generates GeoJSON features
    7. Detects corners
    8. Builds heatmap layers
    
    Args:
        data_file: Path to the GPS data file. Defaults to DEFAULT_DATA_FILE.
        
    Returns:
        Dictionary containing:
        - track: GeoJSON FeatureCollection of the track
        - telemetry: List of telemetry records
        - laps: List of lap records
        - lap_features: GeoJSON features for each lap
        - lap_deltas: Time delta traces comparing laps
        - corners: List of detected corners
        - heatmap: Heatmap data for brake/throttle/lateral
        
    Raises:
        ValueError: If parsed time-series is empty.
    """
    gps_data = data_loading.load_gps_data(data_file)
    df = time_series.extract_time_series(gps_data)
    
    if df.empty:
        raise ValueError("Parsed time-series is empty. Check data file.")
    
    df = metrics.compute_derived_metrics(df)
    telemetry_records = telemetry.build_telemetry_records(df)
    track_geojson = telemetry.telemetry_to_geojson(telemetry_records)
    
    # Detect laps
    laps = lap_analysis.detect_laps(telemetry_records)
    if not laps:
        fallback = lap_analysis.build_fallback_lap(telemetry_records)
        laps = [fallback] if fallback else []
    
    # Annotate samples with lap information
    lap_analysis.annotate_lap_samples(telemetry_records, laps)
    
    # Build lap visualizations
    lap_features = lap_analysis.build_lap_features(telemetry_records, laps)
    lap_deltas = lap_analysis.build_lap_delta_traces(telemetry_records, laps)
    
    # Detect corners and build heatmaps
    corners = corner_detection.detect_corners(telemetry_records)
    heatmap_data = heatmap.build_heatmap_layers(telemetry_records)
    
    return {
        "track": track_geojson,
        "telemetry": telemetry_records,
        "laps": laps,
        "lap_features": lap_features,
        "lap_deltas": lap_deltas,
        "corners": corners,
        "heatmap": heatmap_data,
    }


def get_raw_trajectory() -> Dict:
    """
    Backwards compatibility helper for legacy code.
    
    Returns only the LineString feature from the track GeoJSON, for code
    that expects the old format.
    
    Returns:
        GeoJSON Feature (LineString) representing the track path.
    """
    session = build_session_payload()
    return session["track"]["features"][0]

