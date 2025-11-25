"""
GPS Track Analysis Module

This module processes raw GPS and IMU data from track sessions, extracting
telemetry, detecting laps, identifying corners, and generating visualizations.

This file now serves as a compatibility layer that imports and re-exports
functions from the modular structure. All functionality has been split into
separate modules for better organization.
"""

# Import constants
from .constants import DATA_DIR, HEADER_FILE, DEFAULT_DATA_FILE

# Import utility functions
from .utils import (
    safe_float,
    mean_or_nan,
    round_float,
    preserve_precision,
)

# Import data loading functions
from .data_loading import (
    get_headers,
    get_header,
    parse_timestep_data,
    load_gps_data,
)

# Import time series functions
from .time_series import (
    extract_timestamp_ms,
    extract_fix_fields,
    extract_time_series,
)

# Import metrics functions
from .metrics import (
    latlon_to_xy,
    haversine_m,
    compute_derived_metrics,
)

# Import telemetry functions
from .telemetry import (
    build_telemetry_records,
    telemetry_to_geojson,
)

# Import lap analysis functions
from .lap_analysis import (
    detect_laps,
    build_fallback_lap,
    build_lap_record,
    compute_sector_times,
    annotate_lap_samples,
    build_lap_features,
    speed_band_color,
    build_lap_delta_traces,
)

# Import corner detection functions
from .corner_detection import (
    detect_corners,
)

# Import heatmap functions
from .heatmap import (
    build_heatmap_layers,
)

# Import export functions
from .export import (
    export_lap_csv,
)

# Import session builder functions
from .session import (
    build_session_payload,
    get_raw_trajectory,
)

# Re-export everything for backwards compatibility
__all__ = [
    # Constants
    "DATA_DIR",
    "HEADER_FILE",
    "DEFAULT_DATA_FILE",
    # Utilities
    "safe_float",
    "mean_or_nan",
    "round_float",
    "preserve_precision",
    # Data loading
    "get_headers",
    "get_header",
    "parse_timestep_data",
    "load_gps_data",
    # Time series
    "extract_timestamp_ms",
    "extract_fix_fields",
    "extract_time_series",
    # Metrics
    "latlon_to_xy",
    "haversine_m",
    "compute_derived_metrics",
    # Telemetry
    "build_telemetry_records",
    "telemetry_to_geojson",
    # Lap analysis
    "detect_laps",
    "build_fallback_lap",
    "build_lap_record",
    "compute_sector_times",
    "annotate_lap_samples",
    "build_lap_features",
    "speed_band_color",
    "build_lap_delta_traces",
    # Corner detection
    "detect_corners",
    # Heatmap
    "build_heatmap_layers",
    # Export
    "export_lap_csv",
    # Session builder
    "build_session_payload",
    "get_raw_trajectory",
]
