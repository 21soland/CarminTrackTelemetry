"""
GPS Track Analysis Module

This module processes raw GPS and IMU data from track sessions, extracting
telemetry, detecting laps, identifying corners, and generating visualizations.
"""

import csv
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR = Path(__file__).parent / "GPS Data"
HEADER_FILE = DATA_DIR / "headers.txt"
DEFAULT_DATA_FILE = DATA_DIR / "Parsable Data.txt"


# ============================================================================
# DATA LOADING & PARSING
# ============================================================================

def get_headers(header_file: Path = HEADER_FILE) -> Dict[str, List[str]]:
    """
    Load and parse the header definitions file.
    
    Reads a file where each line contains a header type followed by comma-separated
    field names. Returns a dictionary mapping header types to their field lists.
    
    Args:
        header_file: Path to the headers definition file. Defaults to HEADER_FILE.
        
    Returns:
        Dictionary mapping header type names (e.g., "Fix", "Accel") to lists
        of field names for that header type.
    """
    headers_dict = {}
    
    with header_file.open("r", encoding="utf-8") as file:
        headers = file.readlines()
    
    for header in headers:
        header = header.strip()
        header_list = header.split(',')
        headers_dict[header_list[0]] = header_list[1:]
    
    return headers_dict


def get_header(line: str, headers: List[str]) -> Optional[str]:
    """
    Identify which header type a line belongs to.
    
    Checks if a line starts with any of the known header types.
    
    Args:
        line: The line of text to check.
        headers: List of known header type names.
        
    Returns:
        The matching header type name, or None if no match is found.
    """
    for header in headers:
        if line.startswith(header):
            return header
    return None


def parse_timestep_data(time_step: List[str], headers_dict: Dict[str, List[str]]) -> Dict:
    """
    Parse a single time step's data into a structured dictionary.
    
    Processes all lines in a time step, grouping data by header type and
    handling multiple entries of the same header type.
    
    Args:
        time_step: List of lines belonging to a single time step.
        headers_dict: Dictionary mapping header types to their field names.
        
    Returns:
        Nested dictionary structure: {header_type: {entry_index: {field: value}}}
        Example: {"Fix": {0: {"LatitudeDegrees": "40.123", ...}}, ...}
    """
    headers = list(headers_dict.keys())
    time_step_dict = {}
    
    for line in time_step:
        header_type = get_header(line, headers)
        if not header_type:
            continue
            
        split_line = line.rstrip().split(",")
        
        # Check if there are multiple entries of the same header type
        n = 0
        if header_type in time_step_dict:
            n = len(time_step_dict[header_type].keys())
        else:
            time_step_dict[header_type] = {}
        
        # Add the data for this entry
        time_step_dict[header_type][n] = {}
        
        header_type_keys = headers_dict[header_type]
        for i in range(1, len(header_type_keys)):
            # Note: i-1 because split_line[0] is the header type itself
            time_step_dict[header_type][n][header_type_keys[i-1]] = split_line[i]
    
    return time_step_dict


def load_gps_data(file_path: Path = DEFAULT_DATA_FILE) -> Dict[int, Dict]:
    """
    Load and parse GPS data file into a time-indexed structure.
    
    Splits the raw data file into time steps (delimited by "Fix" lines) and
    parses each time step into a structured dictionary.
    
    Args:
        file_path: Path to the GPS data file. Defaults to DEFAULT_DATA_FILE.
        
    Returns:
        Dictionary mapping time step indices to parsed time step dictionaries.
        Each time step contains header-type-organized data (Fix, Accel, Gyro, etc.).
    """
    header_dict = get_headers(HEADER_FILE)
    data = {}
    
    with Path(file_path).open("r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Split data into time steps (each time step starts with a "Fix" line)
    time_step_data = []
    cur_data = []
    last_was_fix = True
    
    for line in lines:
        if line.rstrip().startswith('Fix'):
            if not last_was_fix:
                # End of previous time step
                time_step_data.append(cur_data)
                cur_data = []
            last_was_fix = True
        else:
            last_was_fix = False
        cur_data.append(line)
    
    # Add the last time step
    time_step_data.append(cur_data)
    
    # Parse each time step
    for t in range(len(time_step_data)):
        data[t] = parse_timestep_data(time_step_data[t], header_dict)
    
    return data


# ============================================================================
# TIME SERIES EXTRACTION
# ============================================================================

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
        lats.append(safe_float(fix.get("LatitudeDegrees")))
        lons.append(safe_float(fix.get("LongitudeDegrees")))
        alts.append(safe_float(fix.get("AltitudeMeters")))
        speeds.append(safe_float(fix.get("SpeedMps")))
        accuracies.append(safe_float(fix.get("AccuracyMeters")))
    
    return {
        "lat": mean_or_nan(lats),
        "lon": mean_or_nan(lons),
        "alt": mean_or_nan(alts),
        "speed_mps": mean_or_nan(speeds),
        "pos_accuracy_m": mean_or_nan(accuracies),
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
            row["imu_accel_x"] = safe_float(accel.get("AccelXMps2") or accel.get("UncalAccelXMps2"))
            row["imu_accel_y"] = safe_float(accel.get("AccelYMps2") or accel.get("UncalAccelYMps2"))
            row["imu_accel_z"] = safe_float(accel.get("AccelZMps2") or accel.get("UncalAccelZMps2"))
        
        # Extract IMU Gyroscope data (prefer calibrated, fall back to uncalibrated)
        gyro_entries = timestep.get("Gyro") or timestep.get("UncalGyro")
        if gyro_entries:
            gyro = gyro_entries[0]
            row["imu_gyro_x"] = safe_float(gyro.get("GyroXRadPerSec") or gyro.get("UncalGyroXRadPerSec"))
            row["imu_gyro_y"] = safe_float(gyro.get("GyroYRadPerSec") or gyro.get("UncalGyroYRadPerSec"))
            row["imu_gyro_z"] = safe_float(gyro.get("GyroZRadPerSec") or gyro.get("UncalGyroZRadPerSec"))
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Clean and sort by timestamp
    df = df.dropna(subset=["timestamp_ms"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def latlon_to_xy(lat_series: pd.Series, lon_series: pd.Series, 
                 ref_lat_rad: float, ref_lon_rad: float) -> tuple:
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


# ============================================================================
# TELEMETRY & GEOJSON CONVERSION
# ============================================================================

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
            "elapsed_s": round_float(row.elapsed_s),
            "lat": preserve_precision(row.lat),
            "lon": preserve_precision(row.lon),
            "alt": preserve_precision(row.alt, digits=2),
            "speed_mps": round_float(row.speed_mps),
            "speed_mph": round_float(row.speed_mph),
            "long_accel_mps2": round_float(row.long_accel_mps2),
            "lat_accel_mps2": round_float(row.lat_accel_mps2),
            "distance_m": preserve_precision(row.distance_along_m),
            "heading_deg": round_float(row.heading_deg, digits=2),
            "imu_accel_x": round_float(getattr(row, "imu_accel_x", np.nan)),
            "imu_accel_y": round_float(getattr(row, "imu_accel_y", np.nan)),
            "imu_accel_z": round_float(getattr(row, "imu_accel_z", np.nan)),
            "imu_gyro_z": round_float(getattr(row, "imu_gyro_z", np.nan)),
            "pos_accuracy_m": round_float(getattr(row, "pos_accuracy_m", np.nan)),
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


# ============================================================================
# LAP ANALYSIS
# ============================================================================

def detect_laps(telemetry: List[Dict], tolerance_m: float = 40.0, 
                skip_samples: int = 50) -> List[Dict]:
    """
    Detect lap boundaries by identifying returns to the start/finish line.
    
    Uses the first valid GNSS fix as the start/finish reference point.
    Increments lap count when the vehicle returns within tolerance_m of this point,
    after at least skip_samples points have passed (to avoid noise near start).
    Also annotates each telemetry sample with its lap number.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        tolerance_m: Distance threshold in meters for lap detection. Default 40.0.
        skip_samples: Minimum samples to skip after boundary before detecting
                     next lap (prevents false positives). Default 50.
        
    Returns:
        List of lap record dictionaries, each containing lap_number, start_time,
        end_time, lap_time_s, distance_m, sector_times_s, sector_splits_m, etc.
    """
    if not telemetry:
        return []
    
    # Find first valid position as start/finish reference
    first_sample = next(
        (s for s in telemetry if s["lat"] is not None and s["lon"] is not None),
        None
    )
    if not first_sample:
        return []
    
    start_lat, start_lon = first_sample["lat"], first_sample["lon"]
    lap_id = 0
    samples_since_boundary = 0
    lap_ranges = []
    current_start_idx = 0
    
    # Scan through telemetry and detect lap boundaries
    for idx, sample in enumerate(telemetry):
        sample["lap"] = lap_id
        lat, lon = sample["lat"], sample["lon"]
        
        if lat is None or lon is None:
            samples_since_boundary += 1
            continue
        
        # Check distance to start/finish
        dist = haversine_m(lat, lon, start_lat, start_lon)
        
        if dist < tolerance_m and samples_since_boundary > skip_samples:
            # New lap detected
            lap_ranges.append((lap_id, current_start_idx, idx))
            lap_id += 1
            samples_since_boundary = 0
            current_start_idx = idx
        
        samples_since_boundary += 1
    
    # Add final lap
    lap_ranges.append((lap_id, current_start_idx, len(telemetry) - 1))
    
    # Build lap records
    laps = []
    for lap_number, start_idx, end_idx in lap_ranges:
        record = build_lap_record(telemetry, start_idx, end_idx, lap_index=lap_number + 1)
        if record:
            laps.append(record)
    
    return laps


def build_fallback_lap(telemetry: List[Dict]) -> Dict:
    """
    Create a single fallback lap covering the entire session.
    
    Used when no lap boundaries are detected, treating the entire session
    as one continuous lap.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        
    Returns:
        Lap record dictionary, or empty dict if telemetry is empty.
    """
    if not telemetry:
        return {}
    
    start_idx = 0
    end_idx = len(telemetry) - 1
    return build_lap_record(telemetry, start_idx, end_idx, lap_index=1)


def build_lap_record(telemetry: List[Dict], start_idx: int, end_idx: int, 
                     lap_index: int, sectors: int = 3) -> Dict:
    """
    Build a lap record with timing and distance information.
    
    Computes lap time, distance, and sector splits for a given lap range.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        start_idx: Starting sample index for this lap.
        end_idx: Ending sample index for this lap.
        lap_index: Lap number (1-indexed).
        sectors: Number of sectors to split the lap into. Default 3.
        
    Returns:
        Dictionary with lap_number, start_time, end_time, lap_time_s,
        distance_m, sector_times_s, sector_splits_m, start_sample_idx,
        end_sample_idx. Returns empty dict if insufficient samples.
    """
    lap_samples = telemetry[start_idx:end_idx + 1]
    
    if len(lap_samples) < 2:
        return {}
    
    lap_distance = lap_samples[-1]["distance_m"] - lap_samples[0]["distance_m"]
    lap_time = lap_samples[-1]["elapsed_s"] - lap_samples[0]["elapsed_s"]
    sector_times = compute_sector_times(lap_samples, sectors)
    
    return {
        "lap_number": lap_index,
        "start_time": lap_samples[0]["timestamp"],
        "end_time": lap_samples[-1]["timestamp"],
        "lap_time_s": round_float(lap_time),
        "distance_m": round_float(lap_distance),
        "sector_times_s": sector_times["times"],
        "sector_splits_m": sector_times["splits"],
        "start_sample_idx": start_idx,
        "end_sample_idx": end_idx,
    }


def compute_sector_times(lap_samples: List[Dict], sectors: int) -> Dict:
    """
    Compute sector times and split distances for a lap.
    
    Divides the lap into equal-distance sectors and computes the time
    taken for each sector.
    
    Args:
        lap_samples: Telemetry samples for a single lap.
        sectors: Number of sectors to divide the lap into.
        
    Returns:
        Dictionary with "times" (list of sector times in seconds) and
        "splits" (list of split distances in meters at each boundary).
    """
    if len(lap_samples) < 2 or sectors < 1:
        return {"times": [], "splits": []}
    
    start_distance = lap_samples[0]["distance_m"]
    lap_distance = lap_samples[-1]["distance_m"] - start_distance
    
    if lap_distance <= 0:
        return {"times": [], "splits": []}
    
    # Calculate sector boundaries
    boundaries = [
        start_distance + lap_distance * (i / sectors)
        for i in range(1, sectors + 1)
    ]
    
    sector_times = []
    splits = []
    boundary_idx = 0
    last_boundary_time = lap_samples[0]["elapsed_s"]
    
    # Find when each boundary is crossed
    for i in range(1, len(lap_samples)):
        if boundary_idx >= len(boundaries):
            break
        
        prev_sample = lap_samples[i - 1]
        curr_sample = lap_samples[i]
        prev_dist = prev_sample["distance_m"]
        curr_dist = curr_sample["distance_m"]
        
        boundary = boundaries[boundary_idx]
        
        # Check if boundary was crossed between prev and curr
        if prev_dist <= boundary <= curr_dist:
            # Interpolate time at boundary
            ratio = 0 if curr_dist == prev_dist else (boundary - prev_dist) / (curr_dist - prev_dist)
            boundary_time = prev_sample["elapsed_s"] + ratio * (curr_sample["elapsed_s"] - prev_sample["elapsed_s"])
            
            sector_times.append(round_float(boundary_time - last_boundary_time))
            splits.append(round_float(boundary - start_distance))
            last_boundary_time = boundary_time
            boundary_idx += 1
    
    return {"times": sector_times, "splits": splits}


def annotate_lap_samples(telemetry: List[Dict], laps: List[Dict]) -> None:
    """
    Annotate telemetry samples with lap-specific metrics.
    
    Adds lap_index, lap_elapsed_s, and lap_distance_m to each sample within
    a lap, relative to the lap start.
    
    Args:
        telemetry: List of telemetry record dictionaries (modified in-place).
        laps: List of lap record dictionaries.
    """
    if not telemetry or not laps:
        return
    
    for lap in laps:
        start_idx = lap.get("start_sample_idx")
        end_idx = lap.get("end_sample_idx")
        
        if start_idx is None or end_idx is None:
            continue
        
        start_idx = max(0, start_idx)
        end_idx = min(len(telemetry) - 1, end_idx)
        
        start_time = telemetry[start_idx].get("elapsed_s") or 0.0
        start_distance = telemetry[start_idx].get("distance_m") or 0.0
        
        for idx in range(start_idx, end_idx + 1):
            sample = telemetry[idx]
            sample["lap_index"] = lap["lap_number"]
            sample["lap_elapsed_s"] = round_float(
                (sample.get("elapsed_s") or 0.0) - start_time, 3
            )
            sample["lap_distance_m"] = preserve_precision(
                (sample.get("distance_m") or 0.0) - start_distance
            )


def build_lap_features(telemetry: List[Dict], laps: List[Dict]) -> Dict:
    """
    Build GeoJSON features for each lap with speed-based coloring.
    
    Creates a LineString feature for each lap, with segment colors based on
    speed bands for visualization.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        laps: List of lap record dictionaries.
        
    Returns:
        GeoJSON FeatureCollection with one LineString feature per lap,
        including segment coordinates and color information.
    """
    features = []
    
    if not laps:
        return {"type": "FeatureCollection", "features": features}
    
    palette = ["#ff2d55", "#00ffa3", "#ffd60a", "#28a2ff", "#b28dff", "#ff8a5b"]
    
    for idx, lap in enumerate(laps):
        start_idx = lap.get("start_sample_idx")
        end_idx = lap.get("end_sample_idx")
        
        if start_idx is None or end_idx is None:
            continue
        
        coords = []
        segments = []
        segment_colors = []
        prev_point = None
        
        for sample in telemetry[start_idx : min(end_idx + 1, len(telemetry))]:
            lat, lon = sample["lat"], sample["lon"]
            if lat is None or lon is None:
                continue
            
            point = [lon, lat]
            coords.append(point)
            
            if prev_point is not None:
                segments.append([prev_point, point])
                speed_kmh = (sample.get("speed_mps") or 0) * 3.6
                segment_colors.append(speed_band_color(speed_kmh))
            
            prev_point = point
        
        if len(coords) < 2:
            continue
        
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "lap_number": lap["lap_number"],
                "lap_time_s": lap["lap_time_s"],
                "segments": segments,
                "segment_colors": segment_colors,
                "stroke": palette[idx % len(palette)],
                "stroke_width": 5,
            },
        })
    
    return {"type": "FeatureCollection", "features": features}


def speed_band_color(speed_kmh: float) -> str:
    """
    Map speed to a color for visualization.
    
    Color bands:
    - < 30 km/h: Blue
    - 30-55 km/h: Green
    - 55-80 km/h: Yellow
    - 80-110 km/h: Orange
    - >= 110 km/h: Red
    
    Args:
        speed_kmh: Speed in kilometers per hour.
        
    Returns:
        Hex color code string.
    """
    if speed_kmh < 30:
        return "#0088ff"  # blue
    if speed_kmh < 55:
        return "#00ff00"  # green
    if speed_kmh < 80:
        return "#ffff00"  # yellow
    if speed_kmh < 110:
        return "#ff8800"  # orange
    return "#ff0000"  # red


def build_lap_delta_traces(telemetry: List[Dict], laps: List[Dict]) -> List[Dict]:
    """
    Build time delta traces comparing each lap to the fastest lap.
    
    For each lap, computes the time difference at each distance point
    compared to the reference (fastest) lap. Used for delta visualization.
    
    Args:
        telemetry: List of telemetry record dictionaries.
        laps: List of lap record dictionaries.
        
    Returns:
        List of delta trace dictionaries, each containing lap_number,
        reference_lap, trace (list of {distance_m, time_delta_s}),
        lap_time_s, and reference_time_s.
    """
    if len(laps) < 2:
        return []
    
    # Find fastest lap as reference
    valid_laps = [lap for lap in laps if lap.get("lap_time_s")]
    if len(valid_laps) < 2:
        return []
    
    reference = min(valid_laps, key=lambda lap: lap["lap_time_s"])
    ref_samples = telemetry[reference["start_sample_idx"] : reference["end_sample_idx"] + 1]
    
    ref_dist = np.array(
        [sample.get("lap_distance_m") or 0.0 for sample in ref_samples], dtype=float
    )
    ref_time = np.array(
        [sample.get("lap_elapsed_s") or 0.0 for sample in ref_samples], dtype=float
    )
    
    if len(ref_dist) < 2:
        return []
    
    deltas = []
    
    for lap in valid_laps:
        if lap["lap_number"] == reference["lap_number"]:
            continue
        
        samples = telemetry[lap["start_sample_idx"] : lap["end_sample_idx"] + 1]
        trace = []
        
        for sample in samples:
            lap_distance = sample.get("lap_distance_m")
            lap_elapsed = sample.get("lap_elapsed_s")
            
            if lap_distance is None or lap_elapsed is None:
                continue
            
            # Interpolate reference time at this distance
            ref_time_at_distance = np.interp(
                float(lap_distance),
                ref_dist,
                ref_time,
                left=ref_time[0],
                right=ref_time[-1],
            )
            
            delta = float(lap_elapsed) - float(ref_time_at_distance)
            
            trace.append({
                "distance_m": round_float(lap_distance, 2),
                "time_delta_s": round_float(delta, 3),
            })
        
        if trace:
            deltas.append({
                "lap_number": lap["lap_number"],
                "reference_lap": reference["lap_number"],
                "trace": trace,
                "lap_time_s": lap["lap_time_s"],
                "reference_time_s": reference["lap_time_s"],
            })
    
    return deltas


# ============================================================================
# CORNER DETECTION
# ============================================================================

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
            "entry_speed_mph": round_float(speeds[0], 1),
            "exit_speed_mph": round_float(speeds[-1], 1),
            "min_speed_mph": round_float(min(speeds), 1),
            "max_lat_g": round_float(max(lat_accels) / 9.80665, 3),  # Convert m/s² to G
            "duration_s": round_float(
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


# ============================================================================
# HEATMAP GENERATION
# ============================================================================

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
                "intensity": round_float(intensity, 3),
                "value": round_float(value, 3),
            })
        
        return points
    
    return {
        "brake": build_points(lambda s: min(s.get("long_accel_mps2") or 0.0, 0.0), max_brake),
        "throttle": build_points(lambda s: max(s.get("long_accel_mps2") or 0.0, 0.0), max_throttle),
        "lateral": build_points(lambda s: s.get("lat_accel_mps2") or 0.0, max_lat),
    }


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

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


# ============================================================================
# MAIN SESSION BUILDER
# ============================================================================

def build_session_payload(data_file: Path = DEFAULT_DATA_FILE) -> Dict:
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
    gps_data = load_gps_data(data_file)
    df = extract_time_series(gps_data)
    
    if df.empty:
        raise ValueError("Parsed time-series is empty. Check data file.")
    
    df = compute_derived_metrics(df)
    telemetry = build_telemetry_records(df)
    track_geojson = telemetry_to_geojson(telemetry)
    
    # Detect laps
    laps = detect_laps(telemetry)
    if not laps:
        fallback = build_fallback_lap(telemetry)
        laps = [fallback] if fallback else []
    
    # Annotate samples with lap information
    annotate_lap_samples(telemetry, laps)
    
    # Build lap visualizations
    lap_features = build_lap_features(telemetry, laps)
    lap_deltas = build_lap_delta_traces(telemetry, laps)
    
    # Detect corners and build heatmaps
    corners = detect_corners(telemetry)
    heatmap = build_heatmap_layers(telemetry)
    
    return {
        "track": track_geojson,
        "telemetry": telemetry,
        "laps": laps,
        "lap_features": lap_features,
        "lap_deltas": lap_deltas,
        "corners": corners,
        "heatmap": heatmap,
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_float(value) -> float:
    """
    Safely convert a value to float, returning NaN on failure.
    
    Args:
        value: Value to convert (string, number, etc.).
        
    Returns:
        Float value, or np.nan if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def mean_or_nan(values: List[float]) -> float:
    """
    Compute mean of values, ignoring None and NaN.
    
    Args:
        values: List of numeric values (may contain None or NaN).
        
    Returns:
        Mean of valid values, or np.nan if no valid values exist.
    """
    cleaned = [v for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def round_float(value, digits: int = 3) -> Optional[float]:
    """
    Round a float value, handling None, NaN, and Inf.
    
    Args:
        value: Value to round.
        digits: Number of decimal places. Default 3.
        
    Returns:
        Rounded float, or None if value is None, NaN, or Inf.
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return round(float(value), digits)


def preserve_precision(value, digits: Optional[int] = None) -> Optional[float]:
    """
    Preserve or round precision of a float value.
    
    Args:
        value: Value to process.
        digits: Number of decimal places. If None, preserves original precision.
        
    Returns:
        Float value (rounded if digits specified), or None if value is None or NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if digits is None:
        return float(value)
    return round(float(value), digits)
