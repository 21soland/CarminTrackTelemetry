"""
Lap Analysis for GPS Track Analysis

This module handles lap detection, lap record building, sector time computation,
lap visualization, and delta trace generation for comparing laps.
"""

import numpy as np
from typing import Dict, List
from . import metrics
from . import utils


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
        dist = metrics.haversine_m(lat, lon, start_lat, start_lon)
        
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
        "lap_time_s": utils.round_float(lap_time),
        "distance_m": utils.round_float(lap_distance),
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
            
            sector_times.append(utils.round_float(boundary_time - last_boundary_time))
            splits.append(utils.round_float(boundary - start_distance))
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
            sample["lap_elapsed_s"] = utils.round_float(
                (sample.get("elapsed_s") or 0.0) - start_time, 3
            )
            sample["lap_distance_m"] = utils.preserve_precision(
                (sample.get("distance_m") or 0.0) - start_distance
            )


def speed_band_color(speed_mph: float) -> str:
    """
    Map speed to a color for visualization, interpolating between bands.

    Color bands:
    - < 20 mph: Blue
    - 20-35 mph: Green
    - 35-50 mph: Yellow
    - 50-70 mph: Orange
    - >= 70 mph: Red

    Args:
        speed_mph: Speed in miles per hour.

    Returns:
        Hex color code string.
    """
    # Define speed band breakpoints and their corresponding RGB values
    bands = [
        (0,   (0, 136, 255)),   # Blue (#0088ff)
        (20,  (0, 255, 0)),     # Green (#00ff00)
        (35,  (255, 255, 0)),   # Yellow (#ffff00)
        (50,  (255, 136, 0)),   # Orange (#ff8800)
        (70,  (255, 0, 0)),     # Red (#ff0000)
    ]

    # Clamp speed_mph if out of bounds
    if speed_mph <= bands[0][0]:
        rgb = bands[0][1]
    elif speed_mph >= bands[-1][0]:
        rgb = bands[-1][1]
    else:
        # Find the interval it belongs to
        for i in range(len(bands) - 1):
            spd0, col0 = bands[i]
            spd1, col1 = bands[i + 1]
            if spd0 <= speed_mph < spd1:
                f = (speed_mph - spd0) / (spd1 - spd0)
                rgb = tuple(
                    int(col0[j] + f * (col1[j] - col0[j])) for j in range(3)
                )
                break

    return "#{:02x}{:02x}{:02x}".format(*rgb)


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
                speed_mph = sample.get("speed_mph") or 0.0
                segment_colors.append(speed_band_color(speed_mph))
            
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
                "distance_m": utils.round_float(lap_distance, 2),
                "time_delta_s": utils.round_float(delta, 3),
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

