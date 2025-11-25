import csv
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path(__file__).parent / "GPS Data"
HEADER_FILE = DATA_DIR / "headers.txt"
DEFAULT_DATA_FILE = DATA_DIR / "Parsable Data.txt"


def get_headers(header_file: Path = HEADER_FILE) -> Dict[str, List[str]]:
    # Create a dictionary of headers
    headers_dict = {}

    # Read them
    with header_file.open("r", encoding="utf-8") as file:
        headers = file.readlines()
    
    # Parse the headers
    for header in headers:
        header = header.strip()
        header_list = header.split(',')
        headers_dict[header_list[0]] = header_list[1:]
    return headers_dict


def load_gps_data(file_path: Path = DEFAULT_DATA_FILE) -> Dict[int, Dict]:
    header_dict = get_headers(HEADER_FILE)
    data = {}

    with Path(file_path).open("r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Split everything by time-step
    time_step_data = []
    cur_data = []
    last_was_fix = True
    for line in lines:
        if line.rstrip().startswith('Fix'):
            if last_was_fix == False:
                time_step_data.append(cur_data)
                cur_data = []
            last_was_fix = True
        else:
            last_was_fix = False
        cur_data.append(line)
    time_step_data.append(cur_data) # Add the last data
    for t in range(0, len(time_step_data)):
        data[t] = parse_timestep_data(time_step_data[t], header_dict)
    return data

def get_header(line,headers):
    for header in headers:
        if line.startswith(header):
            return header

def parse_timestep_data(time_step,headers_dict):
    headers = list(headers_dict.keys())
    time_step_dict = {}
    for line in time_step:
        # Get which header type it is
        header_type = get_header(line,headers)
        split_line = line.rstrip().split(",")

        # Check if there are multiple of the same header
        n = 0 
        if header_type in time_step_dict:
            n = len(time_step_dict[header_type].keys())
        else:
            time_step_dict[header_type] = {}
        # Add the data for [header_type][n]
        time_step_dict[header_type][n] = {}

        header_type_keys = headers_dict[header_type]
        for i in range(1,len(header_type_keys)):
            time_step_dict[header_type][n][header_type_keys[i-1]] = split_line[i] #Note: this skips repeating fix
    return time_step_dict

# Get useful data
def build_session_payload(data_file: Path = DEFAULT_DATA_FILE) -> Dict:
    """
    Prepare the track GeoJSON, per-sample telemetry, and lap timing details.
    """
    gps_data = load_gps_data(data_file)
    df = extract_time_series(gps_data)
    if df.empty:
        raise ValueError("Parsed time-series is empty. Check data file.")

    df = compute_derived_metrics(df)
    telemetry = build_telemetry_records(df)
    track_geojson = telemetry_to_geojson(telemetry)
    laps = detect_laps(telemetry)
    if not laps:
        fallback = build_fallback_lap(telemetry)
        laps = [fallback] if fallback else []

    annotate_lap_samples(telemetry, laps)

    lap_features = build_lap_features(telemetry, laps)
    lap_deltas = build_lap_delta_traces(telemetry, laps)
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


def get_raw_trajectory():
    """
    Backwards compatibility helper for existing code paths that only expect the
    LineString feature.
    """
    session = build_session_payload()
    # Return the first feature (the LineString) for legacy callers
    return session["track"]["features"][0]


def extract_time_series(parsed_data: Dict[int, Dict]) -> pd.DataFrame:
    """
    Flatten the nested GNSS/IMU structure into a time-indexed table that we can
    use for downstream telemetry calculations.
    """
    rows = []
    for idx in sorted(parsed_data.keys()):
        timestep = parsed_data[idx]
        row = {
            "index": idx,
            "timestamp_ms": extract_timestamp_ms(timestep),
        }

        # GNSS Fix data
        fix_entries = timestep.get("Fix", {})
        if fix_entries:
            row.update(extract_fix_fields(fix_entries))

        # IMU Accel data
        accel_entries = timestep.get("Accel") or timestep.get("UncalAccel")
        if accel_entries:
            accel = accel_entries[0]
            row["imu_accel_x"] = safe_float(accel.get("AccelXMps2") or accel.get("UncalAccelXMps2"))
            row["imu_accel_y"] = safe_float(accel.get("AccelYMps2") or accel.get("UncalAccelYMps2"))
            row["imu_accel_z"] = safe_float(accel.get("AccelZMps2") or accel.get("UncalAccelZMps2"))

        # IMU Gyro data
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

    df = df.dropna(subset=["timestamp_ms"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_timestamp_ms(timestep: Dict) -> float | None:
    # Prefer the GNSS fix timestamp if present, else Raw record.
    fix_entries = timestep.get("Fix", {})
    if fix_entries:
        first_fix = fix_entries.get(0, {})
        ts = first_fix.get("UnixTimeMillis")
        if ts:
            return float(ts)

    raw_entries = timestep.get("Raw", {})
    if raw_entries:
        raw = raw_entries.get(0, {})
        ts = raw.get("utcTimeMillis")
        if ts:
            return float(ts)
    return None


def extract_fix_fields(fix_entries: Dict[int, Dict]) -> Dict[str, float]:
    lats, lons, alts, speeds = [], [], [], []
    accuracies = []
    for fix in fix_entries.values():
        lats.append(safe_float(fix.get("LatitudeDegrees")))
        lons.append(safe_float(fix.get("LongitudeDegrees")))
        alts.append(safe_float(fix.get("AltitudeMeters")))
        speeds.append(safe_float(fix.get("SpeedMps")))
        accuracies.append(safe_float(fix.get("AccuracyMeters")))

    row = {}
    row["lat"] = mean_or_nan(lats)
    row["lon"] = mean_or_nan(lons)
    row["alt"] = mean_or_nan(alts)
    row["speed_mps"] = mean_or_nan(speeds)
    row["pos_accuracy_m"] = mean_or_nan(accuracies)
    return row


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def mean_or_nan(values: List[float]) -> float:
    cleaned = [v for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # Use the first valid lat/lon as reference for ENU approximation
    valid_lat = df["lat"].dropna()
    valid_lon = df["lon"].dropna()
    if valid_lat.empty or valid_lon.empty:
        return df
    ref_lat = np.deg2rad(valid_lat.iloc[0])
    ref_lon = np.deg2rad(valid_lon.iloc[0])

    x_m, y_m = latlon_to_xy(df["lat"], df["lon"], ref_lat, ref_lon)
    df["x_m"] = x_m
    df["y_m"] = y_m

    dx = df["x_m"].diff()
    dy = df["y_m"].diff()
    dt = df["elapsed_s"].diff().replace({0: np.nan})
    segment_dist = np.sqrt(dx**2 + dy**2)
    df["segment_distance_m"] = segment_dist.fillna(0)
    df["distance_along_m"] = df["segment_distance_m"].cumsum().fillna(0)

    speed_course = segment_dist / dt
    df["speed_mps"] = df["speed_mps"].fillna(speed_course)
    df["speed_mps"] = df["speed_mps"].ffill().fillna(0)
    df["speed_mph"] = df["speed_mps"] * 2.23694

    heading = np.arctan2(dx.fillna(0), dy.fillna(0))
    heading = np.unwrap(heading)
    df["heading_deg"] = np.rad2deg(heading)

    long_accel = df["speed_mps"].diff() / dt
    df["long_accel_mps2"] = long_accel.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(df) > 1:
        heading_rate = np.gradient(heading, df["elapsed_s"])
        df["lat_accel_mps2"] = df["speed_mps"] * heading_rate
    else:
        df["lat_accel_mps2"] = 0

    df["lat_accel_mps2"] = df["lat_accel_mps2"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def latlon_to_xy(lat_series, lon_series, ref_lat_rad, ref_lon_rad):
    """
    Convert lat/lon to a local tangent plane using a simple equirectangular
    approximation, adequate for small tracks.
    """
    R = 6378137.0
    lat_rad = np.deg2rad(lat_series)
    lon_rad = np.deg2rad(lon_series)
    x = (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad) * R
    y = (lat_rad - ref_lat_rad) * R
    return x, y


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1_rad, lon1_rad = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2_rad, lon2_rad = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_telemetry_records(df: pd.DataFrame) -> List[Dict]:
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


def detect_laps(telemetry: List[Dict], tolerance_m: float = 40.0, skip_samples: int = 50) -> List[Dict]:
    """
    Re-implementation of the original lap detection:
    - Uses the first valid GNSS fix as the start/finish reference.
    - Increments the lap count whenever we re-enter the tolerance circle after
      at least `skip_samples` points (avoids counting noise right after launch).
    - Annotates each telemetry sample with its lap id (zero-based) so we can
      build per-lap overlays and speed-based segments later.
    """
    if not telemetry:
        return []

    first_sample = next((s for s in telemetry if s["lat"] is not None and s["lon"] is not None), None)
    if not first_sample:
        return []

    start_lat, start_lon = first_sample["lat"], first_sample["lon"]
    lap_id = 0
    samples_since_boundary = 0
    lap_ranges = []
    current_start_idx = 0

    for idx, sample in enumerate(telemetry):
        sample["lap"] = lap_id
        lat, lon = sample["lat"], sample["lon"]
        if lat is None or lon is None:
            samples_since_boundary += 1
            continue

        dist = haversine_m(lat, lon, start_lat, start_lon)
        if dist < tolerance_m and samples_since_boundary > skip_samples:
            lap_ranges.append((lap_id, current_start_idx, idx))
            lap_id += 1
            samples_since_boundary = 0
            current_start_idx = idx
        samples_since_boundary += 1

    lap_ranges.append((lap_id, current_start_idx, len(telemetry) - 1))

    laps = []
    for lap_number, start_idx, end_idx in lap_ranges:
        record = build_lap_record(telemetry, start_idx, end_idx, lap_index=lap_number + 1)
        if record:
            laps.append(record)
    return laps


def build_fallback_lap(telemetry: List[Dict]) -> Dict:
    if not telemetry:
        return {}
    start_idx = 0
    end_idx = len(telemetry) - 1
    return build_lap_record(telemetry, start_idx, end_idx, lap_index=1)


def build_lap_record(telemetry: List[Dict], start_idx: int, end_idx: int, lap_index: int, sectors: int = 3) -> Dict:
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


def compute_sector_times(lap_samples: List[Dict], sectors: int):
    if len(lap_samples) < 2 or sectors < 1:
        return {"times": [], "splits": []}

    start_distance = lap_samples[0]["distance_m"]
    lap_distance = lap_samples[-1]["distance_m"] - start_distance
    if lap_distance <= 0:
        return {"times": [], "splits": []}

    boundaries = [
        start_distance + lap_distance * (i / sectors)
        for i in range(1, sectors + 1)
    ]

    sector_times = []
    splits = []
    boundary_idx = 0
    last_boundary_time = lap_samples[0]["elapsed_s"]

    for i in range(1, len(lap_samples)):
        if boundary_idx >= len(boundaries):
            break
        prev_sample = lap_samples[i - 1]
        curr_sample = lap_samples[i]
        prev_dist = prev_sample["distance_m"]
        curr_dist = curr_sample["distance_m"]

        boundary = boundaries[boundary_idx]
        if prev_dist <= boundary <= curr_dist:
            ratio = 0 if curr_dist == prev_dist else (boundary - prev_dist) / (curr_dist - prev_dist)
            boundary_time = prev_sample["elapsed_s"] + ratio * (curr_sample["elapsed_s"] - prev_sample["elapsed_s"])
            sector_times.append(round_float(boundary_time - last_boundary_time))
            splits.append(round_float(boundary - start_distance))
            last_boundary_time = boundary_time
            boundary_idx += 1

    return {"times": sector_times, "splits": splits}


def round_float(value, digits: int = 3):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return round(float(value), digits)


def preserve_precision(value, digits: int | None = None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if digits is None:
        return float(value)
    return round(float(value), digits)


def build_lap_features(telemetry: List[Dict], laps: List[Dict]) -> Dict:
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

        features.append(
            {
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
            }
        )

    return {"type": "FeatureCollection", "features": features}


def speed_band_color(speed_kmh: float) -> str:
    if speed_kmh < 30:
        return "#0088ff"  # blue
    if speed_kmh < 55:
        return "#00ff00"  # green
    if speed_kmh < 80:
        return "#ffff00"  # yellow
    if speed_kmh < 110:
        return "#ff8800"  # orange
    return "#ff0000"  # red


def annotate_lap_samples(telemetry: List[Dict], laps: List[Dict]) -> None:
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
            sample["lap_elapsed_s"] = round_float((sample.get("elapsed_s") or 0.0) - start_time, 3)
            sample["lap_distance_m"] = preserve_precision(
                (sample.get("distance_m") or 0.0) - start_distance
            )


def build_lap_delta_traces(telemetry: List[Dict], laps: List[Dict]) -> List[Dict]:
    if len(laps) < 2:
        return []

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
            ref_time_at_distance = np.interp(
                float(lap_distance),
                ref_dist,
                ref_time,
                left=ref_time[0],
                right=ref_time[-1],
            )
            delta = float(lap_elapsed) - float(ref_time_at_distance)
            trace.append(
                {
                    "distance_m": round_float(lap_distance, 2),
                    "time_delta_s": round_float(delta, 3),
                }
            )
        if trace:
            deltas.append(
                {
                    "lap_number": lap["lap_number"],
                    "reference_lap": reference["lap_number"],
                    "trace": trace,
                    "lap_time_s": lap["lap_time_s"],
                    "reference_time_s": reference["lap_time_s"],
                }
            )
    return deltas


def detect_corners(
    telemetry: List[Dict],
    lat_accel_threshold: float = 1.6,
    min_samples: int = 5,
) -> List[Dict]:
    corners = []
    idx = 0
    while idx < len(telemetry):
        sample = telemetry[idx]
        lat_accel = abs(sample.get("lat_accel_mps2") or 0.0)
        if lat_accel < lat_accel_threshold:
            idx += 1
            continue

        start_idx = idx
        while idx < len(telemetry) and abs(telemetry[idx].get("lat_accel_mps2") or 0.0) >= lat_accel_threshold:
            idx += 1
        end_idx = min(idx - 1, len(telemetry) - 1)

        if end_idx - start_idx + 1 < min_samples:
            continue

        segment = telemetry[start_idx : end_idx + 1]
        speeds = [sample.get("speed_mph") or 0.0 for sample in segment]
        lat_accels = [abs(sample.get("lat_accel_mps2") or 0.0) for sample in segment]
        lap_number = segment[0].get("lap_index")

        apex_idx = int(np.argmin(speeds))
        apex_sample = segment[apex_idx]

        corners.append(
            {
                "corner_id": len(corners) + 1,
                "lap_number": lap_number,
                "entry_speed_mph": round_float(speeds[0], 1),
                "exit_speed_mph": round_float(speeds[-1], 1),
                "min_speed_mph": round_float(min(speeds), 1),
                "max_lat_g": round_float(max(lat_accels) / 9.80665, 3),
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
            }
        )
    return corners


def build_heatmap_layers(telemetry: List[Dict]) -> Dict:
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
            points.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "intensity": round_float(intensity, 3),
                    "value": round_float(value, 3),
                }
            )
        return points

    return {
        "brake": build_points(lambda s: min(s.get("long_accel_mps2") or 0.0, 0.0), max_brake),
        "throttle": build_points(lambda s: max(s.get("long_accel_mps2") or 0.0, 0.0), max_throttle),
        "lateral": build_points(lambda s: s.get("lat_accel_mps2") or 0.0, max_lat),
    }


def export_lap_csv(session: Dict, lap_number: int) -> str:
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
    writer.writerow(
        [
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
        ]
    )

    for sample in telemetry[start_idx : end_idx + 1]:
        writer.writerow(
            [
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
            ]
        )

    return buffer.getvalue()