import numpy as np
import pandas as pd
from datetime import datetime,timezone
import matplotlib.pyplot as plt
import gnss_lib_py as glp

header_file = "GPS Data/headers.txt"

def get_headers(header_file):
    headers_dict = {}

    with open(header_file, 'r') as file:
        headers = file.readlines()
    
    for header in headers:
        header = header.strip()
        header_list = header.split(',')
        headers_dict[header_list[0]] = header_list[1:]
    return headers_dict

def load_gps_data(file_path):
    header_dict = get_headers(header_file)
    data = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
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
    time_step_data.append(cur_data)
    for t in range(0,len(time_step_data)):
        data[t] = parse_timestep_data(time_step_data[t],header_dict)
    return data


def get_header(line, headers):
    line_clean = line.strip()
    for header in headers:
        if line_clean.startswith(header):
            return header
    return None


def parse_timestep_data(time_step, headers_dict):
    headers = list(headers_dict.keys())
    time_step_dict = {}
    
    for line in time_step:
        header_type = get_header(line, headers)
        
        if header_type is None:
            continue 
        
        split_line = line.rstrip().split(",")
        

        n = 0
        if header_type in time_step_dict:
            n = len(time_step_dict[header_type])
        else:
            time_step_dict[header_type] = {}
        

        time_step_dict[header_type][n] = {}
        
        header_type_keys = headers_dict[header_type]
        for i in range(1, len(header_type_keys)):
            key = header_type_keys[i-1]

            if i < len(split_line):
                time_step_dict[header_type][n][key] = split_line[i]
            else:
                time_step_dict[header_type][n][key] = None
    
    return time_step_dict

def get_useful_data(data):
    useful_data = {}
    for n in data.keys():
        useful_data[n] = {}
        try:
            time_millis = int(data[n]["Raw"][0]["utcTimeMillis"])
            utc_dt = datetime.fromtimestamp(time_millis / 1000, tz=timezone.utc)
            useful_data[n]["Time"] = utc_dt
        except:
            useful_data[n]["Time"] = "N/A"

        try:
            num_sats = len(data[n]["Fix"].keys())
            loc = np.zeros(3)
            for i in range(0, num_sats):
                loc[0] += float(data[n]["Fix"][i]["LatitudeDegrees"]) / num_sats
                loc[1] += float(data[n]["Fix"][i]["LongitudeDegrees"]) / num_sats
                loc[2] += float(data[n]["Fix"][i]["AltitudeMeters"]) / num_sats
            useful_data[n]["Location"] = loc

            speeds = []
            for i in range(0, num_sats):
                s = data[n]["Fix"][i].get("SpeedMps", "")
                if s not in ("", None, ""):
                    try:
                        speeds.append(float(s))
                    except ValueError:
                        pass
            useful_data[n]["SpeedMps"] = sum(speeds) / len(speeds) if speeds else 0.0

        except:
            useful_data[n]["Location"] = None
            useful_data[n]["SpeedMps"] = 0.0

    return useful_data


def plot_trajectory(useful_data):
    num_timesteps = len(useful_data.keys())

    ecef_data = np.zeros((num_timesteps,3))
    for n in range(0, num_timesteps):
        ecef_data[n] = useful_data[n]["Location"]


    plt.plot(ecef_data[:,0], ecef_data[:,1], '-o')
    plt.axis('equal')
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("GNSS Trajectory (ENU)")
    plt.grid(True)
    plt.show()

def get_raw_trajectory():
    gps_data = load_gps_data("GPS Data\Parsable Data.txt")
    useful_data = get_useful_data(gps_data)
    num_timesteps = len(useful_data.keys())

    # Get Trajectory in ECEF
    ecef_data = np.zeros((num_timesteps,3))
    for n in range(0, num_timesteps):
        ecef_data[n] = useful_data[n]["Location"]
    # Build GeoJSON LineString object from trajectory data
    # Assume 'ecef_data' contains shape (N, 3): [lat, lon, alt] or [lat, lon, ...]
    # We'll assume latitude is in loc[0], longitude is in loc[1]
    # Prepare the coordinates list as [longitude, latitude] (GeoJSON standard)
    coordinates = []
    for n in range(num_timesteps):
        # Check if Location exists
        loc = useful_data[n]["Location"]
        if loc is not None:
            # For GeoJSON, [lon, lat] order
            coordinates.append([float(loc[1]), float(loc[0])])

    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {}
    }
    return geojson


def detect_laps(useful_data, start_threshold_meters=12):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000 
    
    positions = []
    for n in useful_data:
        loc = useful_data[n]["Location"]
        if loc is not None:
            positions.append((loc[0], loc[1], n))  # lat, lon, index
    
    if len(positions) == 0:
        return useful_data
    
    start_lat, start_lon = positions[0][0], positions[0][1]
    current_lap = 0
    for i, (lat, lon, idx) in enumerate(positions):
        # Distance to start/finish line
        dlat = radians(lat - start_lat)
        dlon = radians(lon - start_lon)
        a = sin(dlat/2)**2 + cos(radians(start_lat)) * cos(radians(lat)) * sin(dlon/2)**2
        dist = R * 2 * atan2(sqrt(a), sqrt(1 - a))
        
        if dist < 40 and i > 50:   # ← 40 meters tolerance, start counting after 50 points
            current_lap += 1
        useful_data[idx]["lap"] = current_lap
    
    print(f"Detected {current_lap + 1} complete laps")
    return useful_data


def get_laps_geojson(useful_data):
    laps = {}
    speeds = {}
    
    for n in useful_data:
        d = useful_data[n]
        # Only keep laps 0 and 1 in the interface for this data
        if "Location" not in d or "lap" not in d or d["lap"] > 1:
            continue
        lap_num = d["lap"]
        laps.setdefault(lap_num, [])
        speeds.setdefault(lap_num, [])
        loc = d["Location"]
        speed_kmh = float(d.get("Speed (m/s)", 0) or 0) * 3.6
        laps[lap_num].append([float(loc[1]), float(loc[0])])
        speeds[lap_num].append(speed_kmh)

    features = []
    lap_colors = ["#ff0000", "#00ff00"]  # red for lap 0, green for lap 1

    for lap_num in [0, 1]:
        if lap_num not in laps or len(laps[lap_num]) < 10:
            continue

        coords = laps[lap_num]
        lap_speeds = speeds[lap_num]

        segments = []
        segment_colors = [] # one color per segment
        for i in range(1, len(coords)):
            speed = lap_speeds[i]
            color = "#0088ff" if speed < 30 else \
                    "#00ff00" if speed < 55 else \
                    "#ffff00" if speed < 80 else \
                    "#ff8800" if speed < 110 else "#ff0000"

            # store each segment as a pair of [lon, lat] points
            segments.append([coords[i-1], coords[i]])
            segment_colors.append(color)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "stroke": lap_colors[lap_num],
                "stroke-width": 7,
                "lap": lap_num,
                "segments": segments,
                "segment_colors": segment_colors
            }
        })

    print(f"Generated {len(features)} clean laps with speed coloring")
    return {"type": "FeatureCollection", "features": features}


"""gps_data = load_gps_data("GPS Data\Parsable Data.txt")
useful_data = get_useful_data(gps_data)
plot_trajectory(useful_data)"""