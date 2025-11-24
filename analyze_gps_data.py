import numpy as np
import pandas as pd
from datetime import datetime,timezone
import matplotlib.pyplot as plt
import gnss_lib_py as glp

header_file = "GPS Data\headers.txt"

def get_headers(header_file):
    # Create a dictionary of headers
    headers_dict = {}

    # Read them
    with open(header_file, 'r') as file:
        headers = file.readlines()
    
    # Parse the headers
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
    for t in range(0,len(time_step_data)):
        data[t] = parse_timestep_data(time_step_data[t],header_dict)
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
            for i in range(0,num_sats):
                loc[0] += float(data[n]["Fix"][i]["LatitudeDegrees"]) / num_sats
                loc[1] += float(data[n]["Fix"][i]["LongitudeDegrees"]) / num_sats
                loc[2] += float(data[n]["Fix"][i]["AltitudeMeters"]) / num_sats
            useful_data[n]["Location"] = loc
        except:
            useful_data[n]["Location"] = None
    return useful_data

def plot_trajectory(useful_data):
    num_timesteps = len(useful_data.keys())

    # Get Trajectory in ECEF
    ecef_data = np.zeros((num_timesteps,3))
    for n in range(0, num_timesteps):
        ecef_data[n] = useful_data[n]["Location"]


    # Convert all points to ENU
    #enu = np.array(glp.ecef_to_geodetic(ecef_data))

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


"""gps_data = load_gps_data("GPS Data\Parsable Data.txt")
useful_data = get_useful_data(gps_data)
plot_trajectory(useful_data)"""