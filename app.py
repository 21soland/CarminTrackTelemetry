from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
from pathlib import Path
import analyze_gps_data

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

def load_and_process_gps():
    FILE_PATH = "GPS Data/Test_Data_11_24_25.txt"

    gps_data = analyze_gps_data.load_gps_data(FILE_PATH)
    useful   = analyze_gps_data.get_useful_data(gps_data)
    useful   = analyze_gps_data.detect_laps(useful, start_threshold_meters=12)
    laps_geojson = analyze_gps_data.get_laps_geojson(useful)

    return laps_geojson

processed_geojson = load_and_process_gps()

@app.get("/api/track")
def get_track():
    return processed_geojson