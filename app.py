from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
from pathlib import Path
import analyze_gps_data

app = FastAPI()

# Serve the static folder (for index.html, JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    # serve the main webpage
    return FileResponse("static/index.html")

# Example: preprocess GPS data once and keep in memory
# In practice, you’d put your real analysis here
def load_and_process_gps():
    # EXAMPLE: return dummy GeoJSON
    # Replace this with reading your file + analysis results
    raw_traj = analyze_gps_data.get_raw_trajectory()
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-122.4194, 37.7749],
                [-118.2437, 34.0522],
            ],
        },
        "properties": {"name": "Example track"},
    }
    return {
        "type": "FeatureCollection",
        "features": [raw_traj],
    }

processed_geojson = load_and_process_gps()

@app.get("/api/track")
def get_track():
    # Return your processed GPS data as GeoJSON
    return processed_geojson

# Run with: uvicorn app:app --reload
