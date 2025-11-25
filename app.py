"""
FastAPI Web Application for GPS Track Analysis

This module provides a REST API and web interface for viewing and exporting
GPS track analysis data, including telemetry, laps, corners, and visualizations.
"""

import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import analyze_gps_data


# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI()

# Serve static files (HTML, CSS, JavaScript)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# SESSION LOADING
# ============================================================================

def load_session() -> dict:
    """
    Load and process GPS session data at application startup.
    
    Calls the analysis pipeline to parse GPS data, compute metrics, detect
    laps, and generate all visualization data. This is done once at startup
    to avoid reprocessing on every request.
    
    Returns:
        Dictionary containing complete session payload with:
        - track: GeoJSON FeatureCollection
        - telemetry: List of telemetry records
        - laps: List of lap records
        - lap_features: GeoJSON features for each lap
        - lap_deltas: Time delta traces
        - corners: List of detected corners
        - heatmap: Heatmap data
        
    Raises:
        HTTPException: If session loading fails (status 500).
    """
    try:
        return analyze_gps_data.build_session_payload()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load telemetry: {exc}"
        ) from exc


# Load session data once at startup
processed_session = load_session()


# ============================================================================
# ROOT & STATIC ROUTES
# ============================================================================

@app.get("/")
def read_root():
    """
    Serve the main web interface.
    
    Returns the index.html file which provides the interactive map and
    visualization interface for viewing GPS track data.
    
    Returns:
        FileResponse: The static index.html file.
    """
    return FileResponse("static/index.html")


# ============================================================================
# API ROUTES - DATA RETRIEVAL
# ============================================================================

@app.get("/api/session")
def get_session():
    """
    Get the complete session payload.
    
    Returns all processed session data including track, telemetry, laps,
    lap features, lap deltas, corners, and heatmap data.
    
    Returns:
        Dictionary containing the complete session payload.
    """
    return processed_session


@app.get("/api/track")
def get_track():
    """
    Get the track GeoJSON data.
    
    Returns the GeoJSON FeatureCollection containing the track path
    (LineString) and start/finish marker (Point).
    
    Returns:
        GeoJSON FeatureCollection with track geometry.
    """
    return processed_session["track"]


@app.get("/api/telemetry")
def get_telemetry():
    """
    Get all telemetry records.
    
    Returns the complete list of telemetry samples, each containing
    timestamp, position, speed, acceleration, IMU data, and lap information.
    
    Returns:
        List of telemetry record dictionaries.
    """
    return processed_session["telemetry"]


@app.get("/api/laps")
def get_laps():
    """
    Get all detected lap records.
    
    Returns information about each detected lap including lap number,
    timing, distance, sector times, and sample indices.
    
    Returns:
        List of lap record dictionaries.
    """
    return processed_session["laps"]


# ============================================================================
# API ROUTES - EXPORT
# ============================================================================

@app.get("/api/export/session")
def export_session():
    """
    Export the complete session data as JSON.
    
    Downloads the entire session payload as a JSON file, suitable for
    backup or external analysis.
    
    Returns:
        PlainTextResponse: JSON file with Content-Disposition header
        for download. Filename: carmintrack_session.json
    """
    body = json.dumps(processed_session, indent=2)
    headers = {"Content-Disposition": "attachment; filename=carmintrack_session.json"}
    return PlainTextResponse(
        body,
        media_type="application/json",
        headers=headers
    )


@app.get("/api/export/lap/{lap_number}")
def export_lap(lap_number: int):
    """
    Export a specific lap's telemetry data as CSV.
    
    Generates a CSV file containing all telemetry samples for the specified
    lap, including position, speed, acceleration, and IMU data.
    
    Args:
        lap_number: The lap number to export (1-indexed).
        
    Returns:
        PlainTextResponse: CSV file with Content-Disposition header
        for download. Filename: lap_{lap_number}.csv
        
    Raises:
        HTTPException: If lap_number is not found (status 404).
    """
    try:
        csv_body = analyze_gps_data.export_lap_csv(processed_session, lap_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    
    headers = {"Content-Disposition": f"attachment; filename=lap_{lap_number}.csv"}
    return PlainTextResponse(
        csv_body,
        media_type="text/csv",
        headers=headers
    )


# ============================================================================
# RUN INSTRUCTIONS
# ============================================================================
# Run with: uvicorn app:app --reload (or run test.bat)
