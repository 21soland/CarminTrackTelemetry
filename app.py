"""
FastAPI Web Application for GPS Track Analysis

This module provides a REST API and web interface for viewing and exporting
GPS track analysis data, including telemetry, laps, corners, and visualizations.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Query
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
# DATASET DISCOVERY
# ============================================================================

def get_available_datasets() -> list:
    """
    Discover available GPS data files in the GPS Data directory.
    
    Scans the GPS Data directory for .txt files (excluding headers.txt)
    and returns a list of available datasets with their filenames.
    
    Returns:
        List of dictionaries with 'filename' and 'display_name' keys.
    """
    data_dir = analyze_gps_data.DATA_DIR
    datasets = []
    
    if not data_dir.exists():
        return datasets
    
    for file_path in data_dir.glob("*.txt"):
        if file_path.name == "headers.txt":
            continue
        
        # Use filename as display name, or create a nicer name
        display_name = file_path.stem.replace("_", " ").title()
        datasets.append({
            "filename": file_path.name,
            "display_name": display_name,
        })
    
    # Sort by filename
    datasets.sort(key=lambda x: x["filename"])
    return datasets


# ============================================================================
# SESSION LOADING & CACHING
# ============================================================================

# Cache for loaded sessions (dataset_filename -> session_data)
session_cache: Dict[str, dict] = {}


def load_session(dataset_filename: Optional[str] = None) -> dict:
    """
    Load and process GPS session data for a specific dataset.
    
    Calls the analysis pipeline to parse GPS data, compute metrics, detect
    laps, and generate all visualization data. Sessions are cached to avoid
    reprocessing on subsequent requests.
    
    Args:
        dataset_filename: Name of the data file to load. If None, uses default.
        
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
    # Use default if no dataset specified
    if dataset_filename is None:
        dataset_filename = analyze_gps_data.DEFAULT_DATA_FILE.name
    
    # Check cache first
    if dataset_filename in session_cache:
        return session_cache[dataset_filename]
    
    # Load the dataset
    try:
        data_file = analyze_gps_data.DATA_DIR / dataset_filename
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_filename}")
        
        session = analyze_gps_data.build_session_payload(data_file)
        
        # Cache the session
        session_cache[dataset_filename] = session
        
        return session
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load telemetry: {exc}"
        ) from exc


# ============================================================================
# ROOT & STATIC ROUTES
# ============================================================================

@app.get("/")
def read_root():
    """
    Serve the dataset selection menu.
    
    Returns the menu.html file which allows users to select which dataset
    they want to analyze.
    
    Returns:
        FileResponse: The static menu.html file.
    """
    return FileResponse("static/menu.html")


@app.get("/map")
def read_map():
    """
    Serve the main map interface.
    
    Returns the index.html file which provides the interactive map and
    visualization interface for viewing GPS track data.
    
    Returns:
        FileResponse: The static index.html file.
    """
    return FileResponse("static/index.html")


# ============================================================================
# API ROUTES - DATASET MANAGEMENT
# ============================================================================

@app.get("/api/datasets")
def get_datasets():
    """
    Get list of available datasets.
    
    Returns a list of all available GPS data files that can be analyzed.
    
    Returns:
        List of dictionaries with 'filename' and 'display_name' keys.
    """
    return get_available_datasets()


# ============================================================================
# API ROUTES - DATA RETRIEVAL
# ============================================================================

@app.get("/api/session")
def get_session(dataset: Optional[str] = Query(None, description="Dataset filename to load")):
    """
    Get the complete session payload for a specific dataset.
    
    Returns all processed session data including track, telemetry, laps,
    lap features, lap deltas, corners, and heatmap data.
    
    Args:
        dataset: Optional dataset filename. If not provided, uses default.
    
    Returns:
        Dictionary containing the complete session payload.
    """
    session = load_session(dataset)
    return session


@app.get("/api/track")
def get_track(dataset: Optional[str] = Query(None, description="Dataset filename to load")):
    """
    Get the track GeoJSON data for a specific dataset.
    
    Returns the GeoJSON FeatureCollection containing the track path
    (LineString) and start/finish marker (Point).
    
    Args:
        dataset: Optional dataset filename. If not provided, uses default.
    
    Returns:
        GeoJSON FeatureCollection with track geometry.
    """
    session = load_session(dataset)
    return session["track"]


@app.get("/api/telemetry")
def get_telemetry(dataset: Optional[str] = Query(None, description="Dataset filename to load")):
    """
    Get all telemetry records for a specific dataset.
    
    Returns the complete list of telemetry samples, each containing
    timestamp, position, speed, acceleration, IMU data, and lap information.
    
    Args:
        dataset: Optional dataset filename. If not provided, uses default.
    
    Returns:
        List of telemetry record dictionaries.
    """
    session = load_session(dataset)
    return session["telemetry"]


@app.get("/api/laps")
def get_laps(dataset: Optional[str] = Query(None, description="Dataset filename to load")):
    """
    Get all detected lap records for a specific dataset.
    
    Returns information about each detected lap including lap number,
    timing, distance, sector times, and sample indices.
    
    Args:
        dataset: Optional dataset filename. If not provided, uses default.
    
    Returns:
        List of lap record dictionaries.
    """
    session = load_session(dataset)
    return session["laps"]


# ============================================================================
# API ROUTES - EXPORT
# ============================================================================

@app.get("/api/export/session")
def export_session(dataset: Optional[str] = Query(None, description="Dataset filename to export")):
    """
    Export the complete session data as JSON.
    
    Downloads the entire session payload as a JSON file, suitable for
    backup or external analysis.
    
    Args:
        dataset: Optional dataset filename. If not provided, uses default.
    
    Returns:
        PlainTextResponse: JSON file with Content-Disposition header
        for download. Filename: carmintrack_session.json
    """
    session = load_session(dataset)
    body = json.dumps(session, indent=2)
    headers = {"Content-Disposition": "attachment; filename=carmintrack_session.json"}
    return PlainTextResponse(
        body,
        media_type="application/json",
        headers=headers
    )


@app.get("/api/export/lap/{lap_number}")
def export_lap(lap_number: int, dataset: Optional[str] = Query(None, description="Dataset filename to export")):
    """
    Export a specific lap's telemetry data as CSV.
    
    Generates a CSV file containing all telemetry samples for the specified
    lap, including position, speed, acceleration, and IMU data.
    
    Args:
        lap_number: The lap number to export (1-indexed).
        dataset: Optional dataset filename. If not provided, uses default.
        
    Returns:
        PlainTextResponse: CSV file with Content-Disposition header
        for download. Filename: lap_{lap_number}.csv
        
    Raises:
        HTTPException: If lap_number is not found (status 404).
    """
    session = load_session(dataset)
    try:
        csv_body = analyze_gps_data.export_lap_csv(session, lap_number)
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
