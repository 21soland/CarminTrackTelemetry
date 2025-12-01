"""
Dependencies:
  - osmnx
  - geopandas
  - shapely
  - matplotlib

Usage:
  python3 osm.py --data-file "GPS Data/Cheswick Laps.txt"
  python3 osm.py --data-file "GPS Data/Cheswick Laps.txt" --dist 500 --output-dir "osm_validation"
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import osmnx as ox
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from pyproj import Transformer

import analyze_gps_data as agd

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False



def latlon_to_enu(lat: float, lon: float, 
                  ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Convert latitude/longitude to ENU (East-North-Up) coordinates.

    Uses WGS84 ellipsoid for accurate conversion.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        ref_lat: Reference latitude in degrees (origin)
        ref_lon: Reference longitude in degrees (origin)

    Returns:
        (east_m, north_m) in meters
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis (m)
    e2 = 0.00669437999014  # First eccentricity squared
    
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)
    
    # Radius of curvature in prime vertical
    N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
    
    # ENU conversion (simplified for small distances)
    dlat = lat_rad - ref_lat_rad
    dlon = lon_rad - ref_lon_rad
    
    # East (x)
    east = dlon * np.cos(ref_lat_rad) * N_ref
    
    # North (y)
    north = dlat * N_ref
    
    return east, north


def latlon_series_to_enu(lat_series: pd.Series, lon_series: pd.Series,
                         ref_lat: float, ref_lon: float) -> Tuple[pd.Series, pd.Series]:
    """
    Convert series of lat/lon to ENU coordinates.

    Args:
        lat_series: Series of latitudes in degrees
        lon_series: Series of longitudes in degrees
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees

    Returns:
        (east_m, north_m) as pandas Series
    """
    east_list = []
    north_list = []
    
    for lat, lon in zip(lat_series, lon_series):
        if pd.isna(lat) or pd.isna(lon):
            east_list.append(np.nan)
            north_list.append(np.nan)
        else:
            e, n = latlon_to_enu(lat, lon, ref_lat, ref_lon)
            east_list.append(e)
            north_list.append(n)
    
    return pd.Series(east_list, index=lat_series.index), pd.Series(north_list, index=lon_series.index)


def geometry_to_enu(geometry, ref_lat: float, ref_lon: float) -> List[Tuple[float, float]]:
    """
    Convert a shapely geometry (LineString, MultiLineString, etc.) to ENU coordinates.

    Args:
        geometry: Shapely geometry object
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees

    Returns:
        List of (east, north) tuples in meters
    """
    if geometry.geom_type == 'LineString':
        coords = list(geometry.coords)
        enu_coords = []
        for lon, lat in coords:
            e, n = latlon_to_enu(lat, lon, ref_lat, ref_lon)
            enu_coords.append((e, n))
        return enu_coords
    elif geometry.geom_type == 'MultiLineString':
        all_coords = []
        for line in geometry.geoms:
            coords = list(line.coords)
            for lon, lat in coords:
                e, n = latlon_to_enu(lat, lon, ref_lat, ref_lon)
                all_coords.append((e, n))
        return all_coords
    else:
        # For other geometry types, extract coordinates
        coords = list(geometry.coords)
        enu_coords = []
        for coord in coords:
            if len(coord) >= 2:
                lon, lat = coord[0], coord[1]
                e, n = latlon_to_enu(lat, lon, ref_lat, ref_lon)
                enu_coords.append((e, n))
        return enu_coords


def extract_osm_ground_truth(lat_center: float, lon_center: float,
                             dist: float = 500.0,
                             lat_min: float = None, lat_max: float = None,
                             lon_min: float = None, lon_max: float = None) -> Tuple[gpd.GeoDataFrame, str]:
    """
    Extract ground truth road geometry from OpenStreetMap.
    
    Tries multiple methods to find roads:
    1. Bounding box query if trajectory bounds provided (best for freeways)
    2. Point-based query with increasing radius
    3. Custom filter for highways/freeways

    Args:
        lat_center: Center latitude in degrees
        lon_center: Center longitude in degrees
        dist: Initial query radius in meters
        lat_min, lat_max, lon_min, lon_max: Optional bounding box for trajectory

    Returns:
        (edges_gdf, crs_proj) where edges_gdf contains road geometries
        and crs_proj is the projected CRS
    """
    print(f"Downloading OSM road network around ({lat_center:.6f}, {lon_center:.6f})...")
    
    edges_gdf = None
    
    # Method 1: Try bounding box if provided (more reliable for long routes like freeways)
    if all([lat_min is not None, lat_max is not None, lon_min is not None, lon_max is not None]):
        print(f"Trying bounding box method: ({lat_min:.6f}, {lon_min:.6f}) to ({lat_max:.6f}, {lon_max:.6f})")
        try:
            # Create bounding box with some padding
            padding = 0.001  # ~100m padding
            bbox = (lat_min - padding, lon_min - padding, lat_max + padding, lon_max + padding)
            G = ox.graph_from_bbox(bbox[0], bbox[2], bbox[1], bbox[3], network_type="drive")
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            if not edges_gdf.empty:
                print(f"✓ Found {len(edges_gdf)} road segments using bounding box method")
        except Exception as e:
            print(f"  Bounding box method failed: {e}")
    
    # Method 2: Try point-based with increasing radius
    if edges_gdf is None or edges_gdf.empty:
        radii = [dist, dist * 2, dist * 5, dist * 10]  # Try progressively larger radii
        network_types = ["drive", "all"]  # Try different network types
        
        for network_type in network_types:
            for radius in radii:
                try:
                    print(f"Trying point-based query: radius={radius:.0f}m, network_type={network_type}...")
                    G = ox.graph_from_point((lat_center, lon_center),
                                          dist=radius,
                                          network_type=network_type)
                    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                    if not edges_gdf.empty:
                        print(f"✓ Found {len(edges_gdf)} road segments using point-based method (radius={radius:.0f}m, type={network_type})")
                        break
                except Exception as e:
                    print(f"  Point-based query failed (radius={radius:.0f}m, type={network_type}): {e}")
                    continue
            if edges_gdf is not None and not edges_gdf.empty:
                break
    
    # Method 3: Try using a custom filter for highways/freeways
    if edges_gdf is None or edges_gdf.empty:
        print("Trying custom filter for highways/freeways...")
        try:
            # Use a larger bounding box and filter for major roads
            if all([lat_min is not None, lat_max is not None, lon_min is not None, lon_max is not None]):
                padding = 0.01  # ~1km padding
                bbox = (lat_min - padding, lon_min - padding, lat_max + padding, lon_max + padding)
                custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
                G = ox.graph_from_bbox(bbox[0], bbox[2], bbox[1], bbox[3], 
                                      network_type="drive", custom_filter=custom_filter)
                edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                if not edges_gdf.empty:
                    print(f"✓ Found {len(edges_gdf)} road segments using custom highway filter")
        except Exception as e:
            print(f"  Custom filter method failed: {e}")
    
    if edges_gdf is None or edges_gdf.empty:
        raise ValueError(
            f"\nNo OSM road edges found around ({lat_center:.6f}, {lon_center:.6f}).\n"
            f"   Tried multiple methods and radii up to {dist * 10:.0f}m.\n"
            f"   This area may have limited OSM coverage.\n\n"
            f"   Suggestions:\n"
            f"   1. Increase --dist parameter (e.g., --dist 2000 or --dist 5000)\n"
            f"   2. Check if the area has OSM data coverage at openstreetmap.org\n"
            f"   3. The route may be on private roads or unmapped areas"
        )

    print(f"Successfully extracted {len(edges_gdf)} road segments")
    
    # Project to local metric CRS
    edges_proj = ox.projection.project_gdf(edges_gdf)
    crs_proj = edges_proj.crs

    return edges_proj, crs_proj


def load_gps_trajectory(data_file: Path, smooth_method: str = "gaussian", 
                       smooth_window: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], List[Dict]]:
    """
    Load GPS trajectory data (raw and smoothed) along with lap information.
    
    Uses the same lap detection and filtering logic as build_session_payload()
    in analyze_gps_data.py to ensure consistency.
    
    Args:
        data_file: Path to GPS data file
        smooth_method: Smoothing method - "gaussian", "savgol", or "ma". Default "gaussian".
        smooth_window: Window size for smoothing. Default 7.
        
    Returns:
        (df_raw, df_smooth, telemetry, laps) where:
        - df_raw, df_smooth: DataFrames with lat, lon, timestamp
        - telemetry: List of telemetry records with lap annotations
        - laps: List of lap records with start/end indices (filtered and validated)
    """
    # Load using existing pipeline
    parsed = agd.load_gps_data(data_file)
    df_raw = agd.extract_time_series(parsed)
    
    if df_raw.empty:
        raise ValueError(f"Extracted time-series is empty for file: {data_file}")
    
    # Compute smoothed version with specified method
    df_smooth = agd.compute_derived_metrics(df_raw, smooth_position=True, 
                                            smooth_method=smooth_method, 
                                            smooth_window=smooth_window)
    
    # Build telemetry records and detect laps
    telemetry = agd.build_telemetry_records(df_smooth)
    laps = agd.detect_laps(telemetry, data_file)
    
    # Apply the same filtering logic as build_session_payload() in analyze_gps_data.py
    MIN_REAL_LAP_TIME_S = 20.0 
    MIN_REAL_LAP_DISTANCE_M = 150.0

    # If detect_laps returned nothing at all, just use fallback
    if not laps:
        fallback = agd.build_fallback_lap(telemetry)
        laps = [fallback] if fallback else []
    else:
        # Filter out obviously bogus laps with very short or tiny distance
        plausible_laps = []
        for lap in laps:
            t = lap.get("lap_time_s")
            d = lap.get("distance_m")
            if t is None or d is None:
                continue
            if t >= MIN_REAL_LAP_TIME_S and d >= MIN_REAL_LAP_DISTANCE_M:
                plausible_laps.append(lap)

        # If no laps look "real", treat the whole session as one lap
        if not plausible_laps:
            fallback = agd.build_fallback_lap(telemetry)
            laps = [fallback] if fallback else []
        else:
            plausible_laps.sort(key=lambda l: l["start_sample_idx"])

            # Re-number them sequentially
            for i, lap in enumerate(plausible_laps, start=1):
                lap["lap_number"] = i

            # If we have 3+ real laps, treat first & last as partial (out / in)
            if len(plausible_laps) >= 3:
                # For OSM validation, we'll keep all laps but mark partial ones
                # The calling code can handle partial laps if needed
                laps = plausible_laps
            else:
                # 1–2 plausible laps: treat them all as full; no partials
                laps = plausible_laps
    
    # Annotate samples with lap information (same as build_session_payload)
    agd.annotate_lap_samples(telemetry, laps)
    
    # Extract relevant columns
    df_raw = df_raw[["timestamp", "lat", "lon"]].copy()
    df_smooth = df_smooth[["timestamp", "lat", "lon"]].copy()
    
    # Reset indices for alignment
    df_raw = df_raw.reset_index(drop=True)
    df_smooth = df_smooth.reset_index(drop=True)
    
    return df_raw, df_smooth, telemetry, laps


def compute_point_to_road_distances_enu(df: pd.DataFrame,
                                    road_union,
                                       ref_lat: float, ref_lon: float,
                                    crs_proj) -> pd.Series:
    """
    Compute distance from each GPS point to nearest OSM road in ENU coordinates.

    Args:
        df: DataFrame with 'lat' and 'lon' columns
        road_union: Shapely union of OSM road geometries (in projected CRS)
        ref_lat: Reference latitude for ENU conversion
        ref_lon: Reference longitude for ENU conversion
        crs_proj: Projected CRS of road_union

    Returns:
        Series of distances in meters
    """
    # Convert GPS points to projected CRS
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    gdf_points_proj = gdf_points.to_crs(crs_proj)

    # Compute distances
    distances = gdf_points_proj.geometry.apply(lambda geom: geom.distance(road_union))

    return distances


def compute_metrics(distances: pd.Series) -> Dict[str, float]:
    """
    Compute comprehensive accuracy metrics.

    Args:
        distances: Series of distance errors in meters

    Returns:
        Dictionary of metrics
    """
    d = distances.replace([np.inf, -np.inf], np.nan).dropna()
    
    if d.empty:
        return {
            "n_points": 0,
            "mean_m": np.nan,
            "rms_m": np.nan,
            "std_m": np.nan,
            "median_m": np.nan,
            "p50_m": np.nan,
            "p75_m": np.nan,
            "p90_m": np.nan,
            "p95_m": np.nan,
            "p99_m": np.nan,
            "max_m": np.nan,
            "min_m": np.nan,
        }

    return {
        "n_points": int(d.size),
        "mean_m": float(d.mean()),
        "rms_m": float(np.sqrt(np.mean(d ** 2))),
        "std_m": float(d.std()),
        "median_m": float(d.median()),
        "p50_m": float(np.percentile(d, 50)),
        "p75_m": float(np.percentile(d, 75)),
        "p90_m": float(np.percentile(d, 90)),
        "p95_m": float(np.percentile(d, 95)),
        "p99_m": float(np.percentile(d, 99)),
        "max_m": float(d.max()),
        "min_m": float(d.min()),
    }


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"{label} - Accuracy Metrics vs OSM Ground Truth")
    print(f"{'='*60}")
    print(f"Number of points:        {metrics['n_points']:>10}")
    print(f"Mean error [m]:          {metrics['mean_m']:>10.2f}")
    print(f"RMS error [m]:           {metrics['rms_m']:>10.2f}")
    print(f"Std deviation [m]:       {metrics['std_m']:>10.2f}")
    print(f"Median error [m]:        {metrics['median_m']:>10.2f}")
    print(f"75th percentile [m]:     {metrics['p75_m']:>10.2f}")
    print(f"90th percentile [m]:     {metrics['p90_m']:>10.2f}")
    print(f"95th percentile [m]:     {metrics['p95_m']:>10.2f}")
    print(f"99th percentile [m]:     {metrics['p99_m']:>10.2f}")
    print(f"Min error [m]:           {metrics['min_m']:>10.2f}")
    print(f"Max error [m]:           {metrics['max_m']:>10.2f}")
    print(f"{'='*60}")


def plot_enu_overlay(df_raw: pd.DataFrame, df_smooth: pd.DataFrame,
                     edges_gdf: gpd.GeoDataFrame,
                     ref_lat: float, ref_lon: float,
                     distances_raw: pd.Series, distances_smooth: pd.Series,
                     telemetry: List[Dict], laps: List[Dict],
                     output_path: Path):
    """
    Create ENU overlay plots with lap markers and error visualization.

    Args:
        df_raw: Raw GPS trajectory
        df_smooth: Smoothed GPS trajectory
        edges_gdf: OSM road edges GeoDataFrame
        ref_lat: Reference latitude for ENU
        ref_lon: Reference longitude for ENU
        distances_raw: Distance errors for raw GPS
        distances_smooth: Distance errors for smoothed GPS
        telemetry: Telemetry records with lap annotations
        laps: List of lap records
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Convert OSM roads to ENU
    print("Converting OSM roads to ENU coordinates...")
    road_enu_lines = []
    for idx, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            gdf_temp = gpd.GeoDataFrame([1], geometry=[geom], crs=edges_gdf.crs)
            gdf_wgs84 = gdf_temp.to_crs("EPSG:4326")
            wgs84_geom = gdf_wgs84.geometry.iloc[0]
            
            enu_coords = []
            for coord in wgs84_geom.coords:
                lon, lat = coord[0], coord[1]
                e, n = latlon_to_enu(lat, lon, ref_lat, ref_lon)
                enu_coords.append((e, n))
            road_enu_lines.append(enu_coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                gdf_temp = gpd.GeoDataFrame([1], geometry=[line], crs=edges_gdf.crs)
                gdf_wgs84 = gdf_temp.to_crs("EPSG:4326")
                wgs84_geom = gdf_wgs84.geometry.iloc[0]
                enu_coords = []
                for coord in wgs84_geom.coords:
                    lon, lat = coord[0], coord[1]
                    e, n = latlon_to_enu(lat, lon, ref_lat, ref_lon)
                    enu_coords.append((e, n))
                road_enu_lines.append(enu_coords)
    
    # Convert GPS tracks to ENU
    print("Converting GPS tracks to ENU coordinates...")
    east_raw, north_raw = latlon_series_to_enu(df_raw['lat'], df_raw['lon'], ref_lat, ref_lon)
    east_smooth, north_smooth = latlon_series_to_enu(df_smooth['lat'], df_smooth['lon'], ref_lat, ref_lon)
    
    # Extract lap start/end coordinates
    lap_starts_enu = []
    lap_ends_enu = []
    lap_numbers = []
    
    for lap in laps:
        start_idx = lap.get('start_sample_idx', 0)
        end_idx = lap.get('end_sample_idx', len(telemetry) - 1)
        
        if start_idx < len(telemetry) and telemetry[start_idx].get('lat') and telemetry[start_idx].get('lon'):
            lat_start = telemetry[start_idx]['lat']
            lon_start = telemetry[start_idx]['lon']
            e_start, n_start = latlon_to_enu(lat_start, lon_start, ref_lat, ref_lon)
            lap_starts_enu.append((e_start, n_start, lap['lap_number']))
        
        if end_idx < len(telemetry) and telemetry[end_idx].get('lat') and telemetry[end_idx].get('lon'):
            lat_end = telemetry[end_idx]['lat']
            lon_end = telemetry[end_idx]['lon']
            e_end, n_end = latlon_to_enu(lat_end, lon_end, ref_lat, ref_lon)
            lap_ends_enu.append((e_end, n_end, lap['lap_number']))
            lap_numbers.append(lap['lap_number'])
    
    # Plot 1: Full track overlay with lap markers (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot OSM roads (ground truth) - more visible
    for i, line_coords in enumerate(road_enu_lines[:200]):  # Increased limit
        if len(line_coords) > 1:
            e_coords = [c[0] for c in line_coords]
            n_coords = [c[1] for c in line_coords]
            ax1.plot(e_coords, n_coords, 'k-', alpha=0.4, linewidth=1.0, 
                    label='OSM Roads (Ground Truth)' if i == 0 else '')
    
    # Plot smoothed GPS track (primary)
    ax1.plot(east_smooth, north_smooth, 'b-', alpha=0.9, linewidth=2.5, label='Smoothed GPS Track', zorder=3)
    
    # Plot raw GPS track (lighter, behind)
    ax1.plot(east_raw, north_raw, 'r-', alpha=0.4, linewidth=1.0, label='Raw GPS Track', zorder=2)
    
    # Mark lap start/end points
    if lap_starts_enu:
        starts_e = [p[0] for p in lap_starts_enu]
        starts_n = [p[1] for p in lap_starts_enu]
        ax1.scatter(starts_e, starts_n, c='green', s=150, marker='o', 
                   edgecolors='darkgreen', linewidths=2, label='Lap Start/Finish', zorder=5)
        # Annotate lap numbers
        for e, n, lap_num in lap_starts_enu[:10]:  # Limit annotations
            ax1.annotate(f'L{lap_num}', (e, n), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('East [m]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('North [m]', fontsize=12, fontweight='bold')
    ax1.set_title('Track Overlay with Lap Markers (ENU Coordinates)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Distance error distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Histogram with better styling
    bins = np.linspace(0, min(distances_raw.max(), distances_smooth.max(), 20), 40)
    ax2.hist(distances_raw.dropna(), bins=bins, alpha=0.6, color='red', 
            label=f'Raw GPS (μ={distances_raw.mean():.2f}m)', density=True, edgecolor='black', linewidth=0.5)
    ax2.hist(distances_smooth.dropna(), bins=bins, alpha=0.6, color='blue', 
            label=f'Smoothed GPS (μ={distances_smooth.mean():.2f}m)', density=True, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for mean
    ax2.axvline(distances_raw.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(distances_smooth.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Distance Error [m]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Error along track (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Compute distance along track for x-axis
    distance_along = np.cumsum(np.sqrt(np.diff(east_smooth)**2 + np.diff(north_smooth)**2))
    distance_along = np.concatenate([[0], distance_along])
    
    ax3.plot(distance_along, distances_raw, 'r-', alpha=0.6, linewidth=1.5, label='Raw GPS Error')
    ax3.plot(distance_along, distances_smooth, 'b-', alpha=0.8, linewidth=2, label='Smoothed GPS Error')
    ax3.fill_between(distance_along, 0, distances_smooth, alpha=0.2, color='blue')
    
    ax3.set_xlabel('Distance Along Track [m]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Distance Error [m]', fontsize=11, fontweight='bold')
    ax3.set_title('Error vs Distance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Error vs Time (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    
    elapsed_s = (df_smooth['timestamp'] - df_smooth['timestamp'].iloc[0]).dt.total_seconds()
    
    ax4.plot(elapsed_s, distances_raw, 'r-', alpha=0.6, linewidth=1.5, label='Raw GPS Error')
    ax4.plot(elapsed_s, distances_smooth, 'b-', alpha=0.8, linewidth=2, label='Smoothed GPS Error')
    
    # Mark lap boundaries
    for lap in laps:
        if 'start_time' in lap:
            start_time = lap['start_time']
            # Handle both Timestamp and string formats
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(start_time, pd.Timestamp):
                t_start = (start_time - df_smooth['timestamp'].iloc[0]).total_seconds()
                ax4.axvline(t_start, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Lap Boundary' if lap == laps[0] else '')
    
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Distance Error [m]', fontsize=11, fontweight='bold')
    ax4.set_title('Error vs Time (Lap Boundaries)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 5: Box plot comparison (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    data_to_plot = [distances_raw.dropna().values, distances_smooth.dropna().values]
    bp = ax5.boxplot(data_to_plot, labels=['Raw GPS', 'Smoothed GPS'], 
                     patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('Distance Error [m]', fontsize=11, fontweight='bold')
    ax5.set_title('Error Statistics Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.suptitle('OSM Validation: GPS Accuracy Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Validate GPS/IMU data against OSM ground truth geometry"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to GPS data file (default: uses analyze_gps_data.DEFAULT_DATA_FILE)"
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=500.0,
        help="OSM query radius in meters (default: 500)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="osm_validation",
        help="Output directory for results (default: osm_validation)"
    )
    parser.add_argument(
        "--smooth-method",
        type=str,
        default="gaussian",
        choices=["gaussian", "savgol", "ma"],
        help="Smoothing method: 'gaussian' (default, best for GPS), 'savgol' (best for sharp turns, requires scipy), or 'ma' (simple moving average)"
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size for smoothing filter (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load GPS data
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        data_file = Path(agd.DEFAULT_DATA_FILE)
    print(f"Loading GPS data from: {data_file}")
    print(f"Using smoothing method: {args.smooth_method} (window={args.smooth_window})")
    df_raw, df_smooth, telemetry, laps = load_gps_trajectory(
        data_file, 
        smooth_method=args.smooth_method,
        smooth_window=args.smooth_window
    )
    print(f"Loaded {len(df_raw)} GPS points")
    print(f"Detected {len(laps)} lap(s)")
    
    # Get reference point (track start or center of trajectory)
    # Try to get from track_start.csv first
    ref_lat = None
    ref_lon = None
    
    track_start_file = Path("GPS Data/track_start.csv")
    if track_start_file.exists():
        track_starts = pd.read_csv(track_start_file)
        # Strip whitespace from column names
        track_starts.columns = track_starts.columns.str.strip()
        
        # Handle different column name formats
        file_col = 'File Name' if 'File Name' in track_starts.columns else 'File'
        lat_col = 'Lat' if 'Lat' in track_starts.columns else 'Latitude'
        lon_col = 'Long' if 'Long' in track_starts.columns else 'Longitude'
        
        match = track_starts[track_starts[file_col] == data_file.name]
        if not match.empty:
            ref_lat = float(str(match.iloc[0][lat_col]).strip().rstrip('s'))
            ref_lon = float(str(match.iloc[0][lon_col]).strip().rstrip('s'))
            print(f"Using track start from CSV: ({ref_lat:.6f}, {ref_lon:.6f})")
    
    # Fallback to center of trajectory
    if ref_lat is None or ref_lon is None:
        valid_lat = df_smooth['lat'].dropna()
        valid_lon = df_smooth['lon'].dropna()
        if not valid_lat.empty and not valid_lon.empty:
            ref_lat = valid_lat.mean()
            ref_lon = valid_lon.mean()
            print(f"Using trajectory center as reference: ({ref_lat:.6f}, {ref_lon:.6f})")
        else:
            raise ValueError("Could not determine reference point")
    
    # Extract OSM ground truth
    # Get trajectory bounds for better OSM query (especially important for freeways)
    valid_lat = df_smooth['lat'].dropna()
    valid_lon = df_smooth['lon'].dropna()
    lat_min = valid_lat.min() if not valid_lat.empty else None
    lat_max = valid_lat.max() if not valid_lat.empty else None
    lon_min = valid_lon.min() if not valid_lon.empty else None
    lon_max = valid_lon.max() if not valid_lon.empty else None
    
    edges_gdf, crs_proj = extract_osm_ground_truth(
        ref_lat, ref_lon, 
        dist=args.dist,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max
    )
    
    # Build road union for distance computation
    print("Building road union for distance computation...")
    road_union = unary_union(edges_gdf.geometry.values)
    
    # Compute distances
    print("Computing distance errors...")
    distances_raw = compute_point_to_road_distances_enu(
        df_raw, road_union, ref_lat, ref_lon, crs_proj
    )
    distances_smooth = compute_point_to_road_distances_enu(
        df_smooth, road_union, ref_lat, ref_lon, crs_proj
    )
    
    # Compute metrics
    metrics_raw = compute_metrics(distances_raw)
    metrics_smooth = compute_metrics(distances_smooth)
    
    # Print metrics
    print_metrics("RAW GPS", metrics_raw)
    print_metrics("SMOOTHED GPS", metrics_smooth)
    
    # Create visualization (include filter method in filename for comparison)
    plot_path = output_dir / f"{data_file.stem}_{args.smooth_method}_enu_overlay.png"
    print(f"\nCreating comprehensive ENU overlay plot...")
    plot_enu_overlay(
        df_raw, df_smooth, edges_gdf,
        ref_lat, ref_lon,
        distances_raw, distances_smooth,
        telemetry, laps,
        plot_path
    )
    
    # Save detailed results to CSV (include filter method in filename)
    results_df = pd.DataFrame({
        'timestamp': df_raw['timestamp'],
        'lat_raw': df_raw['lat'],
        'lon_raw': df_raw['lon'],
        'lat_smooth': df_smooth['lat'],
        'lon_smooth': df_smooth['lon'],
        'dist_error_raw_m': distances_raw,
        'dist_error_smooth_m': distances_smooth,
    })
    
    csv_path = output_dir / f"{data_file.stem}_{args.smooth_method}_validation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to: {csv_path}")
    
    # Save summary metrics (include filter method in filename)
    summary = pd.DataFrame({
        'Metric': ['n_points', 'mean_m', 'rms_m', 'std_m', 'median_m', 
                   'p75_m', 'p90_m', 'p95_m', 'p99_m', 'min_m', 'max_m'],
        'Raw_GPS': [metrics_raw[k] for k in ['n_points', 'mean_m', 'rms_m', 'std_m', 
                                               'median_m', 'p75_m', 'p90_m', 'p95_m', 
                                               'p99_m', 'min_m', 'max_m']],
        'Smoothed_GPS': [metrics_smooth[k] for k in ['n_points', 'mean_m', 'rms_m', 'std_m',
                                                      'median_m', 'p75_m', 'p90_m', 'p95_m',
                                                      'p99_m', 'min_m', 'max_m']],
    })
    
    summary_path = output_dir / f"{data_file.stem}_{args.smooth_method}_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved metrics summary to: {summary_path}")
    
    print(f"\n{'='*60}")
    print("OSM Validation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
