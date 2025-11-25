GarminTrack Telemetry
=====================

Project Structure
-----------------
- `app.py` – FastAPI server that pre-processes the latest log and exposes JSON APIs plus the static front-end.
- `analyze_gps_data.py` – Parser + fusion logic that ingests `GPS Data/Parsable Data.txt`, computes derived telemetry, and outputs GeoJSON + lap stats.
- `static/index.html` – Leaflet UI with a telemetry HUD and lap list.
- Lap selector panel stacks per-lap overlays so you can toggle individual laps while still seeing the reference trajectory.
- Lap overlays inherit the legacy speed-coloring (blue→green→yellow→orange→red) so you can visually scan braking/throttle regions along each lap, and the on-map legend reminds viewers what each hue means.
- Lap delta HUD compares any lap against the fastest lap and shows live time gain/loss wherever you hover.
- Corner list auto-detects high lateral-G segments and lets you highlight them with entry/min/exit speeds plus apex markers.
- Heatmap toggles paint braking, throttle, or lateral load intensities as soft overlays.
- Export buttons generate shareable JSON for the whole session or CSV for any lap.
- `GPS Data/` – Raw files exported from Android GNSS Logger (`headers.txt` + data dumps).

Setup
-----
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" numpy pandas
```

Running Locally
---------------
```bash
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000/ in a browser.

Usage Notes
-----------
- Hover directly over the rendered trajectory to drag the telemetry cursor. The HUD will update with GNSS + IMU values for the nearest sample.
- Lap times and sector splits show up in the panel once the parser detects start/finish crossings (re-entering the start zone with sufficient elapsed time).
- To process a different log, replace `GPS Data/Parsable Data.txt` (and its headers if needed) and restart the FastAPI server.