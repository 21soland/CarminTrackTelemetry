import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import analyze_gps_data

app = FastAPI()

# Serve the static folder (for index.html, JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


def load_session():
    try:
        return analyze_gps_data.build_session_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load telemetry: {exc}") from exc


processed_session = load_session()


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/api/session")
def get_session():
    return processed_session


@app.get("/api/track")
def get_track():
    return processed_session["track"]


@app.get("/api/telemetry")
def get_telemetry():
    return processed_session["telemetry"]


@app.get("/api/laps")
def get_laps():
    return processed_session["laps"]


@app.get("/api/export/session")
def export_session():
    body = json.dumps(processed_session, indent=2)
    headers = {"Content-Disposition": "attachment; filename=carmintrack_session.json"}
    return PlainTextResponse(body, media_type="application/json", headers=headers)


@app.get("/api/export/lap/{lap_number}")
def export_lap(lap_number: int):
    try:
        csv_body = analyze_gps_data.export_lap_csv(processed_session, lap_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers = {"Content-Disposition": f"attachment; filename=lap_{lap_number}.csv"}
    return PlainTextResponse(csv_body, media_type="text/csv", headers=headers)


# Run with: uvicorn app:app --reload
