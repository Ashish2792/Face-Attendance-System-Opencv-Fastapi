from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
import os



ATT_FILE = "attendance_v2.csv"

app = FastAPI(title="Face Attendance Live API")

@app.get("/attendance")
def get_attendance():
    df = pd.read_csv(ATT_FILE)
    return df.to_dict(orient="records")

@app.get("/live-status")
def live_status():
    df = pd.read_csv(ATT_FILE)

    if df.empty:
        return []

    df["state"] = df["state"].astype(int)
    latest = (
        df.sort_values("timestamp")
          .groupby(["name", "id"])
          .tail(1)
    )

    # Currently IN users
    in_users = latest[latest["state"] == 1]
    return in_users.to_dict(orient="records")

@app.get("/")
def root():
    return {
        "message": "Face Attendance API is running",
        "endpoints": ["/attendance", "/live-status", "/docs"]
    }

@app.get("/export/excel")
def export_excel():
    if not os.path.exists(ATT_FILE):
        return {"error": "Attendance file not found"}

    df = pd.read_csv(ATT_FILE)

    # Sort properly
    df = df.sort_values(by=["date", "timestamp"])

    excel_file = "attendance_export.xlsx"
    df.to_excel(excel_file, index=False)

    return FileResponse(
        excel_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=excel_file
    )