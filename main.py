# main.py
"""FastAPI app tying M0-M4 together. Single-container deploy target."""
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import sqlite3

from semantic import profile_dataframe
from analytics import AnalyticsEngine
from detection import detect_anomaly, detect_trend_break, root_cause
from insight import build_insight, narrate_insight

app = FastAPI(title="AI-BI Lab")

DATA_DIR = Path("/data/uploads")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "/data/metadata.db"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            path TEXT,
            rows INTEGER,
            cols INTEGER,
            catalog_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()
    print("[AI-DEBUG] init_db: tables ensured")


init_db()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = DATA_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[AI-DEBUG] upload_file: filename={file.filename} saved_to={dest}")

    try:
        if dest.suffix == '.csv':
            df = pd.read_csv(dest)
        elif dest.suffix in ('.xlsx', '.xls'):
            df = pd.read_excel(dest)
        else:
            raise HTTPException(400, f"Unsupported file type: {dest.suffix}")
    except Exception as e:
        print(f"[AI-DEBUG] upload_file: parse_error={e}")
        raise HTTPException(400, f"Failed to parse file: {e}")

    catalog = profile_dataframe(df)

    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO datasets (filename, path, rows, cols, catalog_json) VALUES (?, ?, ?, ?, ?)",
        (file.filename, str(dest), len(df), len(df.columns), pd.io.json.dumps(catalog)),
    )
    dataset_id = cur.lastrowid
    con.commit()
    con.close()

    print(f"[AI-DEBUG] upload_file: dataset_id={dataset_id} rows={len(df)} cols={len(df.columns)}")

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "profile": catalog,
    }


def _get_dataset(dataset_id: int):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT path, catalog_json FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(404, "Dataset not found")
    path, catalog_json = row
    catalog = pd.io.json.loads(catalog_json)
    return path, catalog


@app.get("/datasets/{dataset_id}/summary")
def summary(dataset_id: int, metric: str, dimension: str = None):
    path, catalog = _get_dataset(dataset_id)
    engine = AnalyticsEngine(path, catalog)

    result = {"total": engine.total(metric)}
    if dimension:
        result["by_dimension"] = engine.by_dimension(metric, dimension).to_dict(orient="records")
        result["contribution"] = engine.contribution(metric, dimension).to_dict(orient="records")

    print(f"[AI-DEBUG] /summary: dataset_id={dataset_id} metric={metric} dim={dimension}")
    return result


@app.get("/datasets/{dataset_id}/trend")
def trend(dataset_id: int, metric: str, granularity: str = "month"):
    path, catalog = _get_dataset(dataset_id)
    engine = AnalyticsEngine(path, catalog)
    ts = engine.mom_yoy(metric)
    print(f"[AI-DEBUG] /trend: dataset_id={dataset_id} metric={metric} gran={granularity}")
    return ts.to_dict(orient="records")


@app.get("/datasets/{dataset_id}/insight")
async def insight(dataset_id: int, metric: str, dimensions: str):
    """dimensions: comma-separated, e.g. 'Region,Product'"""
    path, catalog = _get_dataset(dataset_id)
    engine = AnalyticsEngine(path, catalog)
    dims = [d.strip() for d in dimensions.split(",")]

    ts = engine.mom_yoy(metric)
    if len(ts) < 2:
        raise HTTPException(400, "Not enough time periods for insight")

    current = ts.iloc[-1][metric]
    prior = ts.iloc[-2][metric]

    rc = root_cause(engine, metric, dims)
    insight_obj = build_insight(metric, current, prior, rc["drivers"])

    narrative = "LLM not configured"
    if ANTHROPIC_API_KEY:
        narrative = await narrate_insight(insight_obj, ANTHROPIC_API_KEY)

    print(f"[AI-DEBUG] /insight: dataset_id={dataset_id} metric={metric} severity={insight_obj['severity']}")

    return {**insight_obj, "narrative": narrative}


@app.get("/health")
def health():
    return {"status": "ok"}
