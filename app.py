from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

import pandas as pd
import io
import re


app = FastAPI(title="AI-BI Lab")


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return FileResponse("index.html")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "AI-BI Lab is running"
    }


# =========================================================
# SEMANTIC INFERENCE
# =========================================================

def infer_column_role(column_name, series):

    name = column_name.lower().strip()

    dtype = str(series.dtype)

    unique_count = series.nunique()

    row_count = len(series)

    unique_ratio = (
        unique_count / row_count
        if row_count > 0
        else 0
    )


    # -----------------------------------------------------
    # 1. DATETIME
    # -----------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(series):

        return {
            "role": "time",
            "data_type": "datetime",
            "aggregation": None,
            "confidence": 1.0
        }


    # -----------------------------------------------------
    # 2. NUMERIC
    # -----------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        # Detect ID-like numeric columns
        id_keywords = [
            "id",
            "code",
            "key"
        ]

        if (
            any(keyword in name for keyword in id_keywords)
            and unique_ratio > 0.5
        ):

            return {
                "role": "identifier",
                "data_type": "numeric",
                "aggregation": None,
                "confidence": 0.95
            }


        # Metric
        return {
            "role": "metric",
            "data_type": "numeric",
            "aggregation": "sum",
            "confidence": 0.90
        }


    # -----------------------------------------------------
    # 3. TEXT
    # -----------------------------------------------------

    if pd.api.types.is_string_dtype(series):

        # Identifier-like columns
        id_keywords = [
            "id",
            "code",
            "key"
        ]

        if (
            any(keyword in name for keyword in id_keywords)
            and unique_ratio > 0.5
        ):

            return {
                "role": "identifier",
                "data_type": "text",
                "aggregation": None,
                "confidence": 0.90
            }


        # Low cardinality → dimension
        if unique_ratio < 0.20:

            return {
                "role": "dimension",
                "data_type": "categorical",
                "aggregation": None,
                "confidence": 0.90
            }


        # High cardinality text
        return {
            "role": "dimension",
            "data_type": "text",
            "aggregation": None,
            "confidence": 0.70
        }


    # -----------------------------------------------------
    # 4. FALLBACK
    # -----------------------------------------------------

    return {
        "role": "unknown",
        "data_type": dtype,
        "aggregation": None,
        "confidence": 0.30
    }


# =========================================================
# FILE UPLOAD
# =========================================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):

    content = await file.read()

    print(
        f"UPLOAD DEBUG | "
        f"name={file.filename} | "
        f"size={len(content)} | "
        f"first_bytes={content[:20]}"
    )


    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    if file.filename.lower().endswith(".csv"):

        df = pd.read_csv(
            io.BytesIO(content)
        )

    elif file.filename.lower().endswith(
        (".xlsx", ".xls")
    ):

        df = pd.read_excel(
            io.BytesIO(content)
        )

    else:

        return {
            "success": False,
            "error": "Only CSV and Excel files are supported."
        }


    # -----------------------------------------------------
    # PROFILE
    # -----------------------------------------------------

    columns = []


    for column in df.columns:

        series = df[column]

        semantic = infer_column_role(
            str(column),
            series
        )


        columns.append({

            "name": str(column),

            "dtype": str(
                series.dtype
            ),

            "missing": int(
                series.isna().sum()
            ),

            "missing_pct": round(
                series.isna().mean() * 100,
                2
            ),

            "unique": int(
                series.nunique()
            ),

            "semantic": semantic

        })


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {

        "success": True,

        "filename": file.filename,

        "rows": len(df),

        "columns": len(df.columns),

        "profile": columns

    }
