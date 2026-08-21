from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import io

app = FastAPI(title="AI-BI Lab")


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "AI-BI Lab is running"
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):

    # Read uploaded file into memory
    content = await file.read()

    # Debug uploaded file
    file_size = len(content)

    print(
        f"UPLOAD DEBUG | "
        f"name={file.filename} | "
        f"size={file_size} | "
        f"first_bytes={content[:20]}"
    )

    # Detect file type
    if file.filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))

    elif file.filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))

    else:
        return {
            "success": False,
            "error": "Only CSV and Excel files are supported."
        }

    # Basic profiling
    columns = []

    for column in df.columns:
        columns.append({
            "name": str(column),
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "missing_pct": round(
                df[column].isna().mean() * 100, 2
            ),
            "unique": int(df[column].nunique())
        })

    return {
        "success": True,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "profile": columns
    }
