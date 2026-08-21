from fastapi import FastAPI
from fastapi.responses import FileResponse

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
