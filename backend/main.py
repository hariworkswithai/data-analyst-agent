"""FastAPI entrypoint for Multi-Agent AI Data Analyst."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.config import BASE_DIR, CHART_DIR, REPORT_DIR

load_dotenv(BASE_DIR / "backend" / ".env")
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Multi-Agent AI Data Analyst", version="1.0.0")
app.include_router(router)

app.mount("/charts", StaticFiles(directory=CHART_DIR), name="charts")
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)