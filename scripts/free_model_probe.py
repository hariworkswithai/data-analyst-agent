"""Find the best free OpenRouter model for structured tool-planning JSON.

Usage: python scripts/free_model_probe.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")

API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.getenv("OPENROUTER_API_KEY", "")
if not KEY:
    raise SystemExit("Set the OPENROUTER_API_KEY environment variable first.")

CANDIDATES = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "z-ai/glm-5.2:free",
    "minimax/minimax-m3:free",
    "openrouter/free",
]

SYSTEM = "You are an AI data-analyst planner. Reply with ONLY valid JSON, no markdown."
USER = """Given this dataset profile, pick tools and their arguments.
Dataset: 6 columns - Date, Region, Product, Sales, Profit, Quantity.
Respond as: {"plan":["tool_name"], "note":"why" }"""


async def probe(client: httpx.AsyncClient, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        "max_tokens": 120,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(API, json=payload, headers={"Authorization": f"Bearer {KEY}"})
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return {"model": model, "status": r.status_code, "time": round(dt, 2), "err": r.text[:160]}
        content = r.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            ok = bool(parsed.get("plan"))
        except json.JSONDecodeError:
            parsed = None
            ok = False
        return {"model": model, "status": 200, "time": round(dt, 2), "valid_json": ok, "plan": parsed.get("plan") if parsed else None}
    except Exception as e:  # noqa: BLE001
        return {"model": model, "status": "exc", "err": str(e)[:160]}


async def main():
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[probe(client, m) for m in CANDIDATES])
    for res in results:
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())