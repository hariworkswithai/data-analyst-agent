"""Integration test: upload -> SSE stream -> done.

Usage: python scripts/sse_test.py [port]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from httpx import AsyncClient

PORT = sys.argv[1] if len(sys.argv) > 1 else "8010"
BASE = f"http://127.0.0.1:{PORT}"
SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "sales.csv"


async def main():
    timeout = httpx.Timeout(300.0, connect=10.0)
    async with AsyncClient(timeout=timeout) as client:
        with SAMPLE.open("rb") as fh:
            resp = await client.post(
                f"{BASE}/api/analyze",
                files={"file": ("sales.csv", fh, "text/csv")},
                data={"question": "Analyze sales by region and product."},
            )
        print("POST /api/analyze ->", resp.status_code)
        body = resp.json()
        wf = body.get("workflow_id")
        print("workflow_id:", wf)

        state_events = 0
        statuses = []
        async with client.stream("GET", f"{BASE}/api/workflow/{wf}/events") as stream:
            print("SSE status:", stream.status_code)
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    import json

                    evt = json.loads(line[6:])
                    e = evt.get("event")
                    if e == "state":
                        state_events += 1
                        st = evt["data"].get("status")
                        if st not in statuses:
                            statuses.append(st)
                    elif e == "agent_status":
                        statuses.append(f"agent:{evt['data']['agent']}={evt['data']['status']}")
                    elif e == "activity":
                        statuses.append(f"activity:{evt['data']['message']}")
                    elif e == "done":
                        print("SSE done event:", evt["data"].get("status"))
                        break

        final = (await client.get(f"{BASE}/api/workflow/{wf}")).json()
        print("\nstate_events received:", state_events)
        print("statuses seen:", statuses)
        print("\nFINAL STATUS:", final["status"])
        print("review_status:", final["review_status"])
        print("charts:", len(final.get("charts", [])))
        print("report length:", len(final.get("final_report", "")))
        assert final["status"] == "completed"
        assert state_events > 0
        print("\nSSE_TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())