import asyncio
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv("backend/.env")

from backend.services.llm import LLMService, try_parse_json


async def main():
    llm = LLMService()
    reply = await llm.chat(
        'You reply with exactly: {"ok": true}', "Reply now.", max_tokens=60
    )
    print("CHAT OK:", reply[:120])
    print("JSON:", try_parse_json(reply))
    await llm.close()


asyncio.run(main())