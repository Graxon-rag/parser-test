from pinecone import AsyncPinecone
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()


async def main():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = AsyncPinecone(api_key=api_key)

    results = await pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=["Apple Inc. has revolutionized the tech industry with its sleek designs."],
        parameters={
            "input_type": "passage",   # use "query" when embedding a search query instead
            "truncate": "END",
            "return_tokens": True      # optional — gives you the actual string tokens alongside sparse indices
        }
    )

    print(results)

    for e in results.data:
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
