from dotenv import load_dotenv
from pathlib import Path
from groq import AsyncGroq
import asyncio
import os


load_dotenv()


async def main():
    api_key = os.getenv("GROQ_API_KEY")

    timeout = 60 * 10  # 10 minutes
    max_retry = 3
    client = AsyncGroq(api_key=api_key, timeout=timeout, max_retries=max_retry)

    audio_file_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a")

    translation = await client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=(
            audio_file_path.name,
            audio_file_path.read_bytes(),
        ),
        response_format="verbose_json",
        timestamp_granularities=["segment", "word"],
    )

    output_dir = Path("/home/avvk/Graxon/Graxon/parser/output/grok")
    output_dir.mkdir(parents=True, exist_ok=True) 

    output_path = output_dir / f"{audio_file_path.stem}.json"
    output_path.write_bytes(translation.model_dump_json().encode("utf-8"))

    print(f"Grok transcription saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
