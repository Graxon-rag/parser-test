from gladiaio_sdk import GladiaClient
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os


load_dotenv()


async def main():
    api_key = os.getenv("GLADIA_API_KEY")

    audio_file_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a")
    timeout = 60 * 10  # 10 minutes

    client = GladiaClient(api_key=api_key).pre_recorded_async()
    transcription = await client.transcribe(
        audio_file_path,
        options={
            "model": "solaria-3",
            "diarization": True,
        },
        timeout=timeout,
    )

    output_dir = Path("/home/avvk/Graxon/Graxon/parser/output/gladiaio")
    output_dir.mkdir(parents=True, exist_ok=True) 

    output_path = output_dir / f"{audio_file_path.stem}.json"
    output_path.write_bytes(transcription.to_json().encode("utf-8"))

    print(f"Gladiaio transcription saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
