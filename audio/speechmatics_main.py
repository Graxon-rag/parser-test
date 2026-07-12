from speechmatics.batch import AsyncClient, TranscriptionConfig
from dotenv import load_dotenv
from dataclasses import asdict
from pathlib import Path
import asyncio
import json
import os

load_dotenv()


async def main():
    api_key = os.environ["SPEECHMATICS_API_KEY"]

    audio_file_path = Path(
        "/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a"
    )

    output_dir = Path("/home/avvk/Graxon/Graxon/parser/output/speechmatics")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TranscriptionConfig(
        diarization="speaker",
    )

    client = AsyncClient(api_key=api_key)

    try:
        result = await client.transcribe(
            audio_file=str(audio_file_path),
            transcription_config=config,
        )

        output_path = output_dir / f"{audio_file_path.stem}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(result, str):
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        print(f"Speechmatics transcription saved to {output_path}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
