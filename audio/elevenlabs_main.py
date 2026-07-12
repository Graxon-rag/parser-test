from elevenlabs.client import AsyncElevenLabs
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os


load_dotenv()


async def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    timeout = 60 * 10  # 10 minutes

    audio_file_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a")
    base_url = "https://api.in.residency.elevenlabs.io"

    client = AsyncElevenLabs(api_key=api_key, timeout=timeout, base_url=base_url)

    transcription = await client.speech_to_text.convert(
        # model_id="scribe_v1",
        model_id="scribe_v2",
        file=audio_file_path.read_bytes(),
        tag_audio_events=True,  # Tag audio events like laughter, applause, etc.
        diarize=True,  # Whether to annotate who is speaking
    )

    output_dir = Path("/home/avvk/Graxon/Graxon/parser/output/elevenlabs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{audio_file_path.stem}.json"
    output_path.write_bytes(transcription.json().encode("utf-8"))

    print(f"ElevenLabs transcription saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
