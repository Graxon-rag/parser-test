from deepgram import AsyncDeepgramClient
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os


load_dotenv()


async def main():
    try:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        client = AsyncDeepgramClient(api_key=api_key)

        audio_file_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a")
        response = await client.listen.v1.media.transcribe_file(
            request=audio_file_path.read_bytes(),
            model="nova-3",
            smart_format=True,
            diarize=True,       # Identifies different speakers (0, 1, 2, etc.)
            utterances=True     # Groups the transcript logically by speaker changes
        )

        output_path = Path(f"/home/avvk/Graxon/Graxon/parser/output/deepgram/{audio_file_path.stem}.json")
        output_path.write_bytes(response.json().encode("utf-8"))
        print(f"Deepgram transcription saved to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to process audio file: {e}")


asyncio.run(main())
