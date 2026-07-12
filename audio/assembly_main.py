from dotenv import load_dotenv
from pathlib import Path
import assemblyai as aai
import asyncio
import json
import os

load_dotenv()


async def main():
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    audio_file_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a")

    config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True, speech_model=aai.SpeechModel.universal)
    transcriber = aai.Transcriber()

    future = transcriber.transcribe_async(
        str(audio_file_path),
        config=config,
    )

    transcript = await asyncio.wrap_future(future)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Error: {transcript.error}")
        exit(1)

    output_dir = Path("/home/avvk/Graxon/Graxon/parser/output/assembly")
    output_dir.mkdir(parents=True, exist_ok=True) 

    output_path = output_dir / f"{audio_file_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript.json_response, f, indent=4)

    print(f"Assembly transcription saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
