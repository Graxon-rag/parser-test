from .model import Transcript, Utterance, Word
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import assemblyai as aai
import asyncio
import json
import os

load_dotenv()


async def main():
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    audio_file_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/test_recording.m4a")

    config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True)
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

    if transcript.utterances is None:
        print("No utterances found in the transcript.")
        return

    utterances: List[Utterance] = []
    for utterance in transcript.utterances:
        words: List[Word] = []
        for word in utterance.words:
            words.append(Word(
                text=word.text,
                start=word.start,
                end=word.end,
                confidence=word.confidence,
                speaker=word.speaker,
            ))

        utterances.append(Utterance(
            text=utterance.text,
            start=utterance.start,
            end=utterance.end,
            speaker=utterance.speaker or None,
            confidence=utterance.confidence,
            words=words,
        ))

    return Transcript(
        provider="assemblyai",
        utterances=utterances,
        language=transcript.language_code,
        source_file=str(audio_file_path),
        duration=transcript.audio_duration
    )

if __name__ == "__main__":
    result = asyncio.run(main())
    if result is None:
        exit(1)
    print(result.model_dump_json())
