from .model import Transcript, Utterance, Word
from dotenv import load_dotenv
from groq import AsyncGroq
from pathlib import Path
from typing import List
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

    final_utterances: List[Utterance] = []
    print(f"Grok transcription saved to {output_path}")

    if isinstance(translation, dict):
        segments = translation.get("segments", [])
        duration = translation.get("duration", None)
    else:
        segments = getattr(translation, "segments", [])
        duration = getattr(translation, "duration", None)

    for seg in segments:
        # If seg is a dictionary (which it usually is for verbose_json)
        if isinstance(seg, dict):
            text = seg.get("text", "")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            confidence = seg.get("no_speech_prob", None)
        # If seg is an object
        else:
            text = getattr(seg, "text", "")
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
            confidence = getattr(seg, "no_speech_prob", None)

        utt = Utterance(text=text, start=start, end=end, confidence=confidence)
        final_utterances.append(utt)

    transcript = Transcript(utterances=final_utterances, provider="whisper", duration=duration, source_file=str(audio_file_path))

    return transcript
if __name__ == "__main__":
    result = asyncio.run(main())
    if result is None:
        exit(1)
    print(result.model_dump_json())
