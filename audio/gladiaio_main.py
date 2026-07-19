from .model import Transcript, Utterance, Word
from gladiaio_sdk import GladiaClient
from dotenv import load_dotenv
from pathlib import Path
from typing import List
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

    final_utterances: List[Utterance] = []
    duration = transcription.result and transcription.result.metadata.audio_duration
    language: str | None = None

    if transcription.result and transcription.result.transcription:
        transcription = transcription.result.transcription
        if language is None:
            language = transcription.languages[0]

        for utt in transcription.utterances:
            words = utt.words
            if not words:
                continue

            utt_words: List[Word] = []
            for word in words:
                w: Word = Word(
                    text=word.word,
                    start=word.start,
                    end=word.end,
                    confidence=word.confidence,
                    speaker=str(utt.speaker) if utt.speaker is not None and utt.speaker >= 0 else "unknown",
                )
                utt_words.append(w)

            trans_utt = Utterance(
                text=utt.text,
                start=utt.start,
                end=utt.end,
                speaker=str(utt.speaker) if utt.speaker is not None and utt.speaker >= 0 else "unknown",
                confidence=utt.confidence,
                words=utt_words,
            )
            final_utterances.append(trans_utt)

    transcript = Transcript(
        provider="gladia",
        utterances=final_utterances,
        duration=duration,
        language=language
    )

    return transcript

if __name__ == "__main__":
    result = asyncio.run(main())
    print(result.model_dump_json())
