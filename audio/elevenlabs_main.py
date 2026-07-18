from elevenlabs.types.speech_to_text_chunk_response_model import SpeechToTextChunkResponseModel
from .model import Transcript, Utterance, Word
from elevenlabs.client import AsyncElevenLabs
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import asyncio
import math
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
    final_utterances: List[Utterance] = []
    current_speaker = None
    current_words = []
    if isinstance(transcription, SpeechToTextChunkResponseModel):
        for word_obj in (transcription.words):
            if word_obj.type == "spacing":
                continue
            speaker = word_obj.speaker_id
            if not speaker:
                speaker = "unknown"
            if current_speaker is None:
                current_speaker = speaker

            if speaker != current_speaker:
                if current_words:
                    final_utterances.append(create_utterance(current_words, current_speaker))

                # Reset for the new speaker
                current_speaker = speaker
                current_words = []

            logprob = word_obj.logprob
            confidence = round(math.exp(logprob), 4) if logprob is not None else None

            current_words.append(Word(
                text=word_obj.text.strip(),
                start=word_obj.start or 0,
                end=word_obj.end or 0,
                confidence=confidence,
                speaker=speaker
            ))

        if current_words:
            final_utterances.append(create_utterance(current_words, current_speaker or "unknown"))

    return Transcript(
        provider="elevenlabs",
        language=getattr(transcription, "language_code", None),
        utterances=final_utterances,
        duration=getattr(transcription, "audio_duration_secs", None)
    )


def create_utterance(words: List[Word], speaker: str) -> Utterance:
    """Helper function to create an Utterance object from a list of Words."""
    text = " ".join([w.text for w in words])
    start_time = words[0].start
    end_time = words[-1].end

    return Utterance(
        text=text,
        start=start_time,
        end=end_time,
        speaker=speaker,
        words=words
    )


if __name__ == "__main__":
    result = asyncio.run(main())
    if result is None:
        exit(1)
    print(result.model_dump_json())
