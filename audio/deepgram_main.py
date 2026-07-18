from deepgram.types.listen_v1response import ListenV1Response
from .model import Transcript, Utterance, Word
from deepgram import AsyncDeepgramClient
from dotenv import load_dotenv
from typing import List
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
        duration: float | None = None

        final_utterances = []
        if isinstance(response, ListenV1Response):
            utterances = response.results.utterances
            duration = response.metadata.duration

            if utterances:
                for utt in utterances:
                    words = utt.words
                    if not words:
                        continue
                    current_speaker = None
                    current_words: List[Word] = []
                    for raw_word in words:
                        # Ensure speaker is a string (e.g., "0" instead of 0)
                        word_speaker = str(raw_word.speaker)
                        # Initialize the first speaker
                        if current_speaker is None:
                            current_speaker = word_speaker

                        if word_speaker != current_speaker:
                            if current_words:
                                final_utterances.append(create_utterance(current_words, current_speaker))

                            # Reset for the new speaker
                            current_speaker = word_speaker
                            current_words = []
                        # Add the current word to our running list
                        # Fallback to "word" if "punctuated_word" is missing
                        text = raw_word.punctuated_word or raw_word.word or ""

                        current_words.append(Word(
                            text=text,
                            start=raw_word.start or 0,
                            end=raw_word.end or 0,
                            confidence=raw_word.confidence,
                            speaker=word_speaker
                        ))

                    if current_words:
                        final_utterances.append(create_utterance(current_words, current_speaker or "N/A"))

        return Transcript(
            provider="deepgram",
            utterances=final_utterances,
            source_file=str(audio_file_path),
            duration=duration
        )
    except Exception as e:
        raise RuntimeError(f"Failed to process audio file: {e}")


def create_utterance(words: List[Word], speaker: str) -> Utterance:
    """Helper function to create an Utterance object from a list of Words."""
    # Reconstruct the text by joining the words
    text = " ".join([w.text for w in words])
    # The start time is the start of the first word, end time is the end of the last word
    start_time = words[0].start
    end_time = words[-1].end

    return Utterance(
        text=text,
        start=start_time,
        end=end_time,
        speaker=speaker,
        words=words
    )


result = asyncio.run(main())
if result is None:
    exit(1)
print(result.model_dump_json())
