import os
import math
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from pydub import AudioSegment
from elevenlabs.client import AsyncElevenLabs
from elevenlabs.types.speech_to_text_chunk_response_model import SpeechToTextChunkResponseModel
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
import json

load_dotenv()


class Word(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    confidence: Optional[float] = None
    speaker: Optional[str] = None
    raw: Optional[dict] = None


class Utterance(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: Optional[float] = None
    words: List[Word] = Field(default_factory=list)
    raw: Optional[dict] = None


class Transcript(BaseModel):
    provider: Literal["gladia", "whisper", "assemblyai", "deepgram", "elevenlabs"]
    language: Optional[str] = None
    source_file: Optional[str] = None
    utterances: List[Utterance] = Field(default_factory=list)
    duration: Optional[float] = None
    raw: Optional[dict] = None


TEMP_DIR = Path("temp/elevenlabs")
TEMP_DIR.mkdir(exist_ok=True, parents=True)


class ElevenLabsProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        file_chunk_number: int,
        rag_chunk_start_index: int,

        # Level 1 — audio file splitting
        segment_duration_min: float = 10,

        # Level 2 — RAG chunking from utterances
        max_time_per_rag_chunk_min: float = 2.0,
        max_words_per_rag_chunk: int = 300,

        # ElevenLabs config
        elevenlabs_api_key: Optional[str] = None,
        base_url: str = "https://api.elevenlabs.io",
        model_id: str = "scribe_v2",
        tag_audio_events: bool = True,
        diarize: bool = True,
        timeout: float = 60 * 10,
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.file_chunk_number = file_chunk_number
        self.rag_chunk_start_index = rag_chunk_start_index

        self.segment_duration_min = segment_duration_min
        self.segment_duration_ms = int(segment_duration_min * 60 * 1000)
        self.max_time_per_rag_chunk_ms = int(max_time_per_rag_chunk_min * 60 * 1000)
        self.max_words_per_rag_chunk = max_words_per_rag_chunk

        self.model_id = model_id
        self.tag_audio_events = tag_audio_events
        self.diarize = diarize

        api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY not set. Pass elevenlabs_api_key= or set the env var."
            )

        self.client = AsyncElevenLabs(
            api_key=api_key,
            timeout=timeout,
            base_url=base_url,
        )

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API — same return signature as AssemblyAI + Deepgram processors
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Level 1: Slice audio at file_chunk_number * segment_duration_min via pydub
        Level 2: Transcribe with ElevenLabs Scribe, normalize to Transcript schema
        Level 3: Group utterances into RAG chunks (whichever guard fires first):
                 - accumulated duration >= max_time_per_rag_chunk_min
                 - accumulated word count >= max_words_per_rag_chunk

        Returns:
            documents:             list of Document (one per RAG chunk)
            next_rag_chunk_index:  pass as rag_chunk_start_index to next message
            is_last:               True if this was the final audio segment
        """
        audio_slice_path, is_last = self._slice_audio()
        transcript = await self._transcribe(audio_slice_path)
        documents = self._build_documents(transcript)
        return documents, self.rag_chunk_start_index + len(documents), is_last

    # -------------------------------------------------------------------------
    # Level 1 — Audio slicing (identical across all processors)
    # -------------------------------------------------------------------------

    def _slice_audio(self) -> Tuple[Path, bool]:
        audio = AudioSegment.from_file(str(self.file_path))
        total_ms = len(audio)

        start_ms = self.file_chunk_number * self.segment_duration_ms
        end_ms = min(start_ms + self.segment_duration_ms, total_ms)

        if start_ms >= total_ms:
            raise ValueError(
                f"file_chunk_number={self.file_chunk_number} is out of range. "
                f"Audio duration: {total_ms / 1000:.1f}s, "
                f"segment_duration: {self.segment_duration_ms / 1000:.1f}s"
            )

        is_last = end_ms >= total_ms
        slice_path = TEMP_DIR / f"{self.file_path.stem}_chunk_{self.file_chunk_number}.mp3"
        audio[start_ms:end_ms].export(str(slice_path), format="mp3")

        return slice_path, is_last

    # -------------------------------------------------------------------------
    # Level 2 — ElevenLabs transcription + normalize to Transcript schema
    # -------------------------------------------------------------------------

    async def _transcribe(self, audio_path: Path) -> Transcript:
        """
        Transcribes audio slice with ElevenLabs Scribe.
        Mirrors your reference code exactly.

        Note:
          - Word-level iteration (no utterance object from API — we build them)
          - Speaker changes detected at word level (same as Deepgram)
          - Confidence from logprob: math.exp(logprob) → 0-1 range
          - Skips "spacing" type words (ElevenLabs-specific token type)
          - Timestamps already in seconds — just add offset
        """
        offset_sec = self.file_chunk_number * self.segment_duration_min * 60

        transcription = await self.client.speech_to_text.convert(
            model_id=self.model_id,
            file=audio_path.read_bytes(),
            tag_audio_events=self.tag_audio_events,
            diarize=self.diarize,
        )

        if not isinstance(transcription, SpeechToTextChunkResponseModel):
            raise RuntimeError(
                f"ElevenLabs returned unexpected response type: {type(transcription)}"
            )

        if not transcription.words:
            return Transcript(
                provider="elevenlabs",
                source_file=str(audio_path),
                language=getattr(transcription, "language_code", None),
                duration=getattr(transcription, "audio_duration_secs", None),
            )

        # Iterate words, group into utterances by speaker change
        final_utterances: List[Utterance] = []
        current_speaker: Optional[str] = None
        current_words: List[Word] = []

        for word_obj in transcription.words:
            # Skip spacing tokens — ElevenLabs-specific, not real words
            if word_obj.type == "spacing":
                continue

            speaker = str(word_obj.speaker_id) if word_obj.speaker_id else "unknown"

            if current_speaker is None:
                current_speaker = speaker

            if speaker != current_speaker:
                if current_words:
                    final_utterances.append(
                        self._make_utterance(current_words, current_speaker)
                    )
                current_speaker = speaker
                current_words = []

            # Convert logprob → confidence (0-1)
            logprob = word_obj.logprob
            confidence = round(math.exp(logprob), 4) if logprob is not None else None

            current_words.append(Word(
                text=word_obj.text.strip(),
                # Timestamps already in seconds — add offset for original-file alignment
                start=None if word_obj.start is None else word_obj.start + offset_sec,
                end=None if word_obj.end is None else word_obj.end + offset_sec,
                confidence=confidence,
                speaker=speaker,
            ))

        # Flush last group
        if current_words:
            final_utterances.append(
                self._make_utterance(current_words, current_speaker or "unknown")
            )

        return Transcript(
            provider="elevenlabs",
            utterances=final_utterances,
            language=getattr(transcription, "language_code", None),
            source_file=str(audio_path),
            duration=getattr(transcription, "audio_duration_secs", None),
        )

    @staticmethod
    def _make_utterance(words: List[Word], speaker: str) -> Utterance:
        """Builds Utterance from words. Start/end already offset-adjusted."""
        return Utterance(
            text=" ".join(w.text for w in words),
            start=words[0].start,
            end=words[-1].end,
            speaker=speaker,
            words=words,
        )

    # -------------------------------------------------------------------------
    # Level 3 — Group utterances into RAG chunks (identical across all processors)
    # -------------------------------------------------------------------------

    def _build_documents(self, transcript: Transcript) -> List[Document]:
        if not transcript.utterances:
            return []

        documents = []
        current_utterances: List[Utterance] = []
        current_words = 0
        current_duration_ms = 0.0

        def flush() -> Optional[Document]:
            if not current_utterances:
                return None
            absolute_index = self.rag_chunk_start_index + len(documents)
            lines = []
            for u in current_utterances:
                speaker_label = f"Speaker {u.speaker}" if u.speaker else "Speaker ?"
                lines.append(f"[{speaker_label} | {self._fmt_time(u.start)} - {self._fmt_time(u.end)}]")
                lines.append(u.text)
                lines.append("")

            return Document(
                id=f"{self.filename}-{absolute_index}",
                page_content="\n".join(lines).strip(),
                metadata={
                    "source": str(self.file_path),
                    "file_chunk_number": self.file_chunk_number,
                    "rag_chunk_number": absolute_index,
                    "provider": transcript.provider,
                    "language": transcript.language,
                    "start_time": round(current_utterances[0].start, 2) if current_utterances[0].start is not None else None,
                    "end_time": round(current_utterances[-1].end, 2) if current_utterances[-1].end is not None else None,
                    "duration_sec": (
                        round(current_utterances[-1].end - current_utterances[0].start, 2)
                        if current_utterances[0].start and current_utterances[-1].end
                        else None
                    ),
                    "utterance_count": len(current_utterances),
                    "speakers": list({u.speaker for u in current_utterances if u.speaker}),
                },
            )

        for utterance in transcript.utterances:
            word_count = len(utterance.text.split())
            duration_ms = (
                (utterance.end - utterance.start) * 1000
                if utterance.start is not None and utterance.end is not None
                else 0.0
            )

            would_exceed_time = (
                current_duration_ms + duration_ms > self.max_time_per_rag_chunk_ms
                and current_utterances
            )
            would_exceed_words = (
                current_words + word_count > self.max_words_per_rag_chunk
                and current_utterances
            )

            if would_exceed_time or would_exceed_words:
                doc = flush()
                if doc:
                    documents.append(doc)
                current_utterances = []
                current_words = 0
                current_duration_ms = 0.0

            current_utterances.append(utterance)
            current_words += word_count
            current_duration_ms += duration_ms

        doc = flush()
        if doc:
            documents.append(doc)

        return documents

    @staticmethod
    def _fmt_time(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--:--"
        seconds = int(seconds)
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


async def main() -> List[Document]:
    file_chunk_number = 0
    rag_chunk_index = 0
    results = []

    while True:
        proc = ElevenLabsProcessor(
            file_path="/home/avvk/Graxon/Graxon/parser/test_data/youtube_podcast_audio.mp3",
            filename="youtube_podcast_audio.mp3",
            file_chunk_number=file_chunk_number,
            rag_chunk_start_index=rag_chunk_index,
            segment_duration_min=2.5,
            base_url="https://api.in.residency.elevenlabs.io"
        )
        docs, next_rag_idx, is_last = await proc.process()
        results.extend(docs)
        # docs → Vector DB + Neo4j

        if is_last:
            break
        file_chunk_number += 1
        rag_chunk_index = next_rag_idx

    return results

if __name__ == "__main__":
    result = asyncio.run(main())
    output_dir = Path("output/elevenlabs")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "youtube_podcast_audio.json"  # Replace with your desired filename
    output_path = output_dir / filename

    # objs = []
    # for doc in result:
    #     objs.append(doc.model_dump_json())

    # result = {"documents": objs}

    def serializer(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, default=serializer, indent=4, ensure_ascii=False)

    print(f"Saved output to {output_path}")
