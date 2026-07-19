from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from pydub import AudioSegment
from typing import Literal
from pathlib import Path
import assemblyai as aai
import asyncio
import json
import os

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


TEMP_DIR = Path("temp/assemblyai")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class AssemblyAIProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        file_chunk_number: int,            # which audio segment (0, 1, 2, ...)
        rag_chunk_start_index: int,        # absolute RAG chunk index to continue from

        # Level 1 — audio file splitting
        segment_duration_min: float = 10,  # minutes per audio IO buffer

        # Level 2 — RAG chunking from utterances
        max_time_per_rag_chunk_min: float = 2.0,   # max minutes per RAG chunk
        max_words_per_rag_chunk: int = 300,         # max words per RAG chunk
        # whichever hits first stops the chunk ↑

        # AssemblyAI config
        assemblyai_api_key: Optional[str] = None,
        speaker_labels: bool = True,
        language_detection: bool = True,
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.file_chunk_number = file_chunk_number
        self.rag_chunk_start_index = rag_chunk_start_index

        self.segment_duration_ms = int(segment_duration_min * 60 * 1000)  # pydub uses ms
        self.segment_duration_min = segment_duration_min
        self.max_time_per_rag_chunk_ms = int(max_time_per_rag_chunk_min * 60 * 1000)
        self.max_words_per_rag_chunk = max_words_per_rag_chunk

        self.speaker_labels = speaker_labels
        self.language_detection = language_detection

        api_key = assemblyai_api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY not set. Pass assemblyai_api_key= or set the env var."
            )
        aai.settings.api_key = api_key

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API — same return signature as all other processors
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Level 1: Slice audio file at file_chunk_number * segment_duration_min
                 using pydub — save to temp/{stem}_chunk_{n}.mp3
        Level 2: Transcribe the slice with AssemblyAI
                 Normalize to Transcript schema
        Level 3: Group utterances into RAG chunks by whichever guard fires first:
                 - accumulated duration >= max_time_per_rag_chunk_min
                 - accumulated word count >= max_words_per_rag_chunk
                 Each group → one Document

        Returns:
            documents:             list of Document (one per RAG chunk)
            next_rag_chunk_index:  pass as rag_chunk_start_index to next message
            is_last:               True if this was the final audio segment
        """
        # --- Level 1: slice audio ---
        audio_slice_path, is_last = self._slice_audio()

        # --- Level 2: transcribe + normalize ---
        transcript = await self._transcribe(audio_slice_path)

        # --- Level 3: group utterances into RAG chunks ---
        documents = self._build_documents(transcript)

        return documents, self.rag_chunk_start_index + len(documents), is_last

    # -------------------------------------------------------------------------
    # Level 1 — Audio slicing with pydub
    # -------------------------------------------------------------------------

    def _slice_audio(self) -> Tuple[Path, bool]:
        """
        Loads the full audio file, extracts the segment at file_chunk_number offset.
        Saves to temp/{stem}_chunk_{n}.mp3

        Uses pydub ms-based slicing:
            audio[start_ms : end_ms]

        Returns: (slice_path, is_last)
        """
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
        audio_slice = audio[start_ms:end_ms]

        slice_path = TEMP_DIR / f"{self.file_path.stem}_chunk_{self.file_chunk_number}.mp3"
        audio_slice.export(str(slice_path), format="mp3")

        return slice_path, is_last

    # -------------------------------------------------------------------------
    # Level 2 — AssemblyAI transcription + normalize to Transcript schema
    # -------------------------------------------------------------------------

    async def _transcribe(self, audio_path: Path) -> Transcript:
        """
        Transcribes the audio slice with AssemblyAI.
        Normalizes the response to the universal Transcript schema.
        Mirrors your reference code exactly.
        """
        config = aai.TranscriptionConfig(
            speaker_labels=self.speaker_labels,
            language_detection=self.language_detection,
        )
        transcriber = aai.Transcriber()

        # AssemblyAI SDK uses concurrent.futures — wrap for async
        future = transcriber.transcribe_async(str(audio_path), config=config)
        result = await asyncio.wrap_future(future)

        if result.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {result.error}")

        if not result.utterances:
            return Transcript(
                provider="assemblyai",
                source_file=str(audio_path),
                duration=result.audio_duration,
                language=result.language_code,
            )

        offset_sec = self.file_chunk_number * self.segment_duration_min * 60

        # Normalize to schema
        utterances: List[Utterance] = []

        for u in result.utterances:
            words: List[Word] = [
                Word(
                    text=w.text,
                    start=None if w.start is None else (w.start / 1000) + offset_sec,  # ms → seconds
                    end=None if w.end is None else (w.end / 1000) + offset_sec,
                    confidence=w.confidence,
                    speaker=w.speaker,
                )
                for w in (u.words or [])
            ]

            utterances.append(Utterance(
                text=u.text,
                start=offset_sec if u.start is None else (u.start / 1000) + offset_sec,  # ms → seconds
                end=None if u.end is None else (u.end / 1000) + offset_sec,
                speaker=str(u.speaker) if u.speaker else None,
                confidence=u.confidence,
                words=words,
            ))

        return Transcript(
            provider="assemblyai",
            utterances=utterances,
            language=result.language_code,
            source_file=str(audio_path),
            duration=result.audio_duration,
        )

    # -------------------------------------------------------------------------
    # Level 3 — Group utterances into RAG chunks
    # -------------------------------------------------------------------------

    def _build_documents(self, transcript: Transcript) -> List[Document]:
        """
        Groups consecutive utterances into RAG chunks.
        Stops a chunk when EITHER guard fires first:
          - accumulated duration >= max_time_per_rag_chunk_ms
          - accumulated word count >= max_words_per_rag_chunk

        Each group → one Document.
        page_content format:
            [Speaker A | 00:02:15 - 00:04:30]
            utterance text here...

            [Speaker B | 00:04:30 - 00:05:10]
            next utterance...
        """
        if not transcript.utterances:
            return []

        documents = []
        current_utterances: List[Utterance] = []
        current_words = 0
        current_duration_ms = 0.0

        def flush(utterances: List[Utterance]) -> Optional[Document]:
            if not utterances:
                return None

            absolute_index = self.rag_chunk_start_index + len(documents)

            # Build page_content — one block per utterance
            lines = []
            for u in utterances:
                speaker_label = f"Speaker {u.speaker}" if u.speaker else "Speaker ?"
                start_fmt = self._fmt_time(u.start)
                end_fmt = self._fmt_time(u.end)
                lines.append(f"[{speaker_label} | {start_fmt} - {end_fmt}]")
                lines.append(u.text)
                lines.append("")  # blank line between utterances

            # Chunk time range
            chunk_start = utterances[0].start
            chunk_end = utterances[-1].end

            return Document(
                id=f"{self.filename}-{absolute_index}",
                page_content="\n".join(lines).strip(),
                metadata={
                    "source": str(self.file_path),
                    "file_chunk_number": self.file_chunk_number,   # which 10min audio slice
                    "rag_chunk_number": absolute_index,            # absolute RAG index
                    "provider": transcript.provider,
                    "language": transcript.language,
                    "start_time": round(chunk_start, 2) if chunk_start is not None else None,
                    "end_time": round(chunk_end, 2) if chunk_end is not None else None,
                    "duration_sec": (
                        round(chunk_end - chunk_start, 2)
                        if chunk_start is not None and chunk_end is not None
                        else None
                    ),
                    "utterance_count": len(utterances),
                    "speakers": list({u.speaker for u in utterances if u.speaker}),
                },
            )

        for utterance in transcript.utterances:
            word_count = len(utterance.text.split())
            duration_ms = 0.0
            if utterance.start is not None and utterance.end is not None:
                duration_ms = (utterance.end - utterance.start) * 1000  # sec → ms

            # Check if adding this utterance would breach either guard
            would_exceed_time = (
                current_duration_ms + duration_ms > self.max_time_per_rag_chunk_ms
                and current_utterances  # always include at least one utterance
            )
            would_exceed_words = (
                current_words + word_count > self.max_words_per_rag_chunk
                and current_utterances
            )

            if would_exceed_time or would_exceed_words:
                # Flush current group → Document
                doc = flush(current_utterances)
                if doc:
                    documents.append(doc)
                current_utterances = []
                current_words = 0
                current_duration_ms = 0.0

            current_utterances.append(utterance)
            current_words += word_count
            current_duration_ms += duration_ms

        # Flush final group
        doc = flush(current_utterances)
        if doc:
            documents.append(doc)

        return documents

    @staticmethod
    def _fmt_time(seconds: Optional[float]) -> str:
        """Formats seconds as MM:SS or HH:MM:SS."""
        if seconds is None:
            return "--:--"
        seconds = int(seconds)
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        if h:
            return f"{h:02}:{m:02}:{s:02}"
        return f"{m:02}:{s:02}"


async def main() -> List[Document]:
    file_chunk_number = 0
    rag_chunk_index = 0
    results = []

    while True:
        proc = AssemblyAIProcessor(
            file_path="/home/avvk/Graxon/Graxon/parser/test_data/youtube_podcast_audio.mp3",
            filename="youtube_podcast_audio.mp3",
            file_chunk_number=file_chunk_number,
            rag_chunk_start_index=rag_chunk_index,
            segment_duration_min=2.5,
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
    output_dir = Path("output/assembly")
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
