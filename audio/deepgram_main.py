import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from pydub import AudioSegment
from deepgram import AsyncDeepgramClient
from deepgram.types.listen_v1response import ListenV1Response
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


TEMP_DIR = Path("temp")


class DeepgramProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        file_chunk_number: int,             # which audio segment (0, 1, 2, ...)
        rag_chunk_start_index: int,         # absolute RAG chunk index to continue from

        # Level 1 — audio file splitting
        segment_duration_min: float = 10,   # minutes per audio IO buffer

        # Level 2 — RAG chunking from utterances
        max_time_per_rag_chunk_min: float = 2.0,
        max_words_per_rag_chunk: int = 300,

        # Deepgram config
        deepgram_api_key: Optional[str] = None,
        model: str = "nova-3",
        diarize: bool = True,
        smart_format: bool = True,
        detect_language: bool = True,
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.file_chunk_number = file_chunk_number
        self.rag_chunk_start_index = rag_chunk_start_index

        self.segment_duration_min = segment_duration_min
        self.segment_duration_ms = int(segment_duration_min * 60 * 1000)
        self.max_time_per_rag_chunk_ms = int(max_time_per_rag_chunk_min * 60 * 1000)
        self.max_words_per_rag_chunk = max_words_per_rag_chunk

        self.model = model
        self.diarize = diarize
        self.smart_format = smart_format
        self.detect_language = detect_language

        api_key = deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY not set. Pass deepgram_api_key= or set the env var."
            )
        self.client = AsyncDeepgramClient(api_key=api_key)

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API — same return signature as AssemblyAIProcessor
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Level 1: Slice audio at file_chunk_number * segment_duration_min via pydub
        Level 2: Transcribe with Deepgram Nova-3, normalize to Transcript schema
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
    # Level 1 — Audio slicing (identical to AssemblyAIProcessor)
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
    # Level 2 — Deepgram transcription + normalize to Transcript schema
    # -------------------------------------------------------------------------

    async def _transcribe(self, audio_path: Path) -> Transcript:
        """
        Transcribes audio slice with Deepgram Nova-3.
        Normalizes utterances from the response — mirrors your reference code exactly.

        Key difference from AssemblyAI:
          Deepgram's utterances may have speaker changes MID-utterance (word level),
          so we split on speaker changes within each utterance (your create_utterance logic).
          Timestamps are already in seconds — no ms conversion needed.
          Offset is added to make timestamps relative to the original full file.
        """
        offset_sec = self.file_chunk_number * self.segment_duration_min * 60

        response = await self.client.listen.v1.media.transcribe_file(
            request=audio_path.read_bytes(),
            model=self.model,
            smart_format=self.smart_format,
            diarize=self.diarize,
            utterances=True,
            detect_language=self.detect_language,
        )

        if not isinstance(response, ListenV1Response):
            raise RuntimeError(
                f"Deepgram returned unexpected response type: {type(response)}"
            )

        duration = response.metadata.duration if response.metadata else None
        language = (
            response.results.channels[0].detected_language
            if response.results and response.results.channels
            else None
        )

        raw_utterances = (
            response.results.utterances
            if response.results and response.results.utterances
            else []
        )

        if not raw_utterances:
            return Transcript(
                provider="deepgram",
                source_file=str(audio_path),
                duration=duration,
                language=language,
            )

        # Normalize — split on mid-utterance speaker changes (your reference approach)
        final_utterances: List[Utterance] = []

        for utt in raw_utterances:
            if not utt.words:
                continue

            current_speaker: Optional[str] = None
            current_words: List[Word] = []

            for raw_word in utt.words:
                word_speaker = str(raw_word.speaker)

                if current_speaker is None:
                    current_speaker = word_speaker

                if word_speaker != current_speaker:
                    # Speaker changed — flush current group
                    if current_words:
                        final_utterances.append(
                            self._make_utterance(current_words, current_speaker, offset_sec)
                        )
                    current_speaker = word_speaker
                    current_words = []

                text = raw_word.punctuated_word or raw_word.word or ""
                current_words.append(Word(
                    text=text,
                    # Deepgram gives seconds directly — just add offset
                    start=None if raw_word.start is None else raw_word.start + offset_sec,
                    end=None if raw_word.end is None else raw_word.end + offset_sec,
                    confidence=raw_word.confidence,
                    speaker=word_speaker,
                ))

            # Flush last group
            if current_words:
                final_utterances.append(
                    self._make_utterance(current_words, current_speaker or "unknown", offset_sec)
                )

        return Transcript(
            provider="deepgram",
            utterances=final_utterances,
            source_file=str(audio_path),
            duration=duration,
            language=language,
        )

    @staticmethod
    def _make_utterance(words: List[Word], speaker: str, offset_sec: float) -> Utterance:
        """
        Builds an Utterance from a list of Words.
        Text is reconstructed by joining word texts.
        Start/end taken from first/last word (already offset-adjusted).
        """
        text = " ".join(w.text for w in words)
        return Utterance(
            text=text,
            start=words[0].start,
            end=words[-1].end,
            speaker=speaker,
            words=words,
        )

    # -------------------------------------------------------------------------
    # Level 3 — Group utterances into RAG chunks (identical to AssemblyAIProcessor)
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
        proc = DeepgramProcessor(
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
    output_dir = Path("output/deepgram")
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
