import os
import json
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from pydub import AudioSegment
from langchain_core.documents import Document
import subprocess
from twelvelabs import AsyncTwelveLabs
from twelvelabs.types import (
    AsyncResponseFormat,
    SegmentDefinition,
    SegmentField,
    VideoContext_AssetId,
    AnalyzePromptV2,
)
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = Path("temp")

SEGMENT_FIELDS = [
    SegmentField(
        name="topic_summary",
        type="string",
        description=(
            "1-2 sentence summary of the core topic or theme of this segment. "
            "Be dense and specific — avoid vague summaries like 'people are talking'."
        ),
    ),
    SegmentField(
        name="keywords",
        type="string",
        description=(
            "Comma-separated named entities, topics, products, and people mentioned "
            "or visible in this segment. e.g. 'Donald Trump, vulnerability, trust, podcast'"
        ),
    ),
    SegmentField(
        name="transcript",
        type="string",
        description="Verbatim spoken words and dialogue during this segment, or empty string if no speech.",
    ),
    SegmentField(
        name="detailed_description",
        type="string",
        description=(
            "A thorough description of everything visible: people, actions, setting, "
            "objects, camera framing, movement, and any notable visual details."
        ),
    ),
    SegmentField(
        name="setting",
        type="string",
        description="Where this segment takes place: location, environment, indoor/outdoor, time of day if determinable.",
    ),
    SegmentField(
        name="people_present",
        type="string",
        description="Description of any people visible: appearance, clothing, identity if known, expressions, actions.",
    ),
    SegmentField(
        name="speaker_names",
        type="string",
        description="Names of speakers if identifiable from visual or audio cues, or empty string if unknown.",
    ),
    SegmentField(
        name="on_screen_text",
        type="string",
        description="Any text, captions, lower thirds, titles, or graphics visible on screen, or empty string if none.",
    ),
    SegmentField(
        name="audio_description",
        type="string",
        description="Non-speech audio: music genre/mood, ambient sounds, sound effects.",
    ),
    SegmentField(
        name="mood",
        type="string",
        description="Emotional tone of this segment, e.g. 'intense debate', 'light-hearted', 'emotional', 'informative'.",
    ),
    SegmentField(
        name="has_speech",
        type="string",
        description="'true' if there is spoken dialogue in this segment, 'false' if silent or music only.",
    ),
]

OVERVIEW_PROMPT = (
    "Describe this entire video comprehensively covering: "
    "1. Who appears in the video (names, appearance, roles). "
    "2. The overall topic and purpose of the video. "
    "3. Key themes and topics discussed in chronological order. "
    "4. The setting and production style. "
    "5. Any notable moments, quotes, or visual elements. "
    "6. The tone and target audience."
)


class TwelveLabsProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        file_chunk_number: int,                     # which 10-min window (0, 1, 2 ...)
        rag_chunk_start_index: int,                 # absolute RAG chunk index to continue from

        # Level 1 — video slicing
        chunk_duration_min: float = 10.0,           # core window duration
        overlap_min: float = 1.0,                   # overlap on each side

        # Level 2 — RAG chunking from segments
        max_duration_per_rag_chunk_sec: float = 180.0,
        max_words_per_rag_chunk: int = 400,

        # TwelveLabs config
        twelvelabs_api_key: Optional[str] = None,
        model_name: str = "pegasus1.5",
        poll_interval: float = 5.0,
        max_workers: int = 5,
        max_retries: int = 3,
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.file_chunk_number = file_chunk_number
        self.rag_chunk_start_index = rag_chunk_start_index

        self.chunk_duration_sec = chunk_duration_min * 60
        self.overlap_sec = overlap_min * 60
        self.chunk_duration_ms = int(chunk_duration_min * 60 * 1000)
        self.overlap_ms = int(overlap_min * 60 * 1000)
        self.max_duration_per_rag_chunk_sec = max_duration_per_rag_chunk_sec
        self.max_words_per_rag_chunk = max_words_per_rag_chunk

        self.model_name = model_name
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self.max_retries = max_retries

        api_key = twelvelabs_api_key or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TWELVELABS_API_KEY not set. Pass twelvelabs_api_key= or set the env var."
            )
        self.client = AsyncTwelveLabs(api_key=api_key)

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Level 1: Slice video file into 12-min clip (10 core + 1 overlap each side)
                 using pydub. Each chunk is a separate small upload — no full-file upload.
                 Saved to temp/{stem}_chunk_{n}.mp4

        Level 2: Upload slice to TwelveLabs, wait for asset ready,
                 run segmentation + overview (chunk 0 only) in parallel.

        Level 3: Filter segments to core window only (drop overlap segments).
                 Group segments into RAG chunks by dual guard:
                   - accumulated duration >= max_duration_per_rag_chunk_sec
                   - accumulated word count >= max_words_per_rag_chunk
                 Each group → one Document.

        Returns:
            documents:             list of Document (one per RAG chunk + overview if chunk 0)
            next_rag_chunk_index:  pass as rag_chunk_start_index to next message
            is_last:               True if this was the final video slice
        """
        # --- Level 1: slice video ---
        slice_path, core_start_sec, core_end_sec, offset_sec, is_last = self._slice_video()

        # --- Level 2: upload slice + analyze ---
        asset_id = await self._upload_slice(slice_path)
        seg_task, overview_task = await self._create_analysis_tasks(
            asset_id=asset_id,
            include_overview=(self.file_chunk_number == 0),
        )
        seg_task, overview_task = await asyncio.gather(
            self._poll_task(seg_task.task_id),
            self._poll_task(overview_task.task_id) if overview_task else asyncio.sleep(0),
        )

        # --- Parse segments ---
        raw_segments = []
        if seg_task.status == "ready":
            raw_segments = json.loads(seg_task.result.data).get("segments", [])

        # --- Filter to core window + adjust timestamps to original file ---
        core_segments = self._filter_and_adjust(
            segments=raw_segments,
            overlap_sec=self.overlap_sec if self.file_chunk_number > 0 else 0.0,
            core_duration_sec=self.chunk_duration_sec,
            offset_sec=offset_sec,
        )

        # --- Level 3: build RAG chunk documents ---
        documents = self._build_documents(core_segments)

        # --- Overview document (chunk 0 only) ---
        if (
            self.file_chunk_number == 0
            and overview_task
            and hasattr(overview_task, "status")
            and overview_task.status == "ready"
        ):
            overview_doc = self._build_overview_document(overview_task.result.data)
            documents = [overview_doc] + documents

        return documents, self.rag_chunk_start_index + len(documents), is_last

    # -------------------------------------------------------------------------
    # Level 1 — Video slicing with pydub
    # -------------------------------------------------------------------------

    def _get_video_duration(self, file_path: Path) -> float:
        """Uses ffprobe to get the exact duration of the video in seconds."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _slice_video(self) -> Tuple[Path, float, float, float, bool]:
        """
        Loads full video, extracts the slice for this file_chunk_number using ffmpeg.

        Slice = [core_start - overlap : core_end + overlap]
          → core window: 10 min of actual content to index
          → overlap: 1 min on each side for TwelveLabs context at boundaries

        offset_sec = start of core window in the ORIGINAL file
                   → added to segment timestamps so they reference the original file

        Returns: (slice_path, core_start_sec, core_end_sec, offset_sec, is_last)
        """
        total_sec = self._get_video_duration(self.file_path)

        # Core window in original file
        core_start_sec = self.file_chunk_number * self.chunk_duration_sec
        core_end_sec = min(core_start_sec + self.chunk_duration_sec, total_sec)

        if core_start_sec >= total_sec:
            raise ValueError(
                f"file_chunk_number={self.file_chunk_number} is out of range. "
                f"Video duration: {total_sec:.1f}s"
            )

        # Slice with overlap
        slice_start_sec = max(0, core_start_sec - self.overlap_sec)
        slice_end_sec = min(total_sec, core_end_sec + self.overlap_sec)
        slice_duration_sec = slice_end_sec - slice_start_sec

        is_last = core_end_sec >= total_sec

        slice_path = TEMP_DIR / f"{self.file_path.stem}_chunk_{self.file_chunk_number}.mp4"

        # Fast-seek by placing -ss before -i. 
        # -c copy prevents re-encoding and preserves both video/audio streams losslessly.
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(slice_start_sec),
            "-i", str(self.file_path),
            "-t", str(slice_duration_sec),
            "-c", "copy",
            str(slice_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        # offset_sec = where core_start sits in the original file
        # Used to convert slice-relative timestamps → original-file timestamps
        offset_sec = core_start_sec

        return (
            slice_path,
            core_start_sec,
            core_end_sec,
            offset_sec,
            is_last,
        )

    # -------------------------------------------------------------------------
    # Upload slice + wait for asset ready
    # -------------------------------------------------------------------------

    async def _upload_slice(self, slice_path: Path) -> str:
        """
        Uploads the small video slice (12 min) to TwelveLabs.
        Each chunk is a fresh upload — no full-file upload needed.
        Returns asset_id.
        """
        async def progress_callback(progress):
            print(
                f"\r  Uploading chunk {self.file_chunk_number}: {progress.percentage:.1f}% "
                f"({progress.completed_chunks}/{progress.total_chunks} chunks)",
                end="", flush=True,
            )

        result = await self.client.multipart_upload.upload_file(
            file_path=slice_path,
            file_type="video",
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            progress_callback=progress_callback,
        )
        asset_id = result.asset_id
        print(f"\nChunk {self.file_chunk_number} uploaded. Asset ID: {asset_id}")

        # Brief delay — give TwelveLabs a moment to register the asset
        # before polling, otherwise retrieve can 404 immediately after upload
        await asyncio.sleep(3)

        # Poll until asset is ready — use asset_id from UploadResult directly
        # (Asset.id is aliased from _id and may deserialize as None)
        while True:
            asset = await self.client.assets.retrieve(asset_id)
            if asset.status == "ready":
                print(f"Asset {asset_id} ready.")
                break
            if asset.status == "failed":
                error = getattr(asset, "error", None)
                raise RuntimeError(
                    f"Asset processing failed for chunk {self.file_chunk_number}: "
                    f"asset_id={asset_id}, error={error}"
                )
            print(f"  Asset status: {asset.status} — waiting...")
            await asyncio.sleep(self.poll_interval)

        return asset_id

    # -------------------------------------------------------------------------
    # Analysis tasks
    # -------------------------------------------------------------------------

    async def _create_analysis_tasks(self, asset_id: str, include_overview: bool):
        """
        Creates segmentation task (always) and overview task (chunk 0 only).
        No start_time/end_time needed — the slice IS the window.
        """
        seg_task = await self.client.analyze_async.tasks.create(
            model_name=self.model_name,
            video=VideoContext_AssetId(asset_id=asset_id),
            analysis_mode="time_based_metadata",
            max_tokens=98304,
            response_format=AsyncResponseFormat(
                type="segment_definitions",
                segment_definitions=[
                    SegmentDefinition(
                        id="segments",
                        description=(
                            "Segment the video into distinct scenes, topics, or actions. "
                            "Each segment should cover one coherent topic or scene."
                        ),
                        fields=SEGMENT_FIELDS,
                    )
                ],
            ),
        )

        overview_task = None
        if include_overview:
            overview_task = await self.client.analyze_async.tasks.create(
                model_name=self.model_name,
                video=VideoContext_AssetId(asset_id=asset_id),
                analysis_mode="general",
                max_tokens=4096,
                prompt_v_2=AnalyzePromptV2(input_text=OVERVIEW_PROMPT),
            )

        return seg_task, overview_task

    async def _poll_task(self, task_id: str):
        while True:
            task = await self.client.analyze_async.tasks.retrieve(task_id)
            if task.status in ("ready", "failed"):
                return task
            await asyncio.sleep(self.poll_interval)

    # -------------------------------------------------------------------------
    # Filter overlap segments + adjust timestamps to original file
    # -------------------------------------------------------------------------

    def _filter_and_adjust(
        self,
        segments: List[Dict],
        overlap_sec: float,
        core_duration_sec: float,
        offset_sec: float,
    ) -> List[Dict]:
        """
        Two things happen here:

        1. Filter — drop segments that start in the overlap zones:
             segment.start_time < overlap_sec           → left overlap, drop
             segment.start_time >= overlap_sec + core   → right overlap, drop
           (chunk 0 has no left overlap so overlap_sec=0)

        2. Adjust timestamps — TwelveLabs timestamps are relative to the slice.
           Add offset_sec to make them relative to the original full video:
             adjusted_start = segment.start_time - overlap_sec + offset_sec
             adjusted_end   = segment.end_time   - overlap_sec + offset_sec

        Example:
          Original file: 60 min
          Chunk 1 slice: 09:00 - 21:00 (uploaded as 0:00 - 12:00 to TwelveLabs)
          overlap_sec = 60, core_duration_sec = 600, offset_sec = 600

          Segment at slice time 01:30 (90s):
            → 90 < 60? No → keep
            → 90 >= 60 + 600? No → keep
            → adjusted = 90 - 60 + 600 = 630s = 10:30 in original ✅
        """
        core_end_in_slice = overlap_sec + core_duration_sec
        result = []

        for seg in segments:
            seg_start = seg.get("start_time", 0)
            seg_end = seg.get("end_time", 0)

            # Drop if starts in overlap zones
            if seg_start < overlap_sec:
                continue
            if seg_start >= core_end_in_slice:
                continue

            # Adjust to original file timestamps
            adjusted = dict(seg)
            adjusted["start_time"] = seg_start - overlap_sec + offset_sec
            adjusted["end_time"] = seg_end - overlap_sec + offset_sec

            result.append(adjusted)

        return result

    # -------------------------------------------------------------------------
    # Level 3 — Build RAG chunk Documents
    # -------------------------------------------------------------------------

    def _build_documents(self, segments: List[Dict]) -> List[Document]:
        if not segments:
            return []

        documents = []
        current_segments: List[Dict] = []
        current_duration_sec = 0.0
        current_words = 0

        def flush() -> Optional[Document]:
            if not current_segments:
                return None

            absolute_index = self.rag_chunk_start_index + len(documents)
            chunk_start = current_segments[0].get("start_time")
            chunk_end = current_segments[-1].get("end_time")

            page_content = self._render_page_content(current_segments, chunk_start, chunk_end)

            all_keywords = ", ".join(filter(None, [
                seg.get("metadata", {}).get("keywords", "") for seg in current_segments
            ]))
            has_speech = any(
                seg.get("metadata", {}).get("has_speech", "false").lower() == "true"
                for seg in current_segments
            )

            return Document(
                id=f"{self.filename}-{absolute_index}",
                page_content=page_content,
                metadata={
                    "source": str(self.file_path),
                    "file_chunk_number": self.file_chunk_number,
                    "rag_chunk_number": absolute_index,
                    "provider": "twelvelabs",
                    "model": self.model_name,
                    "start_time": chunk_start,
                    "end_time": chunk_end,
                    "duration_sec": (
                        chunk_end - chunk_start
                        if chunk_start is not None and chunk_end is not None
                        else None
                    ),
                    "segment_count": len(current_segments),
                    "keywords": all_keywords,
                    "has_speech": has_speech,
                },
            )

        for seg in segments:
            meta = seg.get("metadata", {})
            transcript = meta.get("transcript", "") or ""
            word_count = len(transcript.split())
            duration_sec = seg.get("end_time", 0) - seg.get("start_time", 0)

            would_exceed_duration = (
                current_duration_sec + duration_sec > self.max_duration_per_rag_chunk_sec
                and current_segments
            )
            would_exceed_words = (
                current_words + word_count > self.max_words_per_rag_chunk
                and current_segments
            )

            if would_exceed_duration or would_exceed_words:
                doc = flush()
                if doc:
                    documents.append(doc)
                current_segments = []
                current_duration_sec = 0.0
                current_words = 0

            current_segments.append(seg)
            current_duration_sec += duration_sec
            current_words += word_count

        doc = flush()
        if doc:
            documents.append(doc)

        return documents

    def _render_page_content(
        self,
        segments: List[Dict],
        chunk_start: Optional[float],
        chunk_end: Optional[float],
    ) -> str:
        """Renders all segments into page_content. Dense semantic content first."""
        lines = [f"[{self._fmt_time(chunk_start)} - {self._fmt_time(chunk_end)}]"]

        def collect(field: str) -> List[str]:
            return [
                seg.get("metadata", {}).get(field, "")
                for seg in segments
                if seg.get("metadata", {}).get(field)
            ]

        def collect_unique(field: str) -> List[str]:
            return list({
                seg.get("metadata", {}).get(field, "")
                for seg in segments
                if seg.get("metadata", {}).get(field)
            })

        if topics := collect("topic_summary"):
            lines.append(f"TOPIC: {' '.join(topics)}")
        if keywords := ", ".join(filter(None, collect("keywords"))):
            lines.append(f"KEYWORDS: {keywords}")
        if transcripts := collect("transcript"):
            lines.append(f"TRANSCRIPT: {' '.join(transcripts)}")
        if descriptions := collect("detailed_description"):
            lines.append(f"VISUAL: {' '.join(descriptions)}")
        if settings := collect_unique("setting"):
            lines.append(f"SETTING: {', '.join(settings)}")
        if people := collect("people_present"):
            lines.append(f"PEOPLE: {' '.join(people)}")
        if speakers := collect_unique("speaker_names"):
            lines.append(f"SPEAKERS: {', '.join(speakers)}")
        if texts := collect("on_screen_text"):
            lines.append(f"ON-SCREEN TEXT: {' '.join(texts)}")
        if audio := collect("audio_description"):
            lines.append(f"AUDIO: {' '.join(audio)}")
        if moods := collect_unique("mood"):
            lines.append(f"MOOD: {', '.join(moods)}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Overview Document (chunk 0 only)
    # -------------------------------------------------------------------------

    def _build_overview_document(self, overview_text: str) -> Document:
        return Document(
            id=f"{self.filename}-overview",
            page_content=f"VIDEO OVERVIEW\n\n{overview_text}",
            metadata={
                "source": str(self.file_path),
                "file_chunk_number": 0,
                "rag_chunk_number": -1,
                "provider": "twelvelabs",
                "model": self.model_name,
                "document_type": "overview",
                "start_time": None,
                "end_time": None,
            },
        )

    @staticmethod
    def _fmt_time(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--:--"
        seconds = int(seconds)
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


async def process_video(file_path: str):
    file_chunk_number = 0
    rag_chunk_index = 0
    results = []

    while True:
        print(f"\n{'=' * 60}")
        print(f"Processing chunk {file_chunk_number} | rag_chunk_start={rag_chunk_index}")
        print(f"{'=' * 60}")

        processor = TwelveLabsProcessor(
            file_path=file_path,
            filename=Path(file_path).name,
            file_chunk_number=file_chunk_number,
            rag_chunk_start_index=rag_chunk_index,
            chunk_duration_min=2.5,
        )

        try:
            documents, next_rag_idx, is_last = await processor.process()
        except ValueError as e:
            print(f"Done: {e}")
            break

        # Store this file chunk as its own list
        results.append([doc.model_dump() if hasattr(doc, "model_dump") else doc.__dict__ for doc in documents])

        if is_last or not documents:
            print("\nVideo processing completed.")
            break

        file_chunk_number += 1
        rag_chunk_index = next_rag_idx

    # Save JSON
    output_dir = Path("temp/twelvelabs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(file_path).stem}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved output to {output_file}")


if __name__ == "__main__":
    # VIDEO_PATH = "/home/avvk/Graxon/Graxon/parser/test_data/IMG_3509.MOV"
    VIDEO_PATH = "/home/avvk/Graxon/Graxon/parser/test_data/youtube_postcast_video.mp4"
    asyncio.run(process_video(VIDEO_PATH))
