from llama_cloud.types.parsing_get_response import MarkdownPageMarkdownResultPage
from typing import Tuple, Optional, Literal
from llama_cloud import AsyncLlamaCloud
from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import mimetypes
import os

load_dotenv()


# Image MIME types — process whole file in one shot (no page splitting)
IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/tiff", "image/bmp", "image/gif",
}

TEMP_DIR = Path("temp/llamaparse")

# Valid LlamaCloud tier values
TierType = Literal["fast", "cost_effective", "agentic", "agentic_plus"]


class LlamaCloudOCRProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_page: int = 0,                # 0-based page index (PDF only, ignored for images)
        max_pages_per_chunk: int = 100,     # pages per LlamaCloud job
        tier: TierType = "agentic",         # LlamaCloud parsing tier
        version: str = "latest",            # LlamaCloud parsing version
        poll_interval: float = 2.0,         # seconds between status polls
        timeout: float = 60 * 10,           # seconds before giving up on a job
        llama_api_key: Optional[str] = None,
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.start_page = start_page
        self.max_pages_per_chunk = max_pages_per_chunk
        self.tier = tier
        self.version = version
        self.poll_interval = poll_interval
        self.timeout = timeout

        api_key = llama_api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not set. Pass llama_api_key= or set the env var."
            )

        self.client = AsyncLlamaCloud(api_key=api_key)

        self._mime_type = (
            mimetypes.guess_type(str(self.file_path))[0] or "application/octet-stream"
        )
        self._is_image = self._mime_type in IMAGE_MIME_TYPES

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API — same return signature as OCRProcessor + DatalabOCRProcessor
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[Path, int, bool]:
        """
        Single image → parse whole file at once (no page splitting needed).
        PDF         → parse pages [start_page : start_page + max_pages_per_chunk]
                      using LlamaCloud's native target_pages param — no PDF splitting.

        Key difference from Mistral/Datalab: LlamaCloud accepts page ranges server-side
        via target_pages, so we never need to physically split the PDF.

        Returns:
            md_path:    Path to saved .md file in temp/
                        → pass directly to MarkdownProcessor
            next_page:  pass as start_page to the next queue message (0 for images)
            is_last:    True = file fully consumed, no more batches needed
        """
        if self._is_image:
            return await self._process_image()
        else:
            return await self._process_pdf()

    # -------------------------------------------------------------------------
    # Image — single-shot parse (no splitting)
    # -------------------------------------------------------------------------

    async def _process_image(self) -> Tuple[Path, int, bool]:
        """
        Images have no pages — parse the whole file in one LlamaCloud job.
        Always returns is_last=True.
        """
        markdown = await self._upload_and_parse(
            file_path=self.file_path,
            page_range=None,            # no page range for images
        )

        md_path = self._save_markdown(
            markdown=markdown,
            stem=self.file_path.stem,
            suffix="",                  # no chunk suffix for single image
        )

        return md_path, 0, True

    # -------------------------------------------------------------------------
    # PDF — page-range parse (no physical splitting needed)
    # -------------------------------------------------------------------------

    async def _process_pdf(self) -> Tuple[Path, int, bool]:
        """
        Reads total page count, calculates the current batch range, then passes
        it to LlamaCloud as target_pages (e.g. "1-100") — the API handles the rest.

        No temp PDF files needed — unlike Mistral/Datalab, LlamaCloud accepts
        the original full PDF + a page range param.
        """
        reader = PdfReader(str(self.file_path))
        total_pages = len(reader.pages)

        if self.start_page >= total_pages:
            raise ValueError(
                f"start_page={self.start_page} is out of range. "
                f"PDF has {total_pages} pages."
            )

        # Calculate end page (exclusive) for this batch
        end_page = min(self.start_page + self.max_pages_per_chunk, total_pages)
        is_last = end_page >= total_pages

        # LlamaCloud uses 1-based page indexing in target_pages
        page_range = f"{self.start_page + 1}-{end_page}"

        markdown = await self._upload_and_parse(
            file_path=self.file_path,
            page_range=page_range,
        )

        md_path = self._save_markdown(
            markdown=markdown,
            stem=self.file_path.stem,
            suffix=f"_pages_{self.start_page}_{end_page - 1}",
        )

        return md_path, end_page, is_last

    # -------------------------------------------------------------------------
    # LlamaCloud upload → parse → poll → extract markdown
    # -------------------------------------------------------------------------

    async def _upload_and_parse(
        self,
        file_path: Path,
        page_range: Optional[str],
    ) -> str:
        """
        Step 1: Upload file to LlamaCloud Files API
        Step 2: Create a parsing job (with optional page_range)
        Step 3: Poll until COMPLETED / FAILED / CANCELLED
        Step 4: Fetch result with expand=["markdown"]
        Step 5: Extract and return full markdown string

        page_range: LlamaCloud 1-based format e.g. "1-100", "101-200", or None for images
        """
        # --- Upload ---
        with open(file_path, "rb") as f:
            file_obj = await self.client.files.create(
                file=(file_path.name, f, self._mime_type),
                purpose="parse",
            )

        # --- Create parse job ---
        parse_kwargs = dict(
            file_id=file_obj.id,
            tier=self.tier,
            version=self.version,
            output_options={
                "markdown": {
                    "tables": {
                        "output_tables_as_markdown": True
                    }
                }
            },
            timeout=self.timeout,
        )

        # Only add page_ranges for PDFs
        if page_range is not None:
            parse_kwargs["page_ranges"] = {"target_pages": page_range}  # type: ignore

        job = await self.client.parsing.create(**parse_kwargs)  # type: ignore

        # --- Poll for completion ---
        elapsed = 0.0
        while True:
            poll = await self.client.parsing.get(job.id)
            status = poll.job.status

            if status == "COMPLETED":
                break
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"LlamaCloud parse job {job.id} ended with status={status}. "
                    f"Error: {poll.job.error_message or 'no error message'}"
                )

            if elapsed >= self.timeout:
                raise TimeoutError(
                    f"LlamaCloud parse job {job.id} timed out after {self.timeout}s"
                )

            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval

        # --- Fetch result with markdown ---
        result = await self.client.parsing.get(job.id, expand=["markdown"])

        if result.markdown is None:
            raise RuntimeError(
                f"LlamaCloud job {job.id} completed but returned no markdown."
            )

        # --- Extract markdown from pages ---
        parts = []
        for page in result.markdown.pages:
            if not page.success:
                # Log warning but continue — partial results are still useful
                print(f"Warning: page {page.page_number} failed to parse, skipping.")
                continue
            if isinstance(page, MarkdownPageMarkdownResultPage):
                parts.append(page.markdown)

        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # File helpers
    # -------------------------------------------------------------------------

    def _save_markdown(self, markdown: str, stem: str, suffix: str) -> Path:
        """Writes markdown to temp/{stem}{suffix}.md and returns the path."""
        md_path = TEMP_DIR / f"{stem}{suffix}.md"
        md_path.write_text(markdown, encoding="utf-8")
        return md_path


async def main():
    start_page = 0
    while True:
        ocr = LlamaCloudOCRProcessor("/home/avvk/Graxon/Graxon/parser/test_data/some_page.pdf", "some_page", start_page=start_page, max_pages_per_chunk=3)
        # ocr = LlamaCloudOCRProcessor("/home/avvk/Graxon/Graxon/parser/test_data/image.png", "image", start_page=start_page)
        md_path, next_page, is_last = await ocr.process()

        print(f"Next page: {next_page}")
        print("Markdown Path", md_path)

        # # md_path → straight into MarkdownProcessor
        # md_proc = MarkdownProcessor(file_path=md_path, ...)
        # docs, next_idx, _ = await md_proc.process()
        # # → vector DB + Neo4j

        if is_last:
            break
        start_page = next_page


asyncio.run(main())
