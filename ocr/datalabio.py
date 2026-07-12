from datalab_sdk.models import ConversionResult
from datalab_sdk import AsyncDatalabClient
from pypdf import PdfReader, PdfWriter
from typing import Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path
import mimetypes
import asyncio
import shutil
import os


load_dotenv()

# Image MIME types — processed in one shot (no splitting)
IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/tiff", "image/bmp", "image/gif",
}

TEMP_DIR = Path("temp/datalab")


class DatalabOCRProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_page: int = 0,               # 0-based page index (PDF only, ignored for images)
        max_pages_per_chunk: int = 100,    # max pages per convert() call
        max_chunk_size_mb: float = 30,     # stop before exceeding this per chunk
        datalab_api_key: Optional[str] = None,
        timeout: int = 60 * 10,            # seconds (datalab uses seconds, not ms)
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.start_page = start_page
        self.max_pages_per_chunk = max_pages_per_chunk
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.timeout = timeout

        api_key = datalab_api_key or os.getenv("DATALAB_API_KEY")
        if not api_key:
            raise ValueError("DATALAB_API_KEY not set. Pass datalab_api_key= or set the env var.")

        self.client = AsyncDatalabClient(api_key=api_key, timeout=timeout)

        self._mime_type = mimetypes.guess_type(str(self.file_path))[0] or "application/octet-stream"
        self._is_image = self._mime_type in IMAGE_MIME_TYPES

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API — same signature as OCRProcessor (Mistral)
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[Path, int, bool]:
        """
        Single image → convert whole file at once → save markdown → return path.
        PDF         → split pages start_page : start_page + max_pages_per_chunk
                      (capped at max_chunk_size_mb) → convert → save markdown.

        Returns:
            md_path:    Path to saved .md file in temp/
                        → pass directly to MarkdownProcessor
            next_page:  pass as start_page to next queue message (PDF only, 0 for images)
            is_last:    True = no more pages remain
        """
        if self._is_image:
            return await self._process_image()
        else:
            return await self._process_pdf()

    # -------------------------------------------------------------------------
    # Image — single shot, no splitting needed
    # -------------------------------------------------------------------------

    async def _process_image(self) -> Tuple[Path, int, bool]:
        """
        Passes the image directly to datalab convert().
        Always returns is_last=True — images have no pages to paginate.
        """
        markdown = await self._convert(str(self.file_path))

        md_path = self._save_markdown(
            markdown=markdown,
            stem=self.file_path.stem,
            suffix="",
        )

        return md_path, 0, True

    # -------------------------------------------------------------------------
    # PDF — split by page range + size cap, then convert
    # -------------------------------------------------------------------------

    async def _process_pdf(self) -> Tuple[Path, int, bool]:
        """
        Splits the PDF into a page-range slice respecting both:
          - max_pages_per_chunk (count guard)
          - max_chunk_size_mb   (size guard — stops before the page that would breach it)

        Saves the slice as temp/{stem}_pages_{start}_{end}.pdf,
        converts it via datalab, saves markdown as temp/{stem}_pages_{start}_{end}.md.
        """
        reader = PdfReader(str(self.file_path))
        total_pages = len(reader.pages)

        if self.start_page >= total_pages:
            raise ValueError(
                f"start_page={self.start_page} is out of range. "
                f"PDF has {total_pages} pages."
            )

        # Build page-range slice with dual guard
        writer = PdfWriter()
        accumulated_bytes = 0
        end_page = self.start_page

        for page_idx in range(self.start_page, total_pages):
            page = reader.pages[page_idx]

            # Estimate this page's size before adding it
            page_bytes = self._estimate_page_size(page)

            # Size guard — stop before adding a page that would breach the cap
            if accumulated_bytes + page_bytes > self.max_chunk_size_bytes and writer.pages:
                break

            writer.add_page(page)
            accumulated_bytes += page_bytes
            end_page = page_idx + 1  # exclusive end

            # Page count guard
            if (end_page - self.start_page) >= self.max_pages_per_chunk:
                break

        is_last = end_page >= total_pages

        # Save the PDF slice to temp/
        pdf_path = self._save_pdf_chunk(writer, start=self.start_page, end=end_page)

        # Convert via datalab SDK
        markdown = await self._convert(str(pdf_path))

        # Save markdown to temp/
        md_path = self._save_markdown(
            markdown=markdown,
            stem=self.file_path.stem,
            suffix=f"_pages_{self.start_page}_{end_page - 1}",
        )

        return md_path, end_page, is_last

    # -------------------------------------------------------------------------
    # Datalab convert — handles both result types from the SDK
    # -------------------------------------------------------------------------

    async def _convert(self, file_path: str) -> str:
        """
        Calls datalab client.convert() and normalises both return types:

          ConversionResult with .markdown  → return markdown string directly
          ConversionResult with .output_path → read the file and return its text

        Mirrors your reference code exactly.
        """
        result = await self.client.convert(file_path)

        if result.error:
            raise RuntimeError(f"Datalab conversion failed: {result.error}")

        if isinstance(result, ConversionResult):
            if result.markdown is not None:
                # SDK returned markdown directly
                return result.markdown
            else:
                raise RuntimeError("Datalab SDK returned ConversionResult with empty markdown.")
        else:
            # SDK returned a file path to the generated markdown
            output_path = Path(result.output_path)
            return output_path.read_text(encoding="utf-8")

    # -------------------------------------------------------------------------
    # File helpers
    # -------------------------------------------------------------------------

    def _estimate_page_size(self, page) -> int:
        """Estimate byte size of a single PDF page via in-memory write."""
        import io
        probe = PdfWriter()
        probe.add_page(page)
        buf = io.BytesIO()
        probe.write(buf)
        return buf.tell()

    def _save_pdf_chunk(self, writer: PdfWriter, start: int, end: int) -> Path:
        """
        Writes the PDF slice to temp/{stem}_pages_{start}_{end-1}.pdf
        Kept in temp even after conversion (same behaviour as OCRProcessor).
        """
        pdf_path = TEMP_DIR / f"{self.file_path.stem}_pages_{start}_{end - 1}.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return pdf_path

    def _save_markdown(self, markdown: str, stem: str, suffix: str) -> Path:
        """Writes markdown to temp/{stem}{suffix}.md and returns the path."""
        md_path = TEMP_DIR / f"{stem}{suffix}.md"
        md_path.write_text(markdown, encoding="utf-8")
        return md_path


async def main():
    start_page = 0
    while True:
        ocr = DatalabOCRProcessor("/home/avvk/Graxon/Graxon/parser/test_data/some_page.pdf", "some_page", start_page=start_page, max_pages_per_chunk=3)
        # ocr = DatalabOCRProcessor("/home/avvk/Graxon/Graxon/parser/test_data/image.png", "image", start_page=start_page)
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
