from mistralai.client.models import File
from pypdf import PdfReader, PdfWriter
from mistralai.client import Mistral
from typing import Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path
import mimetypes
import asyncio
import os

load_dotenv()

# Image MIME types treated as single-shot (no splitting)
IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/tiff", "image/bmp", "image/gif",
}

# Where all temp files live
TEMP_DIR = Path("temp")


class OCRProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_page: int = 0,               # 0-based page index (PDF only, ignored for images)
        max_pages_per_chunk: int = 100,    # Mistral recommended max per call
        max_chunk_size_mb: float = 30,     # Mistral hard limit is 50MB — stay safe at 30
        mistral_api_key: Optional[str] = None,
        timeout_ms: int = 1000 * 60 * 10,  # 10 minutes
    ):
        self.file_path = Path(file_path)
        self.filename = filename
        self.start_page = start_page
        self.max_pages_per_chunk = max_pages_per_chunk
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.timeout_ms = timeout_ms

        api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set. Pass mistral_api_key= or set the env var.")
        self.client = Mistral(api_key=api_key)

        self._mime_type = mimetypes.guess_type(str(self.file_path))[0] or "application/octet-stream"
        self._is_image = self._mime_type in IMAGE_MIME_TYPES

        # Ensure temp dir exists
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[Path, int, bool]:
        """
        Single image → OCR the whole file at once.
        PDF         → Split pages start_page : start_page + max_pages_per_chunk
                      (capped at max_chunk_size_mb), upload, OCR.

        Returns:
            md_path:        Path to the saved markdown file in temp/
                            → pass directly to MarkdownProcessor
            next_page:      pass as start_page to the next queue message (PDF only)
                            → always 0 for images (single shot)
            is_last:        True = no more pages remain, this was the final batch
        """
        if self._is_image:
            return await self._process_image()
        else:
            return await self._process_pdf()

    # -------------------------------------------------------------------------
    # Image — process entire file in one shot
    # -------------------------------------------------------------------------

    async def _process_image(self) -> Tuple[Path, int, bool]:
        """
        Images have no pages to split — upload and OCR the whole file.
        Always returns is_last=True.
        """
        file_bytes = self.file_path.read_bytes()

        markdown = await self._upload_and_ocr(
            file_bytes=file_bytes,
            upload_filename=self.file_path.name,
            mime_type=self._mime_type,
        )

        md_path = self._save_markdown(
            markdown=markdown,
            stem=self.file_path.stem,
            suffix="",           # no chunk suffix for single image
        )

        return md_path, 0, True

    # -------------------------------------------------------------------------
    # PDF — split by page range + size cap, then OCR
    # -------------------------------------------------------------------------

    async def _process_pdf(self) -> Tuple[Path, int, bool]:
        """
        Reads the PDF, extracts pages start_page → start_page + max_pages_per_chunk
        (stops earlier if accumulated size exceeds max_chunk_size_mb).

        Saves the page-range slice as a temp PDF, uploads it, runs OCR,
        saves the resulting markdown, and returns the path + next_page + is_last.
        """
        reader = PdfReader(str(self.file_path))
        total_pages = len(reader.pages)

        if self.start_page >= total_pages:
            raise ValueError(
                f"start_page={self.start_page} is out of range. "
                f"PDF has {total_pages} pages."
            )

        # Build the page-range slice respecting both count and size caps
        writer = PdfWriter()
        accumulated_bytes = 0
        end_page = self.start_page  # exclusive end index — incremented below

        for page_idx in range(self.start_page, total_pages):
            page = reader.pages[page_idx]

            # Estimate page size by writing it alone to a temp buffer
            probe = PdfWriter()
            probe.add_page(page)
            page_bytes = self._estimate_writer_size(probe)

            # Size cap — stop before adding this page
            if accumulated_bytes + page_bytes > self.max_chunk_size_bytes and writer.pages:
                break

            writer.add_page(page)
            accumulated_bytes += page_bytes
            end_page = page_idx + 1  # exclusive

            # Page count cap
            if (end_page - self.start_page) >= self.max_pages_per_chunk:
                break

        is_last = end_page >= total_pages

        # Save the PDF slice to temp/
        pdf_path = self._save_pdf_chunk(writer, start=self.start_page, end=end_page)

        # Upload and OCR the slice
        file_bytes = pdf_path.read_bytes()
        markdown = await self._upload_and_ocr(
            file_bytes=file_bytes,
            upload_filename=pdf_path.name,
            mime_type="application/pdf",
        )

        # Save markdown to temp/
        md_path = self._save_markdown(
            markdown=markdown,
            stem=self.file_path.stem,
            suffix=f"_pages_{self.start_page}_{end_page - 1}",
        )

        return md_path, end_page, is_last

    # -------------------------------------------------------------------------
    # Mistral upload + OCR
    # -------------------------------------------------------------------------

    async def _upload_and_ocr(
        self,
        file_bytes: bytes,
        upload_filename: str,
        mime_type: str,
    ) -> str:
        """
        Uploads file_bytes to Mistral Files API, runs OCR,
        returns the full markdown string (all pages joined).
        """
        uploaded = await self.client.files.upload_async(
            file=File(
                file_name=upload_filename,
                content=file_bytes,
                content_type=mime_type,
            ),
            purpose="ocr",
            timeout_ms=self.timeout_ms,
        )

        result = await self.client.ocr.process_async(
            model="mistral-ocr-latest",
            timeout_ms=self.timeout_ms,
            document={
                "type": "file",
                "file_id": uploaded.id,
            },
        )

        # Join markdown from all pages with double newline separator
        return "\n\n".join(page.markdown for page in result.pages)

    # -------------------------------------------------------------------------
    # File helpers
    # -------------------------------------------------------------------------

    def _estimate_writer_size(self, writer: PdfWriter) -> int:
        """Estimate byte size of a PdfWriter by writing to an in-memory buffer."""
        import io
        buf = io.BytesIO()
        writer.write(buf)
        return buf.tell()

    def _save_pdf_chunk(self, writer: PdfWriter, start: int, end: int) -> Path:
        """
        Writes the PDF slice to temp/{stem}_pages_{start}_{end-1}.pdf
        Returns the path.
        """
        pdf_path = TEMP_DIR / f"{self.file_path.stem}_pages_{start}_{end - 1}.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return pdf_path

    def _save_markdown(self, markdown: str, stem: str, suffix: str) -> Path:
        """
        Writes markdown to temp/{stem}{suffix}.md
        Returns the path.
        """
        md_path = TEMP_DIR / f"{stem}{suffix}.md"
        md_path.write_text(markdown, encoding="utf-8")
        return md_path


async def main():
    start_page = 0
    while True:
        ocr = OCRProcessor("/home/avvk/Graxon/Graxon/parser/test_data/some_page.pdf", "some_page", start_page=start_page, max_pages_per_chunk=3)
        # ocr = OCRProcessor("/home/avvk/Graxon/Graxon/parser/test_data/some_page.pdf", "some_page", start_page=start_page, max_pages_per_chunk=1)
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
