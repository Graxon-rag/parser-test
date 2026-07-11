from pathlib import Path
from mistralai.client.models import File
from mistralai.client import Mistral
from dotenv import load_dotenv
import asyncio
import mimetypes
import os

load_dotenv()


async def main():
    api_key = os.getenv("MISTRAL_API_KEY")

    input_file = Path("./test_data/image.png")
    output_file = Path("output") / "mistral" / f"{input_file.stem}.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    client = Mistral(api_key=api_key)

    timeout_ms = 1000 * 60 * 10

    # Detect MIME type
    content_type = mimetypes.guess_type(input_file)[0] or "application/octet-stream"

    # Upload file
    uploaded = await client.files.upload_async(
        file=File(
            file_name=input_file.name,
            content=input_file.read_bytes(),
            content_type=content_type,
        ),
        purpose="ocr",
        timeout_ms=timeout_ms,
    )

    # Run OCR
    result = await client.ocr.process_async(
        model="mistral-ocr-latest",
        timeout_ms=timeout_ms,
        document={
            "type": "file",
            "file_id": uploaded.id,
        },
    )

    # Combine markdown from all pages
    markdown = "\n\n".join(page.markdown for page in result.pages)

    output_file.write_text(markdown, encoding="utf-8")

    print(f"Saved OCR markdown to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
