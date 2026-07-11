from pathlib import Path
from datalab_sdk.models import ConversionResult
from datalab_sdk import AsyncDatalabClient
from dotenv import load_dotenv
import asyncio
import os
import shutil

load_dotenv()


async def main():
    api_key = os.getenv("DATALAB_API_KEY")

    input_file = Path("./test_data/image.png")
    output_dir = Path("output")

    output_dir = Path("output") / "datalabio"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_md = output_dir / f"{input_file.stem}.md"

    client = AsyncDatalabClient(
        api_key=api_key,
        timeout=60 * 10,
    )

    result = await client.convert(str(input_file))
    if result.error:
        raise Exception(result.error)

    if isinstance(result, ConversionResult):
        # SDK returned markdown directly
        if result.markdown is None:
            raise Exception("SDK returned empty markdown")
        output_md.write_text(result.markdown, encoding="utf-8")
        print(f"Markdown saved to: {output_md}")

    else:
        # SDK returned a generated markdown file
        output_path = Path(result.output_path)

        if output_path.suffix.lower() == ".md":
            shutil.copy2(output_path, output_md)
        else:
            # Fallback: read as text and save as .md
            output_md.write_text(
                output_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        print(f"Markdown saved to: {output_md}")


if __name__ == "__main__":
    asyncio.run(main())
