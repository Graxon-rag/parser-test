import lxml.etree as etree
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import numpy as np
import asyncio


# Tags treated as standalone content blocks
CONTENT_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "article", "section", "li"}
# Minimum characters for a content block to be worth indexing
MIN_CONTENT_LENGTH = 30


class HTMLProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_unit: int,                  # 0-based index into extracted units (content blocks + table rows)
        rag_chunk_start_index: int,       # absolute RAG chunk index to continue from
        units_per_buffer: int = 500,      # max units to read per batch
        max_chunk_size_mb: float = 50,
        group_size: int = 10,             # target units per RAG chunk
        max_group_size: int = 20,         # hard cap — oversized clusters get split
    ):
        self.file_path = file_path
        self.filename = filename
        self.start_unit = start_unit
        self.rag_chunk_start_index = rag_chunk_start_index
        self.units_per_buffer = units_per_buffer
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.group_size = group_size
        self.max_group_size = max_group_size

    # -------------------------------------------------------------------------
    # Public API — same signature as all other processors
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Step 1: Stream HTML with lxml iterparse — extract content blocks and table rows
        Step 2: Apply start_unit offset + dual guard (count + size)
        Step 3: TF-IDF + KMeans cluster units into semantic groups
        Step 4: Rebalance oversized clusters
        Step 5: Each group → one Document

        A "unit" is either:
          - A content block: text from <p>, <h1-h6>, <article>, <section>
          - A table row: dict of {header: cell_value} from <tr>/<td>

        Returns:
            documents:            list of Document (one per semantic group)
            next_unit_index:      pass as start_unit to the next queue message
            is_last:              True if this was the final batch
        """
        try:
            units, is_last = self._stream_units()
            documents = self._cluster_and_build_documents(units)
            return documents, self.rag_chunk_start_index + len(documents), is_last
        except Exception as e:
            raise RuntimeError(f"Failed to process HTML file {self.file_path}: {e}") from e

    # -------------------------------------------------------------------------
    # Streaming layer — extracts content blocks + table rows via lxml iterparse
    # -------------------------------------------------------------------------

    def _stream_units(self) -> Tuple[List[Dict], bool]:
        """
        Streams the HTML file tag by tag (never loads full file into memory).
        Extracts two types of units:
          - content: { "type": "content", "tag": "p", "text": "..." }
          - row:     { "type": "row", "Name": "iPhone", "Brand": "Apple", ... }

        Stops when EITHER:
          - units_per_buffer units collected after start_unit offset
          - next unit would push accumulated size over max_chunk_size_bytes
        """
        units = []
        total_bytes = 0
        skipped = 0
        is_last = True

        # State for table parsing
        current_headers: List[str] = []
        current_row_cells: List[str] = []
        in_table = False

        parser = etree.iterparse(
            self.file_path,
            events=("start", "end"),
            html=True,       # HTML mode — handles malformed tags, missing closing tags
            recover=True,    # recover from broken HTML gracefully
        )

        done = False
        for event, elem in parser:
            if done:
                break

            tag = elem.tag.lower() if isinstance(elem.tag, str) else ""

            if event == "start":
                if tag == "table":
                    in_table = True
                    current_headers = []

            elif event == "end":
                # --- Content blocks ---
                if tag in CONTENT_TAGS and not in_table:
                    text = "".join(elem.itertext()).strip()
                    if len(text) >= MIN_CONTENT_LENGTH:
                        unit = {"type": "content", "tag": tag, "text": text}
                        unit_text = text
                        unit_bytes = len(unit_text.encode("utf-8"))

                        if skipped < self.start_unit:
                            skipped += 1
                        elif total_bytes + unit_bytes > self.max_chunk_size_bytes:
                            is_last = False
                            done = True
                        else:
                            units.append(unit)
                            total_bytes += unit_bytes
                            if len(units) >= self.units_per_buffer:
                                is_last = False
                                done = True

                # --- Table headers ---
                elif tag == "th":
                    header_text = "".join(elem.itertext()).strip()
                    if header_text:
                        current_headers.append(header_text)

                # --- Table cells ---
                elif tag == "td":
                    cell_text = "".join(elem.itertext()).strip()
                    current_row_cells.append(cell_text)

                # --- End of table row ---
                elif tag == "tr":
                    if current_row_cells:
                        # Map cells to headers (pad with col_N if headers missing)
                        row_dict = {"type": "row"}
                        for i, cell in enumerate(current_row_cells):
                            header = current_headers[i] if i < len(current_headers) else f"col_{i}"
                            row_dict[header] = cell

                        unit_text = " ".join(f"{k}: {v}" for k, v in row_dict.items() if k != "type")
                        unit_bytes = len(unit_text.encode("utf-8"))

                        if skipped < self.start_unit:
                            skipped += 1
                        elif total_bytes + unit_bytes > self.max_chunk_size_bytes:
                            is_last = False
                            done = True
                        else:
                            units.append(row_dict)
                            total_bytes += unit_bytes
                            if len(units) >= self.units_per_buffer:
                                is_last = False
                                done = True

                        current_row_cells = []

                # --- End of table ---
                elif tag == "table":
                    in_table = False
                    current_headers = []

                elem.clear()

        return units, is_last

    # -------------------------------------------------------------------------
    # Semantic clustering — same TF-IDF + KMeans as CSV/Excel/JSON processors
    # -------------------------------------------------------------------------

    def _cluster_and_build_documents(self, units: List[Dict]) -> List[Document]:
        if not units:
            return []

        # Represent each unit as a string for TF-IDF
        row_texts = []
        for u in units:
            if u["type"] == "content":
                row_texts.append(f"{u['tag']} {u['text']}")
            else:
                row_texts.append(" ".join(f"{k} {v}" for k, v in u.items() if k != "type"))

        n_rows = len(row_texts)
        n_clusters = max(1, round(n_rows / self.group_size))

        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            max_features=1000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(row_texts)

        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Group by cluster
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(idx)

        groups = self._rebalance_clusters(clusters, units)

        documents = []
        for group_units, group_indices in groups:
            absolute_index = self.rag_chunk_start_index + len(documents)
            unit_numbers = [self.start_unit + i for i in group_indices]

            # Build page_content — content blocks as prose, table rows as key: value lines
            lines = []
            for u in group_units:
                if u["type"] == "content":
                    lines.append(f"[{u['tag'].upper()}] {u['text']}")
                else:
                    lines.append(", ".join(f"{k}: {v}" for k, v in u.items() if k != "type"))

            doc = Document(
                id=f"{self.filename}-{absolute_index}",
                page_content="\n".join(lines),
                metadata={
                    "source": self.file_path,
                    "rag_chunk_number": absolute_index,
                    "start_unit": self.start_unit,
                    "unit_numbers": unit_numbers,
                    "unit_count": len(group_units),
                    "has_content": any(u["type"] == "content" for u in group_units),
                    "has_table_rows": any(u["type"] == "row" for u in group_units),
                },
            )
            documents.append(doc)

        return documents

    def _rebalance_clusters(self, clusters: Dict, units: List[Dict]):
        groups = []
        for label in sorted(clusters.keys()):
            indices = clusters[label]
            if len(indices) <= self.max_group_size:
                groups.append(([units[i] for i in indices], indices))
            else:
                for start in range(0, len(indices), self.max_group_size):
                    sub = indices[start: start + self.max_group_size]
                    groups.append(([units[i] for i in sub], sub))
        return groups


async def process_html():
    start_unit = 0
    rag_chunk_index = 0

    while True:
        processor = HTMLProcessor(
            file_path="test_products.html",
            filename="test_products.html",
            start_unit=start_unit,
            rag_chunk_start_index=rag_chunk_index,
            max_chunk_size_mb=0.01,
        )

        documents, next_unit, is_last = await processor.process()

        print("=" * 80)
        print(f"Units: {start_unit} -> {next_unit}")
        print(f"Generated {len(documents)} chunks")

        for i, doc in enumerate(documents):
            print(f"\nChunk {i + 1}")
            print(doc)

        if is_last:
            print("\n✅ HTML processing completed.")
            break

        start_unit = next_unit
        rag_chunk_index += len(documents)

asyncio.run(process_html())
