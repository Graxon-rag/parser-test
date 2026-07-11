import yaml
import numpy as np
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
import asyncio


MULTI_DOC = "multi_doc"
SEQUENCE = "sequence"
MAPPING = "mapping"


class YAMLProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_object: int,
        rag_chunk_start_index: int,
        objects_per_buffer: int = 500,
        max_chunk_size_mb: float = 50,
        group_size: int = 10,
        max_group_size: int = 20,
        scan_lines: int = 100,
    ):
        self.file_path = file_path
        self.filename = filename
        self.start_object = start_object
        self.rag_chunk_start_index = rag_chunk_start_index
        self.objects_per_buffer = objects_per_buffer
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.group_size = group_size
        self.max_group_size = max_group_size
        self.scan_lines = scan_lines
        self._structure: Optional[str] = None

    # -------------------------------------------------------------------------
    # Public API — same signature as all other processors
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Step 1: Auto-detect YAML structure (multi_doc / sequence / mapping)
        Step 2: Stream records line-by-line, parse only small individual blocks
                with yaml.safe_load — never loads the full file into memory
        Step 3: Apply start_object offset + dual guard (count + 50MB size cap)
        Step 4: Flatten nested keys with dot notation
        Step 5: TF-IDF + KMeans cluster records into semantic groups
        Step 6: Rebalance oversized clusters
        Step 7: Each group → one Document

        Returns:
            documents:           list of Document (one per semantic group)
            next_object_index:   pass as start_object to the next queue message
            is_last:             True if this was the final batch
        """
        try:
            self._structure = self._detect_structure()
            records, is_last = self._stream_records()
            documents = self._cluster_and_build_documents(records)
            return documents, self.rag_chunk_start_index + len(documents), is_last
        except Exception as e:
            raise RuntimeError(f"Failed to process YAML file {self.file_path}: {e}") from e

    @staticmethod
    def detect_structure(file_path: str, scan_lines: int = 100) -> str:
        """
        Public static — call before enqueuing to inspect detected structure.
        Scans only the first scan_lines lines (never loads full file).
        Returns one of: "multi_doc", "sequence", "mapping"
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= scan_lines:
                    break
                if line.strip() == "---":
                    return MULTI_DOC
                if line.startswith("- "):
                    return SEQUENCE
        return MAPPING

    def _detect_structure(self) -> str:
        return YAMLProcessor.detect_structure(self.file_path, self.scan_lines)

    # -------------------------------------------------------------------------
    # Streaming layer
    # -------------------------------------------------------------------------

    def _stream_records(self) -> Tuple[List[Dict], bool]:
        if self._structure == MULTI_DOC:
            return self._stream_multidoc()
        elif self._structure == SEQUENCE:
            return self._stream_sequence()
        else:
            return self._stream_mapping()

    def _stream_multidoc(self) -> Tuple[List[Dict], bool]:
        """
        Splits on "---" separators.
        Each block between separators is one record.

            ---
            name: iPhone    ← block 1
            brand: Apple
            ---             ← safe boundary
            name: Galaxy    ← block 2
        """
        collected: List[Dict] = []
        total_bytes = 0
        skipped = 0
        is_last = True
        current_lines: List[str] = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "---":
                    cont, skipped, total_bytes, is_last = self._flush_block(
                        "\n".join(current_lines), collected, skipped, total_bytes, is_last
                    )
                    current_lines = []
                    if not cont:
                        break
                else:
                    current_lines.append(line.rstrip())
            else:
                # Flush the last block
                self._flush_block(
                    "\n".join(current_lines), collected, skipped, total_bytes, is_last
                )

        return collected, is_last

    def _stream_sequence(self) -> Tuple[List[Dict], bool]:
        """
        Splits on "- " at column 0 — each such line starts a new record.
        Collects all indented lines that follow as part of that record.

            - name: iPhone      ← col-0 "- " = record boundary
              brand: Apple
              details:
                stock: 50
            - name: Galaxy      ← next record
        """
        collected: List[Dict] = []
        total_bytes = 0
        skipped = 0
        is_last = True
        current_lines: List[str] = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("- ") and current_lines:
                    cont, skipped, total_bytes, is_last = self._flush_block(
                        "\n".join(current_lines), collected, skipped, total_bytes, is_last
                    )
                    current_lines = [line.rstrip()]
                    if not cont:
                        break
                else:
                    current_lines.append(line.rstrip())
            else:
                self._flush_block(
                    "\n".join(current_lines), collected, skipped, total_bytes, is_last
                )

        return collected, is_last

    def _stream_mapping(self) -> Tuple[List[Dict], bool]:
        """
        Splits on top-level keys (zero indentation, contains colon, not a comment).
        Each top-level key + its indented children = one record.

            database:           ← top-level key = record boundary
              host: localhost
              port: 5432
            app:                ← next record
              debug: true
        """
        collected: List[Dict] = []
        total_bytes = 0
        skipped = 0
        is_last = True
        current_key: Optional[str] = None
        current_lines: List[str] = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                is_top_key = (
                    line
                    and line[0] not in (" ", "\t", "#", "\n", "-")
                    and ":" in line
                )
                if is_top_key:
                    if current_key is not None:
                        block = f"{current_key}:\n" + "\n".join(current_lines)
                        cont, skipped, total_bytes, is_last = self._flush_block(
                            block, collected, skipped, total_bytes, is_last
                        )
                        if not cont:
                            break
                    current_key = line.split(":")[0].strip()
                    current_lines = []
                else:
                    current_lines.append(line.rstrip())
            else:
                if current_key:
                    block = f"{current_key}:\n" + "\n".join(current_lines)
                    self._flush_block(block, collected, skipped, total_bytes, is_last)

        return collected, is_last

    # -------------------------------------------------------------------------
    # Shared flush — parses one block, applies offset + dual guard
    # -------------------------------------------------------------------------

    def _flush_block(
        self,
        block: str,
        collected: List[Dict],
        skipped: int,
        total_bytes: int,
        is_last: bool,
    ) -> Tuple[bool, int, int, bool]:
        """
        Parses a YAML block string, applies start_object offset and dual guard.

        Returns: (continue_streaming, skipped, total_bytes, is_last)
          continue_streaming=False → caller should stop the loop
        """
        if not block.strip():
            return True, skipped, total_bytes, is_last

        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError:
            return True, skipped, total_bytes, is_last

        # Normalise: sequence block parses as a list — extract first item
        if isinstance(parsed, list) and parsed:
            obj = parsed[0]
        elif isinstance(parsed, dict):
            obj = parsed
        else:
            return True, skipped, total_bytes, is_last

        if not isinstance(obj, dict):
            return True, skipped, total_bytes, is_last

        # Offset — skip records before start_object
        if skipped < self.start_object:
            return True, skipped + 1, total_bytes, is_last

        flat = self._flatten(obj)
        text = ", ".join(f"{k}: {v}" for k, v in flat.items())
        obj_bytes = len(text.encode("utf-8"))

        # Size guard — stop BEFORE adding this record
        if total_bytes + obj_bytes > self.max_chunk_size_bytes:
            return False, skipped, total_bytes, False

        collected.append(flat)
        total_bytes += obj_bytes

        # Count guard
        if len(collected) >= self.objects_per_buffer:
            return False, skipped, total_bytes, False

        return True, skipped, total_bytes, is_last

    # -------------------------------------------------------------------------
    # Flatten — same dot-notation approach as JSON/XML processors
    # -------------------------------------------------------------------------

    def _flatten(self, obj, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
        """
        Recursively flattens nested dicts with dot notation.
        Lists → count + first-element sample.

        { "details": { "stock": 50 } }  →  { "details.stock": "50" }
        { "tags": ["ml", "ai"] }         →  { "tags_count": "2", "tags_sample": "ml" }
        """
        items: Dict[str, str] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                if isinstance(v, dict):
                    items.update(self._flatten(v, new_key, sep))
                elif isinstance(v, list):
                    items[f"{new_key}_count"] = str(len(v))
                    if v and not isinstance(v[0], (dict, list)):
                        items[f"{new_key}_sample"] = str(v[0])
                else:
                    items[new_key] = str(v) if v is not None else ""
        return items

    # -------------------------------------------------------------------------
    # Semantic clustering — identical to all other processors
    # -------------------------------------------------------------------------

    def _cluster_and_build_documents(self, records: List[Dict]) -> List[Document]:
        if not records:
            return []

        row_texts = [
            " ".join(f"{k} {v}" for k, v in rec.items())
            for rec in records
        ]

        n_clusters = max(1, round(len(row_texts) / self.group_size))

        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            max_features=1000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(row_texts)

        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(idx)

        groups = self._rebalance_clusters(clusters, records)

        documents = []
        for group_records, group_indices in groups:
            absolute_index = self.rag_chunk_start_index + len(documents)
            doc = Document(
                id=f"{self.filename}-{absolute_index}",
                page_content="\n".join(
                    ", ".join(f"{k}: {v}" for k, v in rec.items())
                    for rec in group_records
                ),
                metadata={
                    "source": self.file_path,
                    "structure": self._structure,
                    "rag_chunk_number": absolute_index,
                    "start_object": self.start_object,
                    "object_numbers": [self.start_object + i for i in group_indices],
                    "object_count": len(group_records),
                },
            )
            documents.append(doc)

        return documents

    def _rebalance_clusters(self, clusters: Dict, records: List[Dict]):
        groups = []
        for label in sorted(clusters.keys()):
            indices = clusters[label]
            if len(indices) <= self.max_group_size:
                groups.append(([records[i] for i in indices], indices))
            else:
                for start in range(0, len(indices), self.max_group_size):
                    sub = indices[start: start + self.max_group_size]
                    groups.append(([records[i] for i in sub], sub))
        return groups


async def process_yaml():
    start_object = 0
    rag_chunk_index = 0

    while True:
        processor = YAMLProcessor(
            file_path="test_products.yaml",
            filename="test_products.yaml",
            start_object=start_object,
            rag_chunk_start_index=rag_chunk_index,
        )

        documents, next_object, is_last = await processor.process()

        print("=" * 80)
        print(f"Objects: {start_object} -> {next_object}")
        print(f"Generated {len(documents)} chunks")

        for i, doc in enumerate(documents):
            print(f"\nChunk {i + 1}")
            print(doc)

        if is_last:
            print("\n✅ YAML processing completed.")
            break

        start_object = next_object
        rag_chunk_index += len(documents)


asyncio.run(process_yaml())
