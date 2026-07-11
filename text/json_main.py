import ijson
import json
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict, Any
import asyncio


class JsonProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_object: int,                  # 0-based object index to start from
        rag_chunk_start_index: int,         # absolute RAG chunk index to continue from
        objects_per_buffer: int = 500,      # max objects to read per batch
        max_chunk_size_mb: float = 50,      # hard size cap — stops before breaching
        group_size: int = 10,              # target objects per RAG chunk
        max_group_size: int = 20,          # hard cap — oversized clusters get split
    ):
        self.file_path = file_path
        self.filename = filename
        self.start_object = start_object
        self.rag_chunk_start_index = rag_chunk_start_index
        self.objects_per_buffer = objects_per_buffer
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.group_size = group_size
        self.max_group_size = max_group_size

        # Auto-detected on first use
        self._structure: Optional[str] = None   # "root_array" | "root_object" | "ndjson"
        self._ijson_prefix: Optional[str] = None  # e.g. "item" or "data.item"

    # -------------------------------------------------------------------------
    # Public API — same signature as CSVProcessor and ExcelProcessor
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Step 1: Auto-detect JSON structure (root array / root object / ndjson)
        Step 2: Stream objects from start_object, stop at objects_per_buffer OR 50MB
        Step 3: Flatten nested objects with dot notation  { "a": { "b": 1 } } → { "a.b": "1" }
        Step 4: TF-IDF + KMeans cluster flattened objects into semantic groups
        Step 5: Rebalance oversized clusters
        Step 6: Each group → one Document

        Returns:
            documents:             list of Document (one per semantic group)
            next_object_index:     pass as start_object to the next queue message
            is_last:               True if this was the final batch
        """
        try:
            self._structure, self._ijson_prefix = self._detect_structure()
            objects, total_bytes, is_last = self._stream_objects()
            documents = self._cluster_and_build_documents(objects)
            return documents, self.rag_chunk_start_index + len(documents), is_last

        except Exception as e:
            raise RuntimeError(f"Failed to process JSON file {self.file_path}: {e}") from e

    # -------------------------------------------------------------------------
    # Structure detection — reads only first 512 bytes + a few lines
    # -------------------------------------------------------------------------

    def _detect_structure(self) -> Tuple[str, str]:
        """
        Auto-detects top-level JSON structure and returns the ijson prefix to stream from.

        root_array   [ {}, {}, {} ]              → prefix = "item"
        root_object  { "data": [ {}, {} ] }      → prefix = "data.item"  (first array key found)
        ndjson       one JSON object per line     → prefix = "item" (special handling)

        Reads at most a few hundred bytes + 2 lines — no full file load.
        """
        with open(self.file_path, "rb") as f:
            chunk = f.read(512).lstrip()

        if chunk.startswith(b'['):
            return "root_array", "item"

        if chunk.startswith(b'{'):
            # Distinguish NDJSON from root object by checking if line 2 is also valid JSON
            with open(self.file_path, "r", encoding="utf-8") as f:
                line1 = f.readline().strip()
                line2 = f.readline().strip()
            try:
                json.loads(line1)
                json.loads(line2)
                return "ndjson", "item"
            except (json.JSONDecodeError, ValueError):
                pass

            # Root object — find the first array key using ijson parser
            with open(self.file_path, "rb") as f:
                for prefix, event, _ in ijson.parse(f):
                    if event == "start_array" and prefix:
                        return "root_object", f"{prefix}.item"

        raise ValueError(
            f"Unsupported JSON structure in {self.file_path}. "
            "Expected root array [ ], root object {{ }}, or NDJSON."
        )

    # -------------------------------------------------------------------------
    # Streaming layer — streams objects with count + size guard
    # -------------------------------------------------------------------------

    def _stream_objects(self) -> Tuple[List[Dict], int, bool]:
        """
        Streams JSON objects one by one via ijson (never loads full file).
        Stops when EITHER condition is met:
          - objects_per_buffer objects collected
          - next object would push accumulated size over max_chunk_size_bytes

        The object that would breach the size cap is NOT consumed —
        it becomes the first object of the next batch (start_object advances correctly).

        Returns:
            objects:     list of flattened dicts
            total_bytes: accumulated byte size of this batch
            is_last:     True if the stream was fully exhausted
        """
        collected: List[Dict] = []
        total_bytes = 0
        skipped = 0
        is_last = True  # assume last unless we hit a stopping condition early

        if self._structure == "ndjson":
            collected, total_bytes, is_last = self._stream_ndjson()
        else:
            with open(self.file_path, "rb") as f:
                for obj in ijson.items(f, self._ijson_prefix):
                    # Skip objects before start_object
                    if skipped < self.start_object:
                        skipped += 1
                        continue

                    flat = self._flatten(obj)
                    obj_text = ", ".join(f"{k}: {v}" for k, v in flat.items())
                    obj_bytes = len(obj_text.encode("utf-8"))

                    # Size guard — stop BEFORE adding this object
                    if total_bytes + obj_bytes > self.max_chunk_size_bytes:
                        is_last = False
                        break

                    collected.append(flat)
                    total_bytes += obj_bytes

                    # Count guard
                    if len(collected) >= self.objects_per_buffer:
                        is_last = False
                        break

        return collected, total_bytes, is_last

    def _stream_ndjson(self) -> Tuple[List[Dict], int, bool]:
        """Streams NDJSON (one JSON object per line) with the same count + size guards."""
        collected: List[Dict] = []
        total_bytes = 0
        is_last = True

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                if line_index < self.start_object:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

                flat = self._flatten(obj)
                obj_text = ", ".join(f"{k}: {v}" for k, v in flat.items())
                obj_bytes = len(obj_text.encode("utf-8"))

                # Size guard
                if total_bytes + obj_bytes > self.max_chunk_size_bytes:
                    is_last = False
                    break

                collected.append(flat)
                total_bytes += obj_bytes

                if len(collected) >= self.objects_per_buffer:
                    is_last = False
                    break

        return collected, total_bytes, is_last

    # -------------------------------------------------------------------------
    # Flattening — handles arbitrary nesting depth
    # -------------------------------------------------------------------------

    def _flatten(self, obj: Any, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
        """
        Recursively flattens nested dicts with dot notation.
        Arrays are summarized as count + first-element sample (avoids huge content).

        Examples:
          { "user": { "name": "Alice", "city": "Mumbai" } }
          → { "user.name": "Alice", "user.city": "Mumbai" }

          { "tags": ["python", "ml"] }
          → { "tags_count": "2", "tags_sample": "python" }
        """
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
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
    # Semantic clustering — identical to CSVProcessor and ExcelProcessor
    # -------------------------------------------------------------------------

    def _cluster_and_build_documents(self, objects: List[Dict]) -> List[Document]:
        if not objects:
            return []

        # Represent each object as a single string for TF-IDF
        row_texts = [
            " ".join(f"{k} {v}" for k, v in obj.items())
            for obj in objects
        ]

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

        # Group objects by cluster label
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(idx)

        # Rebalance oversized clusters
        groups = self._rebalance_clusters(clusters, objects)

        # Build Documents
        documents = []
        for group_objects, group_indices in groups:
            absolute_index = self.rag_chunk_start_index + len(documents)
            original_object_numbers = [self.start_object + i for i in group_indices]

            lines = [
                ", ".join(f"{k}: {v}" for k, v in obj.items())
                for obj in group_objects
            ]

            doc = Document(
                id=f"{self.filename}-{absolute_index}",
                page_content="\n".join(lines),
                metadata={
                    "source": self.file_path,
                    "structure": self._structure,
                    "rag_chunk_number": absolute_index,
                    "start_object": self.start_object,
                    "object_numbers": original_object_numbers,
                    "object_count": len(group_objects),
                },
            )
            documents.append(doc)

        return documents

    def _rebalance_clusters(
        self,
        clusters: Dict[int, List[int]],
        objects: List[Dict],
    ) -> List[Tuple[List[Dict], List[int]]]:
        """Splits oversized clusters into sub-chunks of max_group_size."""
        groups = []
        for label in sorted(clusters.keys()):
            indices = clusters[label]
            if len(indices) <= self.max_group_size:
                groups.append(([objects[i] for i in indices], indices))
            else:
                for start in range(0, len(indices), self.max_group_size):
                    sub = indices[start: start + self.max_group_size]
                    groups.append(([objects[i] for i in sub], sub))
        return groups


async def main():
    start_index = 0
    rag_chunk_index = 0

    while True:
        processor = JsonProcessor(
            "nested_test.json",
            "nested_test",
            start_index,
            rag_chunk_index,
            10,      # items per batch
            0.1     # size limit
        )

        documents, next_chunk_index, is_last = await processor.process()

        print(
            f"\nBatch Start={start_index} | "
            f"Documents={len(documents)} | "
            f"Is Last={is_last}"
        )

        for i, doc in enumerate(documents, start=1):
            print("\n" + "=" * 100)
            print(f"Document #{i}")
            print(f"ID: {doc.id}")
            print(f"Metadata: {doc.metadata}")
            print("#" * 100)
            print("content : ", doc.page_content[:1000])
            print("#" * 100)
            print("-" * 100)
            print(doc.page_content[:1000])  # avoid flooding terminal
            print("=" * 100)

        if is_last:
            print("\nFinished processing JSON file.")
            break

        start_index += 10
        rag_chunk_index = next_chunk_index

asyncio.run(main())
