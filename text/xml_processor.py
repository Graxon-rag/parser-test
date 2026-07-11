import lxml.etree as etree
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
import asyncio


class XMLProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_object: int,                      # 0-based record index to start from
        rag_chunk_start_index: int,             # absolute RAG chunk index to continue from
        record_tag: Optional[str] = None,       # repeating element tag — auto-detected if None
        objects_per_buffer: int = 500,          # max records per batch
        max_chunk_size_mb: float = 50,
        group_size: int = 10,                   # target records per RAG chunk
        max_group_size: int = 20,               # hard cap — oversized clusters get split
    ):
        self.file_path = file_path
        self.filename = filename
        self.start_object = start_object
        self.rag_chunk_start_index = rag_chunk_start_index
        self.record_tag = record_tag            # set by auto-detect if not provided
        self.objects_per_buffer = objects_per_buffer
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.group_size = group_size
        self.max_group_size = max_group_size

    # -------------------------------------------------------------------------
    # Public API — same signature as all other processors
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Step 1: Auto-detect repeating record tag if not provided (reads file once lightly)
        Step 2: Stream records from start_object — stop at count OR size cap
        Step 3: Flatten nested tags with dot notation  <details><category> → details.category
        Step 4: TF-IDF + KMeans cluster records into semantic groups
        Step 5: Rebalance oversized clusters
        Step 6: Each group → one Document

        Returns:
            documents:            list of Document (one per semantic group)
            next_object_index:    pass as start_object to the next queue message
            is_last:              True if this was the final batch
        """
        try:
            if not self.record_tag:
                self.record_tag = self._detect_record_tag()

            records, is_last = self._stream_records()
            documents = self._cluster_and_build_documents(records)
            return documents, self.rag_chunk_start_index + len(documents), is_last

        except Exception as e:
            raise RuntimeError(f"Failed to process XML file {self.file_path}: {e}") from e

    @staticmethod
    def detect_record_tag(file_path: str) -> str:
        """
        Public static method — call this if you want to inspect the detected tag
        before enqueuing jobs, e.g. to log or override it.

        Strategy: find the tag that:
          1. Appears many times (it's repeating)
          2. Has multiple distinct child tags (it's a container, not a leaf value)
        """
        parent_child_map: Dict[str, set] = defaultdict(set)
        parent_count: Counter = Counter()
        stack: List[str] = []

        ctx = etree.iterparse(file_path, events=("start", "end"), recover=True)
        for event, elem in ctx:
            if event == "start":
                stack.append(elem.tag)
            elif event == "end":
                if len(stack) >= 2:
                    parent_child_map[stack[-2]].add(elem.tag)
                    parent_count[stack[-2]] += 1
                if stack:
                    stack.pop()
                elem.clear()

        # Record tag = most frequent tag with 2+ distinct child types
        candidates = [
            (tag, count, len(parent_child_map[tag]))
            for tag, count in parent_count.items()
            if len(parent_child_map[tag]) >= 2
        ]
        if not candidates:
            raise ValueError(f"Could not auto-detect a repeating record tag in {file_path}. "
                             "Pass record_tag explicitly.")

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # -------------------------------------------------------------------------
    # Private detect (used internally during process())
    # -------------------------------------------------------------------------

    def _detect_record_tag(self) -> str:
        return XMLProcessor.detect_record_tag(self.file_path)

    # -------------------------------------------------------------------------
    # Streaming layer — streams records with count + size guard
    # -------------------------------------------------------------------------

    def _stream_records(self) -> Tuple[List[Dict], bool]:
        """
        Streams XML with lxml iterparse — event-based, never loads full tree.
        Collects direct + one-level-deep children of record_tag as a flat dict.

        Stops when EITHER:
          - objects_per_buffer records collected (after start_object offset)
          - next record would push accumulated size over max_chunk_size_bytes
        """
        collected: List[Dict] = []
        total_bytes = 0
        skipped = 0
        is_last = True

        current_record: Optional[Dict] = None
        stack: List[str] = []
        in_record = False

        ctx = etree.iterparse(self.file_path, events=("start", "end"), recover=True)
        done = False

        for event, elem in ctx:
            if done:
                break

            tag = elem.tag

            if event == "start":
                stack.append(tag)
                if tag == self.record_tag:
                    in_record = True
                    current_record = {}

            elif event == "end":
                if tag == self.record_tag:
                    # Record complete — apply offset + guards
                    if current_record is not None:
                        if skipped < self.start_object:
                            skipped += 1
                        else:
                            obj_text = ", ".join(f"{k}: {v}" for k, v in current_record.items())
                            obj_bytes = len(obj_text.encode("utf-8"))

                            if total_bytes + obj_bytes > self.max_chunk_size_bytes:
                                is_last = False
                                done = True
                            else:
                                collected.append(current_record)
                                total_bytes += obj_bytes
                                if len(collected) >= self.objects_per_buffer:
                                    is_last = False
                                    done = True

                    in_record = False
                    current_record = None

                elif in_record and current_record is not None:
                    text = "".join(elem.itertext()).strip()
                    if text:
                        depth = len(stack)
                        record_depth = stack.index(self.record_tag) if self.record_tag in stack else -1

                        if depth == record_depth + 2:
                            # Direct child of record tag
                            current_record[tag] = text
                        elif depth == record_depth + 3:
                            # One level nested: <details><category> → details.category
                            parent = stack[-2] if len(stack) >= 2 else "nested"
                            current_record[f"{parent}.{tag}"] = text

                if stack:
                    stack.pop()
                elem.clear()

        return collected, is_last

    # -------------------------------------------------------------------------
    # Semantic clustering — same TF-IDF + KMeans as all other processors
    # -------------------------------------------------------------------------

    def _cluster_and_build_documents(self, records: List[Dict]) -> List[Document]:
        if not records:
            return []

        row_texts = [
            " ".join(f"{k} {v}" for k, v in rec.items())
            for rec in records
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

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(idx)

        groups = self._rebalance_clusters(clusters, records)

        documents = []
        for group_records, group_indices in groups:
            absolute_index = self.rag_chunk_start_index + len(documents)
            object_numbers = [self.start_object + i for i in group_indices]

            lines = [
                ", ".join(f"{k}: {v}" for k, v in rec.items())
                for rec in group_records
            ]

            doc = Document(
                id=f"{self.filename}-{absolute_index}",
                page_content="\n".join(lines),
                metadata={
                    "source": self.file_path,
                    "record_tag": self.record_tag,
                    "rag_chunk_number": absolute_index,
                    "start_object": self.start_object,
                    "object_numbers": object_numbers,
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


async def process_xml():
    start_object = 0
    rag_chunk_index = 0

    while True:
        processor = XMLProcessor(
            file_path="test_products.xml",
            filename="test_products.xml",
            start_object=start_object,
            rag_chunk_start_index=rag_chunk_index,
            max_chunk_size_mb=0.01,
        )

        documents, next_object, is_last = await processor.process()

        print("=" * 80)
        print(f"Objects: {start_object} -> {next_object}")
        print(f"Documents: {len(documents)}")

        for doc in documents:
            print(doc)

        if is_last:
            print("\nXML processing complete.")
            break

        start_object = next_object
        rag_chunk_index += len(documents)


asyncio.run(process_xml())
