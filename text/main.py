import pandas as pd
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Tuple
import asyncio

class CSVProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_row: int,                   # 0-based row index (excluding header)
        rag_chunk_start_index: int,       # absolute RAG chunk index to continue from
        rows_per_io_buffer: int = 500,    # rows to read from disk at once (IO buffer)
        max_chunk_size_mb: float = 50,
        group_size: int = 10,             # target rows per RAG chunk
        max_group_size: int = 20,         # oversized clusters get split at this threshold
    ):
        self.file_path = file_path
        self.filename = filename
        self.start_row = start_row
        self.rag_chunk_start_index = rag_chunk_start_index
        self.rows_per_io_buffer = rows_per_io_buffer
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.group_size = group_size
        self.max_group_size = max_group_size

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Step 1: Read up to rows_per_io_buffer rows from disk (e.g. 500 rows)
        Step 2: Cluster those rows into semantic groups of ~group_size using TF-IDF + KMeans
        Step 3: Rebalance oversized clusters (> max_group_size) by splitting them
        Step 4: Each cluster group → one Document for vector DB + Neo4j

        Returns:
            documents:             list of Document (one per semantic group)
            next_rag_chunk_index:  pass to the next queue message as rag_chunk_start_index
            is_last:               True if this was the final IO buffer
        """
        try:
            df, is_last = self._read_chunk()
            documents = self._cluster_and_build_documents(df)
            return documents, self.rag_chunk_start_index + len(documents), is_last

        except Exception as e:
            raise RuntimeError(f"Failed to process CSV {self.file_path}: {e}") from e

    # -------------------------------------------------------------------------
    # IO layer — reads rows_per_io_buffer rows from disk at start_row offset
    # -------------------------------------------------------------------------

    def _read_chunk(self) -> Tuple[pd.DataFrame, bool]:
        total_rows = self._count_total_rows()

        df = pd.read_csv(
            self.file_path,
            skiprows=range(1, self.start_row + 1),  # skip rows before start_row, keep header
            nrows=self.rows_per_io_buffer,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            on_bad_lines="warn",
        )

        # Enforce size cap
        if not df.empty:
            row_sizes = df.apply(
                lambda row: len(
                    ", ".join(f"{col}: {val}" for col, val in row.items()).encode("utf-8")
                ),
                axis=1,
            )
            within_limit = row_sizes.cumsum() <= self.max_chunk_size_bytes
            if not within_limit.all():
                df = df[within_limit]

        is_last = (self.start_row + len(df)) >= total_rows
        return df, is_last

    def _count_total_rows(self) -> int:
        return len(pd.read_csv(
            self.file_path,
            usecols=[0],
            dtype=str,
            encoding="utf-8-sig",
            on_bad_lines="warn",
        ))

    # -------------------------------------------------------------------------
    # Semantic clustering layer
    # -------------------------------------------------------------------------

    def _cluster_and_build_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Converts each row to a text representation, runs TF-IDF + KMeans to find
        semantically similar rows, rebalances oversized clusters, then builds Documents.

        Why TF-IDF + KMeans:
          - Zero embedding API cost — purely local computation
          - Fast even on 500 rows
          - Groups rows that share similar column values/vocabulary together
          - No domain knowledge or specific grouping column needed
        """
        if df.empty:
            return []

        # Represent each row as: "col1 val1 col2 val2 ..." (no punctuation, just terms)
        row_texts = df.apply(
            lambda row: " ".join(f"{col} {val}" for col, val in row.items()),
            axis=1,
        ).tolist()

        n_rows = len(row_texts)
        n_clusters = max(1, round(n_rows / self.group_size))

        # --- TF-IDF vectorization ---
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            max_features=1000,
            sublinear_tf=True,       # log(1+tf) — dampens very frequent terms
        )
        tfidf_matrix = vectorizer.fit_transform(row_texts)

        # --- KMeans clustering ---
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=5,
            random_state=42,
        )
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Attach cluster + original row number to df
        df = df.copy()
        df["_cluster"] = cluster_labels
        df["_original_row"] = range(self.start_row, self.start_row + n_rows)

        # --- Rebalance: split oversized clusters into equal sub-chunks ---
        groups = self._rebalance_clusters(df)

        # --- Build one Document per group ---
        documents = []
        for group in groups:
            absolute_index = self.rag_chunk_start_index + len(documents)
            original_rows = group["_original_row"].tolist()

            clean = group.drop(columns=["_cluster", "_original_row"])
            lines = [
                ", ".join(f"{col}: {val}" for col, val in row.items())
                for _, row in clean.iterrows()
            ]
            page_content = "\n".join(lines)

            doc = Document(
                id=f"{self.filename}-{absolute_index}",
                page_content=page_content,
                metadata={
                    "source": self.file_path,
                    "rag_chunk_number": absolute_index,
                    "start_row": self.start_row,
                    "row_numbers": original_rows,
                    "row_count": len(original_rows),
                },
            )
            documents.append(doc)

        return documents

    def _rebalance_clusters(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits any cluster larger than max_group_size into equal sub-chunks.
        This keeps every RAG chunk within a predictable size range.

        Example: cluster of 35 rows with max_group_size=20
          → splits into [18, 17] sub-chunks
        """
        groups = []
        for cluster_id in sorted(df["_cluster"].unique()):
            cluster_df = df[df["_cluster"] == cluster_id].reset_index(drop=True)

            if len(cluster_df) <= self.max_group_size:
                groups.append(cluster_df)
            else:
                # Split into sub-chunks of max_group_size rows each
                for start in range(0, len(cluster_df), self.max_group_size):
                    groups.append(cluster_df.iloc[start: start + self.max_group_size])

        return groups


async def main():
    start_row = 0
    rag_chunk_index = 0

    while True:
        documents, next_rag_chunk_index, is_last = await CSVProcessor(
            file_path="test.csv",
            filename="test",
            start_row=start_row,
            rag_chunk_start_index=rag_chunk_index,
            rows_per_io_buffer=50,
        ).process()

        for doc in documents:
            print("\n" + "=" * 100)
            print(f"Document ID: {doc.id}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 100)
            print(doc.page_content)
            print("=" * 100 + "\n")

        if is_last:
            break

        start_row += 50
        rag_chunk_index = next_rag_chunk_index

asyncio.run(main())
