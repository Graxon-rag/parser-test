import pandas as pd
import numpy as np
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional
import asyncio


class ExcelProcessor:
    def __init__(
        self,
        file_path: str,
        filename: str,
        start_row: int,                        # 0-based row index (excluding header)
        rag_chunk_start_index: int,            # absolute RAG chunk index to continue from
        sheet: Optional[str | int] = 0,        # sheet name or 0-based index (default: first sheet)
        rows_per_io_buffer: int = 500,         # rows to read from disk at once
        max_chunk_size_mb: float = 50,
        group_size: int = 10,                  # target rows per RAG chunk
        max_group_size: int = 20,              # hard cap — oversized clusters get split
    ):
        self.file_path = file_path
        self.filename = filename
        self.start_row = start_row
        self.rag_chunk_start_index = rag_chunk_start_index
        self.sheet = sheet
        self.rows_per_io_buffer = rows_per_io_buffer
        self.max_chunk_size_bytes = int(max_chunk_size_mb * 1024 * 1024)
        self.group_size = group_size
        self.max_group_size = max_group_size

    # -------------------------------------------------------------------------
    # Public API — same signature as CSVProcessor
    # -------------------------------------------------------------------------

    async def process(self) -> Tuple[List[Document], int, bool]:
        """
        Step 1: Read up to rows_per_io_buffer rows from the sheet at start_row offset
        Step 2: Cluster rows into semantic groups of ~group_size using TF-IDF + KMeans
        Step 3: Rebalance oversized clusters (> max_group_size) by splitting with iloc
        Step 4: Each cluster group → one Document for vector DB + Neo4j

        Returns:
            documents:             list of Document (one per semantic group)
            next_rag_chunk_index:  pass to the next queue message as rag_chunk_start_index
            is_last:               True if this was the final IO buffer for this sheet
        """
        try:
            df, is_last = self._read_chunk()
            documents = self._cluster_and_build_documents(df)
            return documents, self.rag_chunk_start_index + len(documents), is_last

        except Exception as e:
            raise RuntimeError(f"Failed to process Excel file {self.file_path}: {e}") from e

    @staticmethod
    def get_sheet_names(file_path: str) -> List[str]:
        """
        Returns all sheet names in the workbook.
        Use this to decide which sheets to process before enqueuing jobs.

        Example:
            sheets = ExcelProcessor.get_sheet_names("report.xlsx")
            # ["Sales", "Inventory", "Summary"]
        """
        xl = pd.ExcelFile(file_path, engine="calamine")
        return xl.sheet_names

    # -------------------------------------------------------------------------
    # IO layer — reads rows_per_io_buffer rows from the sheet at start_row offset
    # -------------------------------------------------------------------------

    def _read_chunk(self) -> Tuple[pd.DataFrame, bool]:
        """
        Uses calamine engine (Rust-based, 3-5x faster than openpyxl) with
        pandas skiprows + nrows to read only the requested slice — no full sheet load.
        """
        total_rows = self._count_total_rows()

        df = pd.read_excel(
            self.file_path,
            engine="calamine",                          # fast Rust reader
            sheet_name=self.sheet,
            skiprows=range(1, self.start_row + 1),      # skip rows before start_row, keep header
            nrows=self.rows_per_io_buffer,
            dtype=str,                                  # no type coercion
            keep_default_na=False,                      # empty cells stay as "" not NaN
        )

        # Drop completely empty rows (common in Excel files with formatting artifacts)
        df = df.replace("", np.nan).dropna(how="all").fillna("").reset_index(drop=True)

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
        """Fast row count — reads only the first column via calamine."""
        return len(pd.read_excel(
            self.file_path,
            engine="calamine",
            sheet_name=self.sheet,
            usecols=[0],
            dtype=str,
        ))

    # -------------------------------------------------------------------------
    # Semantic clustering layer — identical logic to CSVProcessor
    # -------------------------------------------------------------------------

    def _cluster_and_build_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        TF-IDF + KMeans clustering — groups rows with similar column values
        into semantic chunks without needing any domain knowledge or column names.
        """
        if df.empty:
            return []

        row_texts = df.apply(
            lambda row: " ".join(f"{col} {val}" for col, val in row.items()),
            axis=1,
        ).tolist()

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

        df = df.copy()
        df["_cluster"] = cluster_labels
        df["_original_row"] = range(self.start_row, self.start_row + n_rows)

        groups = self._rebalance_clusters(df)

        documents = []
        for group in groups:
            absolute_index = self.rag_chunk_start_index + len(documents)
            original_rows = group["_original_row"].tolist()

            clean = group.drop(columns=["_cluster", "_original_row"])
            lines = [
                ", ".join(f"{col}: {val}" for col, val in row.items())
                for _, row in clean.iterrows()
            ]

            doc = Document(
                id=f"{self.filename}-sheet{self.sheet}-{absolute_index}",
                page_content="\n".join(lines),
                metadata={
                    "source": self.file_path,
                    "sheet": self.sheet,                  # sheet name or index
                    "rag_chunk_number": absolute_index,
                    "start_row": self.start_row,
                    "row_numbers": original_rows,
                    "row_count": len(original_rows),
                },
            )
            documents.append(doc)

        return documents

    def _rebalance_clusters(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Splits any cluster larger than max_group_size into equal iloc sub-chunks."""
        groups = []
        for cluster_id in sorted(df["_cluster"].unique()):
            cluster_df = df[df["_cluster"] == cluster_id].reset_index(drop=True)
            if len(cluster_df) <= self.max_group_size:
                groups.append(cluster_df)
            else:
                for start in range(0, len(cluster_df), self.max_group_size):
                    groups.append(cluster_df.iloc[start: start + self.max_group_size])
        return groups


async def main():
    sheet = 0
    start_row = 1
    rag_chunk_index = 0

    while True:
        documents, rag_chunk_index, is_last = await ExcelProcessor(
            file_path="test_multisheet.xlsx",
            filename="test_multisheet",
            start_row=start_row,
            rag_chunk_start_index=rag_chunk_index,
            sheet=sheet,
            rows_per_io_buffer=50
        ).process()

        print(
            f"\nSheet={sheet} | Start Row={start_row} | "
            f"Documents={len(documents)} | Is Last={is_last}"
        )

        for doc in documents:
            print(
                f"Content={doc.page_content}"
                f"Chunk={doc.metadata.get('rag_chunk_number')} "
                f"Rows={doc.metadata.get('row_numbers')} "
                f"Count={doc.metadata.get('row_count')}"
            )

        if is_last:
            break

        # Adjust based on your processor's chunk size
        start_row += 50

asyncio.run(main())
