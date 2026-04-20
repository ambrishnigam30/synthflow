# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : DatasetService — upload, schema detection, NL queries
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import logging
import uuid
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import SupabaseStorageClient
from app.models.dataset import UploadedDataset
from app.models.user import User
from app.schemas.dataset import ColumnSummary, DatasetInfo, QueryResult

logger = logging.getLogger(__name__)

# 50 MB max upload size
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

_SAFE_OPERATIONS: dict[str, Any] = {
    "count":     lambda df, col: df[col].count() if col else len(df),
    "mean":      lambda df, col: df[col].mean(),
    "sum":       lambda df, col: df[col].sum(),
    "min":       lambda df, col: df[col].min(),
    "max":       lambda df, col: df[col].max(),
    "unique":    lambda df, col: df[col].nunique(),
    "head":      lambda df, col: df.head(10),
    "describe":  lambda df, col: df.describe().reset_index(),
    "groupby":   lambda df, col: df.groupby(col).size().reset_index(name="count"),
}


class DatasetFileTooLargeError(Exception):
    pass


class DatasetNotFoundError(Exception):
    pass


class DatasetService:
    def __init__(self, db: AsyncSession, storage: SupabaseStorageClient | None = None) -> None:
        self._db = db
        self._storage = storage or SupabaseStorageClient()

    # ── Upload ──────────────────────────────────────────────────────────────

    async def upload(
        self,
        file_bytes: bytes,
        filename: str,
        user: User,
        description: str | None = None,
    ) -> tuple[DatasetInfo, list[dict[str, Any]], list[ColumnSummary]]:
        """
        Parse, profile, and persist an uploaded dataset file.
        Returns (DatasetInfo, preview_rows, column_summaries).
        """
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise DatasetFileTooLargeError(
                f"File exceeds the 50 MB limit ({len(file_bytes) / 1024 / 1024:.1f} MB)."
            )

        df = self._parse_file(file_bytes, filename)
        column_summaries = self._profile_columns(df)
        schema_summary = {
            "columns": [cs.model_dump() for cs in column_summaries],
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        preview_rows = df.head(5).to_dict(orient="records")

        # Store in Supabase (graceful fallback)
        dataset_id = str(uuid.uuid4())
        storage_path = f"datasets/{user.id}/{dataset_id}/{filename}"
        try:
            await self._storage.upload_file("synthflow", storage_path, file_bytes)
        except Exception as exc:
            logger.warning("Storage upload skipped: %s", exc)
            storage_path = f"local/{dataset_id}/{filename}"

        record = UploadedDataset(
            id=dataset_id,
            user_id=user.id,
            name=filename,
            description=description,
            file_path=storage_path,
            file_size=len(file_bytes),
            row_count=len(df),
            column_count=len(df.columns),
            schema_summary=schema_summary,
        )
        self._db.add(record)
        await self._db.commit()
        await self._db.refresh(record)

        info = DatasetInfo.model_validate(record)
        return info, preview_rows, column_summaries

    def _parse_file(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """Parse CSV, XLSX, Parquet, or JSON into a DataFrame."""
        buf = io.BytesIO(file_bytes)
        name_lower = filename.lower()

        if name_lower.endswith(".csv"):
            return pd.read_csv(buf)
        if name_lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(buf)
        if name_lower.endswith(".parquet"):
            return pd.read_parquet(buf)
        if name_lower.endswith(".json") or name_lower.endswith(".jsonl"):
            return pd.read_json(buf)
        # Fallback: try CSV
        return pd.read_csv(buf)

    def _profile_columns(self, df: pd.DataFrame) -> list[ColumnSummary]:
        """Compute per-column statistics."""
        summaries: list[ColumnSummary] = []
        for col in df.columns:
            series = df[col]
            try:
                samples = series.dropna().head(5).tolist()
                samples = [str(s) if not isinstance(s, (int, float, bool)) else s for s in samples]
            except Exception:
                samples = []
            summaries.append(
                ColumnSummary(
                    name=str(col),
                    dtype=str(series.dtype),
                    null_count=int(series.isna().sum()),
                    unique_count=int(series.nunique()),
                    sample_values=samples,
                )
            )
        return summaries

    # ── Query (NL → pandas) ────────────────────────────────────────────────

    async def query(
        self,
        dataset_id: str,
        question: str,
        user: User,
        max_rows: int = 100,
    ) -> QueryResult:
        """
        Translate a natural language question into a safe pandas operation
        and return the result.
        """
        dataset = await self._get_dataset(dataset_id, user.id)
        df = await self._load_dataframe(dataset)
        operation, col = self._parse_question(question, df.columns.tolist())

        try:
            op_fn = _SAFE_OPERATIONS.get(operation, _SAFE_OPERATIONS["head"])
            raw = op_fn(df, col)

            if isinstance(raw, pd.DataFrame):
                result_df = raw.head(max_rows)
                data = result_df.to_dict(orient="records")
                columns = result_df.columns.tolist()
            elif isinstance(raw, pd.Series):
                result_df = raw.reset_index()
                data = result_df.head(max_rows).to_dict(orient="records")
                columns = result_df.columns.tolist()
            else:
                data = [{"result": raw}]
                columns = ["result"]

            return QueryResult(
                question=question,
                operation=f"{operation}({col or ''})",
                data=[{str(k): v for k, v in row.items()} for row in data],
                row_count=len(data),
                columns=[str(c) for c in columns],
                summary=f"{operation.capitalize()} of '{col}': {raw if not isinstance(raw, (pd.DataFrame, pd.Series)) else f'{len(data)} rows'}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Dataset query error: %s", exc)
            # Fallback to showing first rows
            preview = df.head(min(max_rows, 5)).to_dict(orient="records")
            return QueryResult(
                question=question,
                operation="head(5)",
                data=preview,
                row_count=len(preview),
                columns=df.columns.tolist(),
                summary=f"Could not compute '{operation}'. Showing first rows instead.",
            )

    def _parse_question(self, question: str, columns: list[str]) -> tuple[str, str | None]:
        """
        Map a natural language question to a (operation, column) pair.
        This is a keyword-based router; production would use an LLM.
        """
        q = question.lower()
        found_col: str | None = None
        for col in columns:
            if col.lower() in q:
                found_col = col
                break

        if any(kw in q for kw in ("average", "mean", "avg")):
            return "mean", found_col
        if any(kw in q for kw in ("total", "sum")):
            return "sum", found_col
        if any(kw in q for kw in ("count", "how many")):
            return "count", found_col
        if any(kw in q for kw in ("minimum", "min", "lowest", "smallest")):
            return "min", found_col
        if any(kw in q for kw in ("maximum", "max", "highest", "largest")):
            return "max", found_col
        if any(kw in q for kw in ("unique", "distinct", "different")):
            return "unique", found_col
        if any(kw in q for kw in ("distribution", "group", "breakdown")):
            return "groupby", found_col
        if any(kw in q for kw in ("summary", "statistics", "describe", "stats")):
            return "describe", None
        return "head", None

    async def _load_dataframe(self, dataset: UploadedDataset) -> pd.DataFrame:
        """
        Load the dataset into memory.
        Tries Supabase Storage; falls back to empty DF with schema_summary columns.
        """
        try:
            file_bytes = await self._storage.download_file("synthflow", dataset.file_path)
            return self._parse_file(file_bytes, dataset.name)
        except Exception as exc:
            logger.warning("Could not load file from storage (%s); building stub df.", exc)
            schema = dataset.schema_summary or {}
            cols = [c["name"] for c in schema.get("columns", [])] if schema else ["id", "value"]
            return pd.DataFrame(columns=cols)

    async def get_stats(self, dataset_id: str, user_id: str) -> dict[str, Any]:
        """Return per-column statistics for a dataset."""
        dataset = await self._get_dataset(dataset_id, user_id)
        df = await self._load_dataframe(dataset)
        summaries = self._profile_columns(df)
        return {"columns": [cs.model_dump() for cs in summaries], "row_count": len(df)}

    async def delete(self, dataset_id: str, user: User) -> None:
        """Remove the DB record and the file from storage."""
        dataset = await self._get_dataset(dataset_id, user.id)
        try:
            await self._storage.delete_file("synthflow", dataset.file_path)
        except Exception as exc:
            logger.warning("File delete from storage failed: %s", exc)
        await self._db.delete(dataset)
        await self._db.commit()

    async def list_datasets(self, user_id: str) -> list[UploadedDataset]:
        stmt = select(UploadedDataset).where(UploadedDataset.user_id == user_id).order_by(UploadedDataset.created_at.desc())
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_dataset_info(self, dataset_id: str, user_id: str) -> UploadedDataset:
        return await self._get_dataset(dataset_id, user_id)

    async def _get_dataset(self, dataset_id: str, user_id: str) -> UploadedDataset:
        stmt = select(UploadedDataset).where(
            UploadedDataset.id == dataset_id,
            UploadedDataset.user_id == user_id,
        )
        result = await self._db.execute(stmt)
        dataset = result.scalar_one_or_none()
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found.")
        return dataset
