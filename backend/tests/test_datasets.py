# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : B-004 Datasets integration tests
# ───────────────────────────────────────────────────────────────

import io

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio

# ── Helpers ───────────────────────────────────────────────────────────────────

_USER = {"email": "dsuser@synthflow.ai", "password": "Pass1234!", "full_name": "DS User"}

# Minimal CSV content with named columns
_CSV_CONTENT = (
    "name,age,salary,department\n"
    "Alice,30,75000,Engineering\n"
    "Bob,25,55000,Marketing\n"
    "Carol,35,90000,Engineering\n"
    "David,28,62000,Sales\n"
    "Eve,32,80000,Engineering\n"
)


async def _signup_get_token(client: AsyncClient) -> tuple[str, str]:
    """Sign up and return (access_token, user_id)."""
    resp = await client.post("/api/auth/signup", json=_USER)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    return data["access_token"], data["user"]["id"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _upload_csv(
    client: AsyncClient, token: str, content: str = _CSV_CONTENT, filename: str = "test.csv"
) -> dict:
    """Upload a CSV and return the response data dict."""
    resp = await client.post(
        "/api/datasets/upload",
        files={"file": (filename, io.BytesIO(content.encode()), "text/csv")},
        headers=_auth(token),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


# ── Tests ──────────────────────────────────────────────────────────────────────


async def test_b004_01_upload_csv(client: AsyncClient) -> None:
    """B-004-01: POST /api/datasets/upload with CSV → 201 + dataset info."""
    token, _ = await _signup_get_token(client)

    resp = await client.post(
        "/api/datasets/upload",
        files={"file": ("employees.csv", io.BytesIO(_CSV_CONTENT.encode()), "text/csv")},
        headers=_auth(token),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "dataset" in data
    dataset = data["dataset"]
    assert "id" in dataset
    assert dataset["name"] == "employees.csv"
    assert dataset["file_size"] > 0
    assert isinstance(data["preview_rows"], list)
    assert isinstance(data["columns"], list)


async def test_b004_02_upload_detects_schema(client: AsyncClient) -> None:
    """B-004-02: Uploaded dataset has schema_summary with detected columns."""
    token, _ = await _signup_get_token(client)
    data = await _upload_csv(client, token)

    dataset = data["dataset"]
    assert dataset["row_count"] == 5
    assert dataset["column_count"] == 4
    assert dataset["schema_summary"] is not None

    # Column summaries must include the CSV column names
    columns = data["columns"]
    col_names = {c["name"] for c in columns}
    assert "name" in col_names
    assert "age" in col_names
    assert "salary" in col_names
    assert "department" in col_names

    # Preview rows should be non-empty
    assert len(data["preview_rows"]) > 0


async def test_b004_03_query_returns_result(client: AsyncClient) -> None:
    """B-004-03: POST /api/datasets/{id}/query with natural language → result."""
    token, _ = await _signup_get_token(client)
    data = await _upload_csv(client, token)
    dataset_id: str = data["dataset"]["id"]

    resp = await client.post(
        f"/api/datasets/{dataset_id}/query",
        json={"question": "average salary"},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    result_wrapper = body["data"]
    result = result_wrapper["result"]

    # Must echo the question
    assert result["question"] == "average salary"
    # Must have a columns list
    assert isinstance(result["columns"], list)
    # Must have a data list
    assert isinstance(result["data"], list)
    # Must have an operation identifier
    assert isinstance(result["operation"], str)
    # execution_time_ms must be a non-negative integer
    assert result_wrapper["execution_time_ms"] >= 0


async def test_b004_04_upload_size_limit(client: AsyncClient) -> None:
    """B-004-04: Uploading a file over 50 MB → 413."""
    token, _ = await _signup_get_token(client)

    # 50 MB + 1 byte — exceeds the limit
    large_bytes = b"x" * (50 * 1024 * 1024 + 1)

    resp = await client.post(
        "/api/datasets/upload",
        files={"file": ("huge.csv", io.BytesIO(large_bytes), "text/csv")},
        headers=_auth(token),
    )
    assert resp.status_code == 413


async def test_b004_05_delete_dataset(client: AsyncClient) -> None:
    """B-004-05: DELETE /api/datasets/{id} → 204 and dataset gone."""
    token, _ = await _signup_get_token(client)
    data = await _upload_csv(client, token)
    dataset_id: str = data["dataset"]["id"]

    resp = await client.delete(f"/api/datasets/{dataset_id}", headers=_auth(token))
    assert resp.status_code == 204

    # Confirm the dataset is no longer retrievable
    resp = await client.get(f"/api/datasets/{dataset_id}", headers=_auth(token))
    assert resp.status_code == 404
