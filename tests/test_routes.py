from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.dependencies.index_dependencies import get_dependencies
from src.domain.schemas.requests import IndexRequest, SearchRequest
from src.exceptions.exceptions import AppError
from src.service.search_service import SearchService


# --- Fixtures ---
@pytest.fixture
def mock_search_service() -> MagicMock:
    mock = MagicMock(spec=SearchService)
    mock.semantic_search.return_value = {"results": ["semantic"]}
    mock.conversational_search.return_value = {"results": ["generative"]}
    mock.index_documents.return_value = None
    return mock


@pytest.fixture
def client(mock_search_service: MagicMock) -> Generator[TestClient, Any]:
    app.dependency_overrides[get_dependencies] = lambda: mock_search_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --- Tests ---
def test_index_documents(client: TestClient) -> None:
    payload = [
        IndexRequest(
            id="1",
            url="www.test.com",
            body="body",
        ).model_dump(),
    ]

    response = client.post("/index/document", json=payload)
    assert response.status_code == 202
    assert response.json() == {"status": "success"}


def test_index_multiple_documents(client: TestClient) -> None:
    payload = [
        IndexRequest(
            id="1",
            url=f"https://example.com/{i}",
            body="text",
        ).model_dump()
        for i in range(3)
    ]

    response = client.post("/index/document", json=payload)
    assert response.status_code == 202
    assert response.json() == {"status": "success"}


def test_semantic_search_limit_zero(client: TestClient) -> None:
    payload = {"query": "test", "limit": 0}
    response = client.post("search/semantic", json=payload)
    assert response.status_code == 422


def test_request_validation_fail(client: TestClient) -> None:
    payload = [{"title": "missing fields"}]
    response = client.post("/index/document", json=payload)
    assert response.status_code == 422


def test_semantic_search(client: TestClient) -> None:
    payload = SearchRequest(query="test", limit=5).model_dump()

    response = client.post("search/semantic", json=payload)
    assert response.status_code == 200
    assert response.json() == {"results": ["semantic"]}


def test_generative_search(client: TestClient) -> None:
    payload = {"query": "latest AI news", "filters": {}, "top_k": 3}

    response = client.post("search/conversational", json=payload)
    assert response.status_code == 200
    assert response.json() == {"results": ["generative"]}


def test_error_handling(client: TestClient, mock_search_service: MagicMock) -> None:
    mock_search_service.semantic_search.side_effect = AppError(
        message="Invalid query",
    )

    payload = {"query": "", "filters": {}, "top_k": 3}

    response = client.post("search/semantic", json=payload)
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Invalid query"},
    }
