from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from src.dependencies.index_dependencies import get_search_service
from src.domain.dataclasses.dataclasses import (
    Document,
    SearchRequestDataClass,
    SimilarityRequestDataClass,
)
from src.domain.schemas.requests import IndexRequest, SearchRequest, SimilarityRequest
from src.exceptions.exceptions import AppError
from src.infrastructure.logger import setup_logger
from src.service.search_service import SearchService

logger = setup_logger(name="logger")

app = FastAPI()


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.exception("Exception: %s", exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            },
        },
    )


@app.post(
    "/index/document",
    status_code=202,
)
def index(
    documents: list[IndexRequest],
    search_service: Annotated[SearchService, Depends(get_search_service)],
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    docs = [Document(**doc.model_dump()) for doc in documents]
    background_tasks.add_task(search_service.index_documents, docs)
    return {"status": "success"}


@app.post("/search/semantic")
def semantic_search(
    request: SearchRequest,
    search_service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, Any]:
    request_data = SearchRequestDataClass(**request.model_dump())
    return search_service.semantic_search(request=request_data)


@app.post("/search/conversational")
def generative_search(
    request: SearchRequest,
    search_service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, Any]:
    request_data = SearchRequestDataClass(**request.model_dump())
    return search_service.conversational_search(request=request_data)


@app.post("/search/similar")
def similar_search(
    request: SimilarityRequest,
    search_service: Annotated[SearchService, Depends(get_search_service)],
) -> dict[str, Any]:
    request_data = SimilarityRequestDataClass(**request.model_dump())
    return search_service.similar_search(request=request_data)
