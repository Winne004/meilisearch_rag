from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    id: str
    url: str | None = None
    body: str


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, gt=0, lt=20)
