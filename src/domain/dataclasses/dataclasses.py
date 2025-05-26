from dataclasses import dataclass


@dataclass
class Document:
    id: str
    body: str
    url: str | None = None


@dataclass
class VectorisedDocument:
    vector: list[float]
    id: str
    chunk: str
    url: str | None = None


@dataclass
class SearchRequestDataClass:
    query: str
    limit: int


@dataclass
class SimilarityRequestDataClass:
    id: str | int
    limit: int
