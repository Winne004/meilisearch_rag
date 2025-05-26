# src/domain/interfaces/llm.py
from abc import ABC, abstractmethod
from typing import Any


class LLMABC(ABC):
    @abstractmethod
    def extract_keywords(self, query: str) -> str: ...

    @abstractmethod
    def summarise(self, query: str, results: dict[str, Any]) -> str: ...
