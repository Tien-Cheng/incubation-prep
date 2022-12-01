
from logging import Logger
from abc import ABC, abstractmethod
from typing import Optional
from docarray import DocumentArray


class Component(ABC):
    
    logger = Logger(__name__)

    @abstractmethod
    def __call__(
        self, data: DocumentArray, parameters: Optional[dict] = {}, **kwargs
    ) -> DocumentArray:
        return data
