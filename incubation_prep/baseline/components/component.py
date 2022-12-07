from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

from docarray import DocumentArray


class Component(ABC):

    logger = Logger(__name__)

    @abstractmethod
    def __call__(
        self, data: DocumentArray, parameters: Optional[dict] = {}, **kwargs
    ) -> DocumentArray:

        # TODO: If tensors not present, load

        # TODO: Check for dropped frames


        # TODO: with self.timer:
        # TODO: Main call function (give a different name)

        # Return output
        return data
