from abc import ABC


class LLMGeneratorBase(ABC):
    def __init__(self, model: str):
        self.model = model

    def generate(self, **kwargs) -> list[str]:
        raise NotImplementedError("LLM generate method is not implemented")
