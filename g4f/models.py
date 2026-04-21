from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .Provider import ProviderType, OpenaiProxy


class IterListProvider:
    def __init__(self, providers: list[ProviderType], shuffle: bool = False):
        self.providers = providers
        self.shuffle = shuffle


class ModelRegistry:
    _models: Dict[str, "Model"] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls, model: "Model", aliases: List[str] = None):
        if model.name:
            cls._models[model.name] = model
            if aliases:
                for alias in aliases:
                    cls._aliases[alias] = model.name

    @classmethod
    def get(cls, name: str) -> Optional["Model"]:
        if name in cls._models:
            return cls._models[name]
        if name in cls._aliases:
            return cls._models[cls._aliases[name]]
        return None

    @classmethod
    def all_models(cls) -> Dict[str, "Model"]:
        return cls._models.copy()


@dataclass(unsafe_hash=True)
class Model:
    name: str
    base_provider: str
    best_provider: ProviderType = None
    long_name: Optional[str] = None

    def get_long_name(self) -> str:
        return self.long_name if self.long_name else self.name

    def __post_init__(self):
        if self.name:
            ModelRegistry.register(self)

    @staticmethod
    def __all__() -> list[str]:
        return list(ModelRegistry.all_models().keys())


class ImageModel(Model):
    pass


class AudioModel(Model):
    pass


class VideoModel(Model):
    pass


class VisionModel(Model):
    pass


default = Model(
    name="",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)

default_vision = VisionModel(
    name="",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)

gpt_4o = VisionModel(
    name="gpt-4o",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)

gpt_4o_mini = Model(
    name="gpt-4o-mini",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)

gpt_4_1 = Model(
    name="gpt-4.1",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)

o3_mini = Model(
    name="o3-mini",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)

dall_e_3 = ImageModel(
    name="dall-e-3",
    base_provider="OpenAI Compatible Proxy",
    best_provider=OpenaiProxy,
)


class ModelUtils:
    convert: Dict[str, Model] = {}

    @classmethod
    def refresh(cls):
        cls.convert = ModelRegistry.all_models()

    @classmethod
    def get_model(cls, name: str) -> Optional[Model]:
        return ModelRegistry.get(name)

    @classmethod
    def register_alias(cls, alias: str, model_name: str):
        ModelRegistry._aliases[alias] = model_name


ModelRegistry._aliases["gpt4o"] = "gpt-4o"
ModelRegistry._aliases["gpt4.1"] = "gpt-4.1"
ModelUtils.convert = ModelRegistry.all_models()

demo_models = {}


def _get_working_providers(model: Model) -> List:
    if model.best_provider is None:
        return []
    return [model.best_provider] if model.best_provider.working else []


__models__ = {
    name: (model, _get_working_providers(model))
    for name, model in ModelRegistry.all_models().items()
    if name and _get_working_providers(model)
}

_all_models = list(__models__.keys())
Model.__all__ = staticmethod(lambda: _all_models)
