from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .Provider import Chatai, EncryptedProxy, IterListProvider, ProviderType


class ModelRegistry:
    """Registry for automatic model discovery."""

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

    @classmethod
    def clear(cls):
        cls._models.clear()
        cls._aliases.clear()


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
    base_provider="",
    best_provider=IterListProvider([EncryptedProxy, Chatai], shuffle=False),
)

default_vision = VisionModel(
    name="",
    base_provider="",
    best_provider=IterListProvider([EncryptedProxy], shuffle=False),
)

gpt_4o = Model(
    name="gpt-4o",
    base_provider="OpenAI",
    best_provider=EncryptedProxy,
)

gpt_4o_mini = Model(
    name="gpt-4o-mini",
    base_provider="OpenAI",
    best_provider=IterListProvider([EncryptedProxy, Chatai], shuffle=False),
)


class ModelUtils:
    """Utility class for mapping string identifiers to Model instances."""

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


ModelUtils.convert = ModelRegistry.all_models()

demo_models = {}


def _get_working_providers(model: Model) -> List:
    if model.best_provider is None:
        return []
    if isinstance(model.best_provider, IterListProvider):
        return [p for p in model.best_provider.providers if p.working]
    return [model.best_provider] if model.best_provider.working else []


__models__ = {
    name: (model, _get_working_providers(model))
    for name, model in ModelRegistry.all_models().items()
    if name and _get_working_providers(model)
}

_all_models = list(__models__.keys())
Model.__all__ = staticmethod(lambda: _all_models)
