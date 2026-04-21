from __future__ import annotations

import sys

from ..providers.types import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider, RotatedProvider
from ..providers.base_provider import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images import CreateImagesProvider

from .template import OpenaiTemplate, OpenaiProxyTemplate, BackendApi
from .needs_auth import OpenaiAPI, Custom, Feature, OpenaiProxy

__modules__: list = [
    getattr(sys.modules[__name__], provider) for provider in dir()
    if not provider.startswith("__")
]
__providers__: list[ProviderType] = [
    provider for provider in __modules__
    if isinstance(provider, type)
    and issubclass(provider, BaseProvider)
]
__all__: list[str] = [
    provider.__name__ for provider in __providers__
]
__map__: dict[str, ProviderType] = {
    provider.__name__: provider for provider in __providers__
}


class ProviderUtils:
    convert: dict[str, ProviderType] = __map__

    @classmethod
    def get_by_label(cls, label: str) -> ProviderType:
        if not label:
            raise ValueError("Label must be provided")
        provider = cls.convert.get(label)
        if provider is None:
            for provider_cls in cls.convert.values():
                if provider_cls.working and provider_cls.__name__.lower().startswith(label.lower()):
                    provider = provider_cls
                    break
        if provider is None:
            raise ValueError(f"Provider with label '{label}' not found")
        return provider
