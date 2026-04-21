from __future__ import annotations

from ..typing import AsyncResult, Messages, MediaListType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..Provider.needs_auth import OpenaiProxy
from .. import Provider


class AnyModelProviderMixin(ProviderModelMixin):
    default_model = "gpt-4o-mini"
    audio_models = []
    image_models = []
    vision_models = []
    video_models = []
    models_count = {}
    models = []
    model_map: dict[str, dict[str, str]] = {}
    model_aliases: dict[str, str] = {}

    @classmethod
    def get_models(cls, ignored: list[str] = [], **kwargs) -> list[str]:
        if "OpenaiProxy" in ignored:
            return []
        try:
            cls.models = list(OpenaiProxy.get_models(**kwargs).keys())
        except Exception:
            cls.models = []
        if not cls.models:
            cls.models = [OpenaiProxy.default_model] if OpenaiProxy.default_model else [cls.default_model]
        cls.model_map = {model: {"OpenaiProxy": model} for model in cls.models}
        cls.models_count = {model: 1 for model in cls.models}
        return cls.models


class AnyProvider(AsyncGeneratorProvider, AnyModelProviderMixin):
    working = True
    supports_stream = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        media: MediaListType = None,
        api_key: str = None,
        **kwargs
    ) -> AsyncResult:
        async for chunk in OpenaiProxy.create_async_generator(
            model=model,
            messages=messages,
            stream=stream,
            media=media,
            api_key=api_key,
            **kwargs
        ):
            yield chunk


setattr(Provider, "AnyProvider", AnyProvider)
Provider.__map__["AnyProvider"] = AnyProvider
Provider.__providers__.append(AnyProvider)
