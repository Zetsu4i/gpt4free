from __future__ import annotations

from typing import Union

from .. import Provider, debug, models
from ..errors import ModelNotFoundError
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.config_provider import ConfigModelProvider, RouterConfig
from ..providers.retry_provider import RotatedProvider
from ..typing import AsyncResult, MediaListType, Messages


class AnyModelProviderMixin(ProviderModelMixin):
    default_model = "default"
    audio_models: list[str] = []
    image_models: list[str] = []
    vision_models: list[str] = []
    video_models: list[str] = []
    models_count: dict[str, int] = {}
    model_map: dict[str, dict[str, str]] = {}
    model_aliases: dict[str, str] = {}
    models: list[str] = []

    @classmethod
    def create_model_map(cls) -> None:
        cls.model_map = {
            "default": {
                provider.__name__: ""
                for provider in models.default.best_provider.providers
            }
        }
        cls.model_map.update(
            {
                name: {
                    provider.__name__: model.get_long_name()
                    for provider in providers
                    if provider.working
                }
                for name, (model, providers) in models.__models__.items()
            }
        )
        cls.models = list(cls.model_map.keys())

    @classmethod
    def get_models(cls, ignored: list[str] = [], **kwargs) -> list[str]:
        if not cls.model_map:
            cls.create_model_map()
        if not ignored:
            return cls.models
        ignored_set = set(ignored)
        return [
            model
            for model, providers in cls.model_map.items()
            if any(provider not in ignored_set for provider in providers.keys())
        ]

    @classmethod
    def get_grouped_models(cls, ignored: list[str] = []) -> list[dict[str, list[str]]]:
        models_list = cls.get_models(ignored=ignored)
        return [
            {"group": "Default", "models": ["default"]},
            {"group": "Models", "models": [model for model in models_list if model != "default"]},
        ]


class AnyProvider(AsyncGeneratorProvider, AnyModelProviderMixin):
    working = True
    active_by_default = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        media: MediaListType = None,
        ignored: list[str] = [],
        api_key: Union[str, dict[str, str]] = None,
        **kwargs,
    ) -> AsyncResult:
        providers = []
        if not model or model == cls.default_model:
            model = ""
            providers = (
                models.default_vision.best_provider.providers
                if media
                else models.default.best_provider.providers
            )
        elif model in RouterConfig.routes:
            async for chunk in ConfigModelProvider(RouterConfig.routes.get(model)).create_async_generator(
                model, messages, stream=stream, media=media, api_key=api_key, **kwargs
            ):
                yield chunk
            return
        elif model in Provider.__map__:
            provider = Provider.__map__[model]
            if provider.working and provider.get_parent() not in ignored:
                model = getattr(provider, "default_model", model)
                providers.append(provider)
        elif model and ":" in model:
            provider_name, submodel = model.split(":", maxsplit=1)
            if provider_name in Provider.__map__:
                provider = Provider.__map__[provider_name]
                if provider.working and provider.get_parent() not in ignored:
                    providers.append(provider)
                    model = submodel
        else:
            if not cls.model_map:
                cls.create_model_map()
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
            if model in cls.model_map:
                for provider_name, alias in cls.model_map[model].items():
                    provider = Provider.__map__.get(provider_name)
                    if provider and provider.working:
                        if model not in provider.model_aliases:
                            provider.model_aliases[model] = alias
                        providers.append(provider)

        if not providers:
            proxy_provider = Provider.__map__.get("EncryptedProxy")
            if proxy_provider:
                providers = [proxy_provider]

        providers = [
            provider
            for provider in providers
            if provider.working and provider.get_parent() not in ignored
        ]
        providers = list({provider.__name__: provider for provider in providers}.values())

        if not providers:
            raise ModelNotFoundError(
                f"AnyProvider: Model {model} not found in any provider."
            )

        debug.log(
            f"AnyProvider: Using providers: {[provider.__name__ for provider in providers]} for model '{model}'"
        )

        async for chunk in RotatedProvider(providers).create_async_generator(
            model, messages, stream=stream, media=media, api_key=api_key, **kwargs
        ):
            yield chunk


setattr(Provider, "AnyProvider", AnyProvider)
Provider.__map__["AnyProvider"] = AnyProvider
Provider.__providers__.append(AnyProvider)
