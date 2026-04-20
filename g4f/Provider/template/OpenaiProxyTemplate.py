from __future__ import annotations

import requests

from ..helper import filter_none
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin
from ...typing import Union, AsyncResult, Messages
from ...requests import StreamSession
from ...providers.response import JsonRequest
from ...errors import MissingAuthError
from .OpenaiTemplate import read_response


class OpenaiProxyTemplate(AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin):
    """
    Lightweight OpenAI-compatible template for proxy/wrapper APIs.
    Supports `/models` and `/chat/completions` with streaming responses.
    """

    base_url = ""
    api_key = None
    api_endpoint = None
    default_model = ""
    fallback_models = []
    sort_models = True
    needs_auth = False
    ssl = None
    max_tokens: int = None

    @classmethod
    def get_models(cls, api_key: str = None, base_url: str = None, timeout: int = None) -> list[str]:
        if not cls.models:
            try:
                api_key = api_key if api_key is not None else cls.api_key
                base_url = base_url or cls.base_url
                response = requests.get(
                    f"{base_url.rstrip('/')}/models",
                    headers=cls.get_headers(False, api_key),
                    verify=cls.ssl,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                data = data.get("data", data.get("models")) if isinstance(data, dict) else data
                if data:
                    cls.live += 1
                cls.models = {
                    model.get("id", model.get("name")): {"id": model.get("id", model.get("name")), **model}
                    for model in data
                }
                if cls.sort_models and isinstance(cls.models, list):
                    cls.models.sort()
            except Exception:
                if cls.fallback_models:
                    return cls.fallback_models
                raise
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        api_key: str = None,
        api_endpoint: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: Union[str, list[str]] = None,
        stream: bool = None,
        user: str = None,
        headers: dict = None,
        extra_body: dict = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None and cls.api_key is not None:
            api_key = cls.api_key
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')

        model = cls.get_model(model, api_key=api_key, base_url=base_url)
        base_url = base_url or cls.base_url
        api_endpoint = api_endpoint or cls.api_endpoint or f"{base_url.rstrip('/')}/chat/completions"

        body = {}
        if kwargs:
            body.update(kwargs)
        if extra_body:
            body.update(extra_body)

        data = filter_none(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else cls.max_tokens,
            top_p=top_p,
            stop=stop,
            stream=True if stream is None else stream,
            user=user,
            **body
        )
        yield JsonRequest.from_dict(data)

        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(data.get("stream"), api_key, headers),
            timeout=timeout,
        ) as session:
            async with session.post(api_endpoint, json=data, ssl=cls.ssl) as response:
                async for chunk in read_response(response, data.get("stream"), "", cls.get_dict(), False):
                    yield chunk

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            **({} if headers is None else headers),
        }
