from __future__ import annotations

import requests

from ..helper import filter_none
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin
from ...typing import Optional, Union, AsyncResult, Messages
from ...requests import StreamSession, StreamResponse, raise_for_status, sse_stream
from ...providers.response import *
from ...errors import MissingAuthError
from ... import debug

class OpenaiTemplate(AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin):
    """
    OpenAI-compatible template focused on proxy/wrapper scenarios.

    Supports `/models` discovery and `/chat/completions` with streaming.
    """

    base_url = ""
    backup_url = None
    api_key = None
    api_endpoint = None
    supports_message_history = True
    supports_system_message = True
    default_model = ""
    fallback_models = []
    sort_models = True
    needs_auth = False
    ssl = None
    max_tokens: Optional[int] = None
    live = 0

    @classmethod
    def is_provider_api_key(cls, api_key: str) -> bool:
        if cls.backup_url is None:
            return True
        return api_key and not api_key.startswith("g4f_") and not api_key.startswith("gfs_")

    @classmethod
    def get_models(cls, api_key: str = None, base_url: str = None, timeout: int = None) -> dict:
        if not cls.models:
            try:
                if api_key is None and cls.api_key is not None:
                    api_key = cls.api_key
                if base_url is None:
                    base_url = cls.base_url
                    if not cls.is_provider_api_key(api_key):
                        base_url = cls.backup_url
                response = requests.get(f"{base_url}/models", headers=cls.get_headers(False, api_key), verify=cls.ssl, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                data = data.get("data", data.get("models")) if isinstance(data, dict) else data
                if data is None:
                    data = []
                if data:
                    cls.live += 1
                cls.models = {
                    model.get("id", model.get("name")): {"id": model.get("id", model.get("name")), **model}
                    for model in data
                }
            except Exception as e:
                if cls.fallback_models:
                    debug.error(e)
                    return {model: {"id": model} for model in cls.fallback_models}
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
        impersonate: str = None,
        extra_body: dict = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None and cls.api_key is not None:
            api_key = cls.api_key
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')
        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            model = cls.get_model(model, api_key=api_key, base_url=base_url)
            if base_url is None:
                base_url = cls.base_url if cls.is_provider_api_key(api_key) else cls.backup_url

            if extra_body is None:
                extra_body = {}
            body = {**kwargs, **extra_body}
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
            if api_endpoint is None:
                if api_endpoint is None:
                    api_endpoint = cls.api_endpoint
                if api_endpoint is None:
                    api_endpoint = f"{base_url.rstrip('/')}/chat/completions"
            yield JsonRequest.from_dict(data)
            async with session.post(api_endpoint, json=data, ssl=cls.ssl) as response:
                async for chunk in read_response(response, data.get("stream"), cls.get_dict()):
                    yield chunk

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key else {}
            ),
            **({} if headers is None else headers)
        }


async def read_response(response: StreamResponse, stream: bool, provider_info: dict) -> AsyncResult:
    yield HeadersResponse.from_dict({key: value for key, value in response.headers.items() if key.lower().startswith("x-")})
    content_type = response.headers.get("content-type", "text/event-stream" if stream else "application/json")
    if content_type.startswith("application/json"):
        data = await response.json()
        if isinstance(data, list):
            data = next(iter(data), {})
        if isinstance(data, dict):
            yield JsonResponse.from_dict(data)
        OpenaiTemplate.raise_error(data, response.status)
        await raise_for_status(response)
        model = data.get("model")
        if model:
            yield ProviderInfo(**provider_info, model=model)
        if "usage" in data:
            yield Usage.from_dict(data["usage"])
        if "conversation" in data:
            yield JsonConversation.from_dict(data["conversation"])
        if "choices" in data:
            choice = next(iter(data["choices"]), None)
            message = choice.get("message", {})
            if choice and "content" in message and message["content"]:
                yield message["content"].strip()
            if "tool_calls" in message:
                yield ToolCalls(message["tool_calls"])
            if choice:
                reasoning_content = choice.get("delta", {}).get("reasoning_content", choice.get("delta", {}).get("reasoning"))
                if reasoning_content:
                    yield Reasoning(reasoning_content, status="")
            if choice and "finish_reason" in choice and choice["finish_reason"] is not None:
                yield FinishReason(choice["finish_reason"])
                return
    elif content_type.startswith("text/event-stream"):
        await raise_for_status(response)
        reasoning = False
        first = True
        model_returned = False
        async for data in sse_stream(response):
            yield JsonResponse.from_dict(data)
            OpenaiTemplate.raise_error(data)
            model = data.get("model")
            if not model_returned and model:
                yield ProviderInfo(**provider_info, model=model)
                model_returned = True
            choice = next(iter(data.get("choices", [])), None)
            if choice:
                content = choice.get("delta", {}).get("content")
                if content:
                    if first:
                        content = content.lstrip()
                    if content:
                        first = False
                        if reasoning:
                            yield Reasoning(status="")
                            reasoning = False
                        yield content
                tool_calls = choice.get("delta", {}).get("tool_calls")
                if tool_calls:
                    yield ToolCalls(tool_calls)
                reasoning_content = choice.get("delta", {}).get("reasoning_content", choice.get("delta", {}).get("reasoning"))
                if reasoning_content:
                    reasoning = True
                    yield Reasoning(reasoning_content)
            if "usage" in data and data["usage"] and "total_tokens" in data["usage"]:
                yield Usage.from_dict(data["usage"])
            if "conversation" in data and data["conversation"]:
                yield JsonConversation.from_dict(data["conversation"])
            if choice and choice.get("finish_reason") is not None:
                yield FinishReason(choice["finish_reason"])
