from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from functools import lru_cache
from typing import Any, AsyncIterator

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from .. import debug
from ..errors import MissingAuthError
from ..image.copy_images import save_response_media
from ..providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin
from ..providers.helper import filter_none
from ..providers.response import (
    AudioResponse,
    FinishReason,
    HeadersResponse,
    JsonConversation,
    JsonRequest,
    JsonResponse,
    ProviderInfo,
    Reasoning,
    ToolCalls,
    Usage,
)
from ..requests import StreamSession, raise_for_status
from ..tools.media import render_messages
from ..typing import AsyncResult, MediaListType, Messages


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data)


class EncryptedProxy(AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin):
    label = "EncryptedProxy"
    url = "https://encrypted-proxy.local"
    api_endpoint = None
    working = True
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    supports_system_message = True

    default_model = "gpt-4o-mini"
    model_aliases = {"gpt-4o-mini": default_model}
    models = list(model_aliases.keys())

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        if not model:
            return cls.default_model
        return cls.model_aliases.get(model, model)

    @classmethod
    def _get_api_endpoint(cls, api_endpoint: str | None, base_url: str | None) -> str:
        if api_endpoint:
            return api_endpoint
        env_endpoint = os.environ.get("G4F_ENCRYPTED_PROXY_URL")
        if env_endpoint:
            return env_endpoint
        if cls.api_endpoint:
            return cls.api_endpoint
        if base_url:
            return f"{base_url.rstrip('/')}/chat/completions"
        raise MissingAuthError("Set G4F_ENCRYPTED_PROXY_URL or provide api_endpoint/base_url.")

    @classmethod
    def _get_shared_secret(cls, api_key: str | None, encryption_key: str | None) -> str:
        secret = encryption_key or api_key or os.environ.get("G4F_ENCRYPTED_PROXY_KEY")
        if not secret:
            raise MissingAuthError("Set G4F_ENCRYPTED_PROXY_KEY or pass api_key/encryption_key.")
        return secret

    @classmethod
    @lru_cache(maxsize=8)
    def _derive_keys(cls, secret: str) -> tuple[bytes, bytes]:
        secret_bytes = secret.encode("utf-8")
        key_material = hashlib.pbkdf2_hmac(
            "sha256",
            secret_bytes,
            b"g4f-encrypted-proxy",
            200_000,
            dklen=64,
        )
        return key_material[:32], key_material[32:]

    @classmethod
    def _sign(cls, sig_key: bytes, data: bytes) -> str:
        return _b64encode(hmac.new(sig_key, data, hashlib.sha256).digest())

    @classmethod
    def _verify_signature(cls, sig_key: bytes, data: bytes, signature: str | None) -> bool:
        if not signature:
            return False
        try:
            expected = hmac.new(sig_key, data, hashlib.sha256).digest()
            return hmac.compare_digest(expected, _b64decode(signature))
        except Exception:
            return False

    @classmethod
    def _encrypt_payload(cls, payload: dict, secret: str) -> dict:
        enc_key, sig_key = cls._derive_keys(secret)
        nonce = get_random_bytes(12)
        cipher = AES.new(enc_key, AES.MODE_GCM, nonce=nonce)
        plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        signed = nonce + tag + ciphertext
        return {
            "alg": "AES-256-GCM-HMAC-SHA256",
            "nonce": _b64encode(nonce),
            "tag": _b64encode(tag),
            "ciphertext": _b64encode(ciphertext),
            "signature": cls._sign(sig_key, signed),
        }

    @classmethod
    def _decrypt_payload(cls, envelope: dict, secret: str) -> dict:
        enc_key, sig_key = cls._derive_keys(secret)
        nonce = _b64decode(envelope["nonce"])
        tag = _b64decode(envelope["tag"])
        ciphertext = _b64decode(envelope["ciphertext"])
        signed = nonce + tag + ciphertext
        signature = envelope.get("signature")
        if signature and not cls._verify_signature(sig_key, signed, signature):
            raise ValueError("EncryptedProxy signature verification failed.")
        cipher = AES.new(enc_key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return json.loads(plaintext.decode("utf-8"))

    @classmethod
    def _maybe_decrypt(cls, payload: Any, secret: str) -> Any:
        if isinstance(payload, dict) and "payload" in payload:
            payload = payload["payload"]
        if isinstance(payload, dict) and {"ciphertext", "nonce", "tag"}.issubset(payload):
            return cls._decrypt_payload(payload, secret)
        return payload

    @classmethod
    async def _iter_decrypted_sse(
        cls,
        response,
        secret: str,
    ) -> AsyncIterator[dict]:
        async for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            if raw.startswith(b"[DONE]"):
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                debug.error("EncryptedProxy: invalid SSE JSON payload")
                continue
            payload = cls._maybe_decrypt(event, secret)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    debug.error("EncryptedProxy: invalid decrypted SSE JSON")
                    continue
            if isinstance(payload, dict):
                yield payload

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        conversation: JsonConversation = None,
        media: MediaListType = None,
        api_key: str = None,
        api_endpoint: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        seed: int = None,
        n: int = None,
        stop: str | list[str] = None,
        stream: bool = None,
        prompt: str = None,
        user: str = None,
        headers: dict = None,
        impersonate: str = None,
        download_media: bool = True,
        encryption_key: str = None,
        extra_parameters: list[str] | None = None,
        extra_body: dict = None,
        **kwargs,
    ) -> AsyncResult:
        secret = cls._get_shared_secret(api_key, encryption_key)
        api_endpoint = cls._get_api_endpoint(api_endpoint, base_url)
        model = cls.get_model(model)

        if stream or stream is None:
            kwargs.setdefault("stream_options", {"include_usage": True})
        if extra_parameters is None:
            extra_parameters = [
                "tools",
                "parallel_tool_calls",
                "tool_choice",
                "reasoning_effort",
                "logit_bias",
                "modalities",
                "audio",
                "stream_options",
                "include_reasoning",
                "response_format",
                "max_completion_tokens",
                "search_settings",
            ]
        extra_parameters = {key: kwargs[key] for key in extra_parameters if key in kwargs}
        if extra_body is None:
            extra_body = {}
        data = filter_none(
            messages=list(render_messages(messages, media)),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            n=n,
            stop=stop,
            stream="audio" not in extra_parameters if stream is None else stream,
            user=user,
            conversation=conversation.get_dict() if conversation else None,
            **extra_parameters,
            **extra_body,
        )
        stream = data.get("stream", False)
        yield JsonRequest.from_dict(data)

        envelope = cls._encrypt_payload(data, secret)
        request_body = {"payload": envelope}

        request_headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(headers)

        async with StreamSession(
            proxy=proxy,
            headers=request_headers,
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            async with session.post(api_endpoint, json=request_body) as response:
                yield HeadersResponse.from_dict(
                    {key: value for key, value in response.headers.items() if key.lower().startswith("x-")}
                )
                if stream:
                    if response.status >= 400:
                        await raise_for_status(response)
                    reasoning = False
                    first = True
                    model_returned = False
                    async for chunk in cls._iter_decrypted_sse(response, secret):
                        yield JsonResponse.from_dict(chunk)
                        cls.raise_error(chunk)
                        model_name = chunk.get("model")
                        if not model_returned and model_name:
                            yield ProviderInfo(**cls.get_dict(), model=model_name)
                            model_returned = True
                        choice = next(iter(chunk.get("choices", [])), None)
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
                            reasoning_content = choice.get("delta", {}).get(
                                "reasoning_content", choice.get("delta", {}).get("reasoning")
                            )
                            if reasoning_content:
                                reasoning = True
                                yield Reasoning(reasoning_content)
                        if "usage" in chunk and chunk["usage"] and "total_tokens" in chunk["usage"]:
                            yield Usage.from_dict(chunk["usage"])
                        if "conversation" in chunk and chunk["conversation"]:
                            yield JsonConversation.from_dict(chunk["conversation"])
                        if choice and choice.get("finish_reason") is not None:
                            yield FinishReason(choice["finish_reason"])
                else:
                    response_data = await response.json()
                    response_data = cls._maybe_decrypt(response_data, secret)
                    if isinstance(response_data, str):
                        response_data = json.loads(response_data)
                    yield JsonResponse.from_dict(response_data)
                    cls.raise_error(response_data, response.status)
                    await raise_for_status(response)
                    model_name = response_data.get("model")
                    if model_name:
                        yield ProviderInfo(**cls.get_dict(), model=model_name)
                    if "usage" in response_data:
                        yield Usage.from_dict(response_data["usage"])
                    if "conversation" in response_data:
                        yield JsonConversation.from_dict(response_data["conversation"])
                    if "choices" in response_data:
                        choice = next(iter(response_data["choices"]), None)
                        message = choice.get("message", {}) if choice else {}
                        if choice and "content" in message and message["content"]:
                            yield message["content"].strip()
                        if "tool_calls" in message:
                            yield ToolCalls(message["tool_calls"])
                        if choice:
                            reasoning_content = choice.get("delta", {}).get(
                                "reasoning_content", choice.get("delta", {}).get("reasoning")
                            )
                            if reasoning_content:
                                yield Reasoning(reasoning_content, status="")
                        audio = message.get("audio", {})
                        if "data" in audio:
                            if download_media:
                                async for chunk in save_response_media(audio, prompt, [model_name or model]):
                                    yield chunk
                            else:
                                yield AudioResponse(
                                    f"data:audio/mpeg;base64,{audio['data']}",
                                    transcript=audio.get("transcript"),
                                )
                        if choice and "finish_reason" in choice and choice["finish_reason"] is not None:
                            yield FinishReason(choice["finish_reason"])
