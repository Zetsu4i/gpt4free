from __future__ import annotations

from ..template import OpenaiProxyTemplate


class OpenaiProxy(OpenaiProxyTemplate):
    label = "OpenAI Proxy"
    working = True
    needs_auth = False
    base_url = "http://localhost:8080/v1"
    sort_models = False
