from __future__ import annotations

from ..template import OpenaiProxyTemplate

class Custom(OpenaiProxyTemplate):
    label = "Custom Provider"
    working = True
    needs_auth = False
    models_needs_auth = False
    base_url = "http://localhost:8080/v1"
    sort_models = False

class Feature(Custom):
    label = "Feature Provider"
    working = False


class OpenaiProxy(OpenaiProxyTemplate):
    label = "OpenAI Proxy"
    working = True
    needs_auth = False
    base_url = "http://localhost:8080/v1"
    sort_models = False
