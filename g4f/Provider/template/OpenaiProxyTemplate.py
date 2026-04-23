from __future__ import annotations

from .OpenaiTemplate import OpenaiTemplate


class OpenaiProxyTemplate(OpenaiTemplate):
    """
    Backward-compatible alias for the OpenAI proxy template.

    Kept as a dedicated class name for existing imports and provider
    declarations that already inherit from OpenaiProxyTemplate.
    """
