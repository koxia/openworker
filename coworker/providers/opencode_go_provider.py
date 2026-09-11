"""OpenCode Go provider — OpenAI-compatible endpoint with session header requirement.

OpenCode Go is a $10/month subscription service that provides access to various
open coding models. The key requirement is sending a stable `x-opencode-session`
header with each request for routing optimization.

This provider extends OpenAIProvider and adds the session header to all requests.
The session ID is generated per-provider instance (per conversation) and remains
stable throughout the conversation.

Models available through OpenCode Go:
- GLM-5.3-Flash, GLM-5.3, GLM-5.2, GLM-5.1
- Kimi K3, Kimi K2.7 Code, Kimi K2.6
- DeepSeek V4.1 Flash, DeepSeek V4 Pro, DeepSeek V4 Flash
- MiMo-V2.5, MiMo-V2.5-Pro
- Qwen3.8 Max, Qwen3.8 Flash, Qwen3.7 Max, Qwen3.7 Plus, Qwen3.6 Plus
- MiniMax M3, M2.7, M2.5 (Anthropic Messages API)
- Muse Spark 1.3/1.2 Contributor
- Grok 4.6, GPT 5.6 Luna (Responses API)
- LongCat-2.0, Hy4 preview, Hy3
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from .openai_provider import OpenAIProvider


class OpenCodeGoProvider(OpenAIProvider):
    """OpenCode Go provider with session header support.
    
    OpenCode Go requires a stable `x-opencode-session` header for routing optimization.
    This provider generates a session ID per instance (per conversation) and includes
    it in all API requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        secrets: Any = None,
        default_model: str = "glm-5.2",
    ):
        """Initialize OpenCode Go provider.
        
        Args:
            api_key: OpenCode Go API key (from subscription, starts with sk-)
            base_url: Base URL for the API (defaults to https://opencode.ai/zen/go/v1)
            secrets: SecretStore for credential management
            default_model: Default model to use
        """
        # Default base URL for OpenCode Go
        if base_url is None:
            base_url = "https://opencode.ai/zen/go/v1"
        
        # Generate a stable session ID for this conversation
        self._session_id = str(uuid.uuid4())
        
        # Call parent init
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            secrets=secrets,
            default_model=default_model,
        )

    def _ensure_client(self) -> Any:
        """Override to add x-opencode-session header to the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            from .openai_provider import resolve_api_key

            key = self._api_key or resolve_api_key(self._secrets)
            if not key:
                raise RuntimeError(
                    "No OpenCode Go API key configured. Add your key in Settings ▸ Models."
                )
            kwargs: dict[str, Any] = {
                "api_key": key,
                "default_headers": {
                    "x-opencode-session": self._session_id,
                },
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client
