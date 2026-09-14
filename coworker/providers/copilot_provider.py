"""`github-copilot` provider — models through a GitHub Copilot subscription.

The backend speaks OpenAI-compatible Chat Completions at `api.githubcopilot.com`,
so all conversion/parsing is inherited from `OpenAIProvider` — this subclass only
swaps the credential: a short-lived Copilot bearer from `copilot_auth` instead of
an API key, plus the editor headers the backend requires.

Differences from the API-key path:
  - 401 → one refresh-and-retry (the Copilot token died mid-flight); a rejected
    GitHub token surfaces as a typed sign-in-required error, never a crash loop.
  - The Copilot token is refreshed from the long-lived GitHub OAuth token, so
    the user only signs in once (via device flow) and tokens refresh silently.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import AssistantTurn
from .copilot_auth import (
    COPILOT_API_BASE,
    CopilotAuthError,
    CopilotSignInRequired,
    CopilotTokenStore,
    api_headers,
)
from .openai_provider import OpenAIProvider
from .openai_responses import OpenAIResponsesProvider


def _status_code(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


class CopilotProvider(OpenAIProvider):
    def __init__(
        self,
        client: Any = None,
        *,
        secrets: Any = None,
        default_model: str = "gpt-5.5",
    ):
        super().__init__(
            client=client,
            default_model=default_model,
            base_url=COPILOT_API_BASE,
        )
        self._store = CopilotTokenStore(secrets)
        self._client_token: Optional[str] = None
        self._responses_provider: Optional[OpenAIResponsesProvider] = None
        self._responses_token: Optional[str] = None
        self._injected = client is not None

    def _ensure_client(self) -> Any:
        if self._injected:
            return self._client
        # The Copilot token is short-lived (~2h): fetch per call (refreshes itself
        # near expiry) and rebuild the SDK client whenever the token rotated.
        token = self._store.access_token()
        if self._client is None or token != self._client_token:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=token,
                base_url=COPILOT_API_BASE,
                default_headers=api_headers(token),
            )
            self._client_token = token
        return self._client

    def _responses(self, model: str) -> Optional[OpenAIResponsesProvider]:
        model_id = model.split(":", 1)[-1]
        metadata = self._store.models().get(model_id) or {}
        if metadata.get("endpoint") != "responses":
            return None
        token = self._store.access_token()
        if self._responses_provider is None or token != self._responses_token:
            from openai import OpenAI

            client = OpenAI(
                api_key=token,
                base_url=COPILOT_API_BASE,
                default_headers=api_headers(token),
            )
            self._responses_provider = OpenAIResponsesProvider(client=client)
            self._responses_token = token
        return self._responses_provider

    def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        try:
            return super()._create(client, kwargs)
        except Exception as exc:
            status = _status_code(exc)
            if status == 401 and not self._injected:
                # The Copilot API uses the GitHub OAuth token directly. A 401
                # means that token was revoked or is otherwise unusable; retrying
                # the same token only creates a loop, so require an explicit sign-in.
                self._store.clear()
                raise CopilotSignInRequired(
                    "GitHub Copilot session expired — sign in again in Settings ▸ Models."
                ) from exc
            raise

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **settings: Any,
    ) -> AssistantTurn:
        responses = self._responses(model)
        if responses is not None:
            return responses.complete(
                model=model, messages=messages, tools=tools, **settings
            )
        # Ensure client is built with current token before delegating.
        self._ensure_client()
        try:
            return super().complete(
                model=model, messages=messages, tools=tools, **settings
            )
        except Exception as exc:
            if _status_code(exc) == 401 and not self._injected:
                self._store.clear()
                raise CopilotSignInRequired(
                    "GitHub Copilot session expired — sign in again in Settings ▸ Models."
                ) from exc
            raise

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **settings: Any,
    ):
        responses = self._responses(model)
        if responses is not None:
            return responses.stream(
                model=model, messages=messages, tools=tools, **settings
            )
        self._ensure_client()
        return super().stream(
            model=model, messages=messages, tools=tools, **settings
        )
