"""GitHub Copilot sign-in for the `github-copilot` provider (OAuth 2.0 Device Flow).

Instead of an API key, the user signs in with their GitHub account that has a Copilot
subscription. The flow uses GitHub's OAuth Device Flow — the user visits a URL and
enters a code, no loopback server needed — then uses the GitHub OAuth token directly
with the Copilot API.

The pieces:

  - `sign_in()`         — async, explicit-action only: start device flow, poll for
    the GitHub token, and persist it.
  - `CopilotTokenStore` — persistence for the long-lived GitHub OAuth token used
    directly by the Copilot API.
  - `verify()`          — the Test-button probe: one cheap authenticated request.

Tokens land in the SecretStore profile `provider:github-copilot` — the same local-only
storage every provider profile uses, never a plaintext config file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# -- GitHub OAuth Device Flow endpoints ----------------------------------------

DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
# This public OAuth client id is also used by OpenCode. It supports the device
# flow; the resulting GitHub OAuth token is accepted directly by the Copilot API.
CLIENT_ID = "Ov23li8tweQw6odWQebz"
SCOPE = "read:user"

COPILOT_API_BASE = "https://api.githubcopilot.com"
PROFILE = "provider:github-copilot"
API_VERSION = "2026-06-01"
FLOW_TIMEOUT_SECONDS = 300
# Device flow default poll interval (GitHub returns one; we fall back to this).
DEFAULT_POLL_INTERVAL = 5

# Headers expected by GitHub Copilot's third-party-agent API.
EDITOR_VERSION = "vscode/1.91.0"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.19.0"
COPILOT_INTEGRATION_ID = "vscode-chat"

# Smallest model for the verify probe — minimize cost.
_VERIFY_MODEL = "gpt-5-mini"

SIGNED_OUT_ERROR = (
    "Not signed in to GitHub Copilot — connect your account in Settings ▸ Models "
    "to use the Copilot provider."
)
EXPIRED_ERROR = "GitHub Copilot session expired — sign in again in Settings ▸ Models."
NO_COPILOT_ERROR = (
    "GitHub account does not have a Copilot subscription — "
    "enable Copilot in your GitHub settings first."
)


class CopilotAuthError(RuntimeError):
    """A Copilot-auth failure with a user-readable message."""


class CopilotSignInRequired(CopilotAuthError):
    """No usable tokens — the fix is an explicit sign-in, never a silent browser."""


# -- HTTP helpers (module-level so tests can stub the wire) ---------------------


def _post_json(url: str, **kwargs: Any) -> Any:
    """One POST returning parsed JSON. Blocking (httpx sync); callers run via
    `asyncio.to_thread`."""
    import httpx

    resp = httpx.post(url, **kwargs)
    if resp.status_code >= 300:
        return None, resp.status_code
    try:
        return resp.json(), resp.status_code
    except Exception:
        return None, resp.status_code


def _post_form(url: str, **kwargs: Any) -> Any:
    """One form POST. Returns (parsed_or_None, status_code)."""
    import httpx

    # Merge caller-supplied headers with the default Accept header so we don't
    # pass `headers` twice (which would raise "got multiple values for keyword argument").
    caller_headers = kwargs.pop("headers", None) or {}
    merged_headers = {"Accept": "application/json", **caller_headers}
    resp = httpx.post(url, headers=merged_headers, **kwargs)
    if resp.status_code >= 500:
        return None, resp.status_code
    try:
        return resp.json(), resp.status_code
    except Exception:
        return None, resp.status_code


# -- Copilot token acquisition --------------------------------------------------


def _copilot_headers(github_token: str) -> dict[str, str]:
    """Headers for the Copilot token endpoint and API calls."""
    return {
        "Authorization": f"Bearer {github_token}",
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
    }


def api_headers(copilot_token: str) -> dict[str, str]:
    """Headers for Copilot API calls (catalog and chat completions)."""
    return {
        "Authorization": f"Bearer {copilot_token}",
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "X-GitHub-Api-Version": API_VERSION,
        "Openai-Intent": "conversation-edits",
        "User-Agent": "OpenWorker",
        "Content-Type": "application/json",
    }


def fetch_copilot_models(github_token: str, timeout: float = 10.0) -> dict[str, dict[str, Any]]:
    """Fetch models that are usable through Copilot's chat-completions API.

    The static GitHub model list includes models that are only available through
    `/responses` or `/v1/messages`; sending those through OpenAI's chat client
    produces `unsupported_api_for_model`. Keep only enabled models that advertise
    `/chat/completions` and tool calling.
    """
    import httpx

    try:
        resp = httpx.get(
            COPILOT_API_BASE + "/models",
            headers=api_headers(github_token),
            timeout=timeout,
        )
    except Exception as exc:
        raise CopilotAuthError(f"Couldn't fetch GitHub Copilot models ({exc.__class__.__name__}).") from exc
    if resp.status_code >= 300:
        raise CopilotAuthError(f"GitHub Copilot model catalog failed (HTTP {resp.status_code}).")

    try:
        raw = resp.json().get("data", [])
    except Exception as exc:
        raise CopilotAuthError("GitHub Copilot returned an invalid model catalog.") from exc

    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        supports = item.get("capabilities", {}).get("supports", {}) or {}
        limits = item.get("capabilities", {}).get("limits", {}) or {}
        endpoints = item.get("supported_endpoints", []) or []
        if item.get("policy", {}).get("state") == "disabled" or item.get("model_picker_enabled") is False:
            continue
        if not any(
            endpoint in endpoints
            for endpoint in ("/responses", "/v1/responses", "/chat/completions", "/v1/chat/completions")
        ) or supports.get("tool_calls") is not True:
            continue
        model_id = item.get("id")
        if not model_id or not limits.get("max_prompt_tokens") or not limits.get("max_output_tokens"):
            continue
        result[model_id] = {
            "label": item.get("name") or model_id,
            "context_window": limits.get("max_context_window_tokens") or limits.get("max_prompt_tokens"),
            "reasoning": bool(
                supports.get("adaptive_thinking")
                or supports.get("reasoning_effort")
                or supports.get("max_thinking_budget")
                or supports.get("min_thinking_budget")
            ),
            "reasoning_effort": supports.get("reasoning_effort") or [],
            "endpoint": (
                "responses"
                if any(endpoint in endpoints for endpoint in ("/responses", "/v1/responses"))
                else "chat"
            ),
        }
    if not result:
        raise CopilotAuthError("GitHub Copilot returned no chat-completions models available to this account.")
    return result


# -- Token persistence + refresh ------------------------------------------------


class CopilotTokenStore:
    """Token set in the `provider:github-copilot` SecretStore profile.

    Stores the long-lived GitHub OAuth token. The Copilot API accepts this token
    directly; unlike the old implementation, OpenWorker does not call the
    undocumented `copilot_internal/v2/token` endpoint.
    """

    def __init__(self, secrets: Any) -> None:
        self._secrets = secrets

    def _data(self) -> dict[str, Any]:
        if self._secrets is None:
            return {}
        return self._secrets.get(PROFILE) or {}

    def _merge(self, patch: dict[str, Any]) -> None:
        self._secrets.put(PROFILE, {**self._data(), **patch})

    def signed_in(self) -> bool:
        return bool(self._data().get("github_token"))

    def account_label(self) -> Optional[str]:
        data = self._data()
        return data.get("github_login") or data.get("github_id") or None

    def save_github_token(self, github_token: str, login: str = "", github_id: str = "") -> None:
        """Persist the GitHub OAuth token (long-lived) after device flow."""
        patch: dict[str, Any] = {
            "github_token": github_token,
            "github_token_saved_at": int(time.time()),
        }
        if login:
            patch["github_login"] = login
        if github_id:
            patch["github_id"] = github_id
        # Also update the `tokens` key so `descriptor_configured` sees the provider
        # as signed in (it checks `bool(profile.get("tokens"))` for OAuth providers).
        existing_tokens = self._data().get("tokens") or {}
        patch["tokens"] = {**existing_tokens, "github_token": github_token}
        self._merge(patch)

    def save_copilot_token(self, token_data: dict[str, Any]) -> None:
        """Accept legacy cached token data during migration; it is never used."""
        self._merge({"legacy_copilot_token": token_data.get("token", "")})

    def save_models(self, models: dict[str, dict[str, Any]]) -> None:
        """Persist the account-specific, chat-compatible model catalog."""
        self._merge({
            "supported_models": models,
            "supported_models_saved_at": int(time.time()),
            "supported_models_catalog_version": 2,
        })

    def models(self) -> dict[str, dict[str, Any]]:
        data = self._data().get("supported_models")
        return data if isinstance(data, dict) else {}

    def clear(self) -> bool:
        if self._secrets is None:
            return False
        return bool(self._secrets.delete(PROFILE))

    def access_token(self) -> str:
        """Return the GitHub OAuth token used directly by the Copilot API."""
        github_token = self._data().get("github_token") or ""
        if not github_token:
            raise CopilotSignInRequired(SIGNED_OUT_ERROR)
        return github_token

    def refresh(self) -> str:
        """Return the stored GitHub token; it is not exchanged for another token."""
        return self.access_token()


# -- GitHub user info -----------------------------------------------------------


def fetch_github_user(github_token: str, timeout: float = 10.0) -> dict[str, str]:
    """Fetch the authenticated user's login and id from GitHub."""
    import httpx

    try:
        resp = httpx.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=timeout,
        )
        if resp.status_code < 300:
            data = resp.json()
            return {
                "login": data.get("login", ""),
                "id": str(data.get("id", "")),
            }
    except Exception:
        pass
    return {"login": "", "id": ""}


# -- Interactive sign-in flow (Device Flow) -------------------------------------

# The last verification URL + user code, surfaced over REST so the GUI can show them.
last_verification_uri: Optional[str] = None
last_user_code: Optional[str] = None


async def sign_in(
    secrets: Any,
    *,
    timeout: float = FLOW_TIMEOUT_SECONDS,
    open_browser: bool = True,
    on_user_code: Optional[Any] = None,
) -> dict[str, Any]:
    """Run the full device flow: request code → poll for GitHub token → get Copilot token.

    Explicit-action only (a Settings button) — never called from an engine turn.
    `on_user_code` is an optional callable(uri, code) the GUI can use to display
    the code inline instead of relying on the browser alone.
    """
    global last_verification_uri, last_user_code
    import httpx

    # Step 1: Request device code
    try:
        resp = httpx.post(
            DEVICE_CODE_URL,
            data={"client_id": CLIENT_ID, "scope": SCOPE},
            headers={"Accept": "application/json"},
            timeout=30.0,
        )
    except Exception as exc:
        raise CopilotAuthError(
            f"Couldn't reach GitHub ({exc.__class__.__name__})."
        ) from exc

    if resp.status_code >= 300:
        raise CopilotAuthError(
            f"GitHub device code request failed (HTTP {resp.status_code})."
        )

    device_data = resp.json()
    device_code = device_data.get("device_code", "")
    verification_uri = device_data.get("verification_uri", "")
    user_code = device_data.get("user_code", "")
    interval = int(device_data.get("interval", DEFAULT_POLL_INTERVAL))
    expires_in = int(device_data.get("expires_in", 900))

    if not device_code or not verification_uri or not user_code:
        raise CopilotAuthError("GitHub returned an invalid device code response.")

    last_verification_uri = verification_uri
    last_user_code = user_code

    # Notify the GUI if it wants to display the code
    if on_user_code is not None:
        try:
            on_user_code(verification_uri, user_code)
        except Exception:
            pass

    # Step 2: Open browser
    if open_browser:
        import webbrowser

        logger.info("copilot auth: opening browser for device flow")
        await asyncio.get_running_loop().run_in_executor(
            None, webbrowser.open, verification_uri
        )

    # Step 3: Poll for the GitHub token
    deadline = time.time() + expires_in
    github_token = ""
    while time.time() < deadline:
        await asyncio.sleep(interval)
        try:
            poll_resp = httpx.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
                timeout=30.0,
            )
        except Exception as exc:
            raise CopilotAuthError(
                f"Couldn't reach GitHub during poll ({exc.__class__.__name__})."
            ) from exc

        poll_data = poll_resp.json() if poll_resp.status_code < 500 else {}
        error = poll_data.get("error", "")

        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval = int(poll_data.get("interval", interval + 5))
            continue
        if error == "expired_token":
            raise CopilotAuthError(
                "The device code expired — start the sign-in again."
            )
        if error == "access_denied":
            raise CopilotAuthError("Sign-in was denied — the user declined access.")
        if error:
            raise CopilotAuthError(f"GitHub sign-in failed: {error}")

        # Success — we got the GitHub token
        github_token = poll_data.get("access_token", "")
        if github_token:
            break
    else:
        raise CopilotAuthError(
            "Sign-in timed out — the code was not entered in time."
        )

    if not github_token:
        raise CopilotAuthError("GitHub sign-in failed — no token received.")

    # Step 4: Fetch user info
    user_info = await asyncio.to_thread(fetch_github_user, github_token)

    # Step 5: Persist the GitHub token. The Copilot API accepts it directly;
    # do not call the undocumented token-exchange endpoint, which returns 404
    # for accounts that work in supported third-party clients.
    store = CopilotTokenStore(secrets)
    store.save_github_token(
        github_token,
        login=user_info.get("login", ""),
        github_id=user_info.get("id", ""),
    )
    store.save_models(fetch_copilot_models(github_token))
    return {
        "ok": True,
        "account": store.account_label(),
    }


# -- Verify probe ---------------------------------------------------------------


def verify(secrets: Any, timeout: float = 10.0) -> dict[str, Any]:
    """Test-button probe: one cheap authenticated request against the Copilot API.

    Distinguishes signed-out vs expired vs OK. Never raises.
    """
    import httpx

    store = CopilotTokenStore(secrets)
    if not store.signed_in():
        return {"ok": False, "error": SIGNED_OUT_ERROR, "state": "signed_out"}
    try:
        token = store.access_token()
    except CopilotSignInRequired as exc:
        return {"ok": False, "error": str(exc), "state": "signed_out"}
    except CopilotAuthError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        resp = httpx.post(
            COPILOT_API_BASE + "/chat/completions",
            headers=api_headers(token),
            json={
                "model": _VERIFY_MODEL,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 16,
                "stream": False,
            },
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Couldn't reach the Copilot API ({exc.__class__.__name__}).",
        }

    if resp.status_code < 300:
        return {"ok": True, "account": store.account_label()}
    if resp.status_code in (401, 403):
        return {"ok": False, "error": EXPIRED_ERROR, "state": "expired"}
    return {
        "ok": False,
        "error": f"The Copilot API returned HTTP {resp.status_code}.",
    }
