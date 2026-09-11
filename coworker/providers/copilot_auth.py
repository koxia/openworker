"""GitHub Copilot sign-in for the `github-copilot` provider (OAuth 2.0 Device Flow).

Instead of an API key, the user signs in with their GitHub account that has a Copilot
subscription. The flow uses GitHub's OAuth Device Flow — the user visits a URL and
enters a code, no loopback server needed — then we exchange the GitHub token for a
short-lived Copilot API token.

The pieces:

  - `sign_in()`         — async, explicit-action only: start device flow, poll for
    the GitHub token, fetch the Copilot token, persist both.
  - `CopilotTokenStore` — persistence + proactive refresh of the Copilot token
    (the GitHub token is long-lived; the Copilot token expires in ~2 hours and is
    refreshed on demand using the GitHub token).
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
# The public GitHub CLI client id (ships in the vendor's own tooling — not a secret).
# This client id supports the device flow and has the scopes needed for Copilot access.
CLIENT_ID = "178c6fc778ccc68e1d6a"
# Scope needed for Copilot access. The `copilot` scope grants access to the Copilot API.
SCOPE = "read:user"

# -- Copilot token endpoint ----------------------------------------------------

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_API_BASE = "https://api.githubcopilot.com"
PROFILE = "provider:github-copilot"

# Refresh this close to the Copilot token expiry (tokens last ~2 hours).
REFRESH_MARGIN_SECONDS = 300
FLOW_TIMEOUT_SECONDS = 300
# Device flow default poll interval (GitHub returns one; we fall back to this).
DEFAULT_POLL_INTERVAL = 5

# Editor headers the Copilot API requires (mimicking VS Code).
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


def fetch_copilot_token(github_token: str, timeout: float = 30.0) -> dict[str, Any]:
    """Exchange a GitHub OAuth token for a short-lived Copilot API token.

    Returns the full token response: `{token, expires_in, ...}`.
    Raises `CopilotAuthError` on failure.
    """
    data, status = _post_form(
        COPILOT_TOKEN_URL,
        headers=_copilot_headers(github_token),
        timeout=timeout,
    )
    if status == 401 or status == 403:
        raise CopilotAuthError(NO_COPILOT_ERROR)
    if status == 404:
        # GitHub returns 404 when the account doesn't have API access to Copilot.
        # This happens with Free and Pro plans, which don't support third-party coding agents.
        # Only Pro+, Business, and Enterprise plans have API access.
        raise CopilotAuthError(
            "Your GitHub Copilot plan doesn't support API access for third-party coding agents. "
            "Copilot Free and Pro plans only work within GitHub's official interfaces (VS Code, GitHub.com). "
            "To use Copilot models via API in OpenWorker, you need Copilot Pro+, Business, or Enterprise. "
            "Visit https://github.com/features/copilot/plans to upgrade."
        )
    if status >= 400:
        raise CopilotAuthError(
            f"Failed to obtain Copilot token (HTTP {status})."
        )
    if not data or not data.get("token"):
        raise CopilotAuthError(
            "Copilot token response had no token — your GitHub account may not "
            "have a Copilot subscription."
        )
    return data


def api_headers(copilot_token: str) -> dict[str, str]:
    """Headers for Copilot API calls (chat completions etc.)."""
    return {
        "Authorization": f"Bearer {copilot_token}",
        "Editor-Version": EDITOR_VERSION,
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "Content-Type": "application/json",
    }


# -- Token persistence + refresh ------------------------------------------------


class CopilotTokenStore:
    """Token set in the `provider:github-copilot` SecretStore profile.

    Stores both the long-lived GitHub OAuth token and the short-lived Copilot API
    token. `access_token()` returns a live Copilot token, refreshing it from the
    GitHub token when stale.
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
        """Persist the Copilot API token (short-lived)."""
        patch: dict[str, Any] = {
            "copilot_token": token_data.get("token", ""),
            "copilot_token_saved_at": int(time.time()),
        }
        expires_in = token_data.get("expires_in")
        if isinstance(expires_in, (int, float)):
            patch["copilot_token_expires_at"] = int(time.time()) + int(expires_in)
        self._merge(patch)

    def clear(self) -> bool:
        if self._secrets is None:
            return False
        return bool(self._secrets.delete(PROFILE))

    def access_token(self) -> str:
        """Live Copilot API token — refreshing from the GitHub token when stale."""
        data = self._data()
        github_token = data.get("github_token") or ""
        if not github_token:
            raise CopilotSignInRequired(SIGNED_OUT_ERROR)

        copilot_token = data.get("copilot_token") or ""
        expires_at = data.get("copilot_token_expires_at") or 0
        stale = not copilot_token or (
            isinstance(expires_at, (int, float))
            and expires_at - time.time() < REFRESH_MARGIN_SECONDS
        )
        if stale:
            return self.refresh()
        return copilot_token

    def refresh(self) -> str:
        """Re-fetch the Copilot token using the stored GitHub token."""
        github_token = (self._data().get("github_token") or "")
        if not github_token:
            self.clear()
            raise CopilotSignInRequired(EXPIRED_ERROR)
        try:
            token_data = fetch_copilot_token(github_token)
        except CopilotAuthError:
            # If the GitHub token was rejected, it may have been revoked.
            self.clear()
            raise CopilotSignInRequired(EXPIRED_ERROR)
        self.save_copilot_token(token_data)
        return token_data.get("token", "")


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

    # Step 5: Get the Copilot token
    try:
        copilot_data = await asyncio.to_thread(fetch_copilot_token, github_token)
    except CopilotAuthError as exc:
        raise exc

    # Step 6: Persist everything
    store = CopilotTokenStore(secrets)
    store.save_github_token(
        github_token,
        login=user_info.get("login", ""),
        github_id=user_info.get("id", ""),
    )
    store.save_copilot_token(copilot_data)

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
