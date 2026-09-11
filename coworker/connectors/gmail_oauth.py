"""Local Google OAuth for Gmail (loopback redirect, no cloud broker needed).

The managed cloud OAuth path is paused (CASA verification pending). This module
provides a direct local OAuth flow: the user authorizes in their browser, the
callback lands on a loopback port, and tokens are stored locally.

First-time setup requires the user to provide their own Google OAuth client ID
and client secret (from Google Cloud Console). These are stored in the SecretStore
and reused for all subsequent Gmail connections.

The pieces:
  - `sign_in()`         — async, explicit-action only: bind loopback port, open
    browser, wait for callback, exchange code, persist tokens.
  - `GmailTokenStore`   — persistence + proactive refresh of access tokens.
  - `verify()`          — the Test-button probe: one cheap authenticated request.

Tokens land in the SecretStore profile `gmail:account:<email>` — the same
multi-account storage the managed path uses.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets as pysecrets
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

logger = logging.getLogger(__name__)

# -- Google OAuth endpoints ----------------------------------------------------

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Gmail scopes: read messages + send + modify labels
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "openid",
    "email",
    "profile",
]

# Loopback callback config
CALLBACK_PORT = 18421  # unlikely to conflict
CALLBACK_PATH = "/oauth/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"

# Profile keys
CLIENT_CONFIG_KEY = "gmail:oauth_client"
PROFILE_PREFIX = "gmail:account:"

# Refresh this close to the JWT `exp` instead of sending an about-to-die bearer.
REFRESH_MARGIN_SECONDS = 300
FLOW_TIMEOUT_SECONDS = 300

SIGNED_OUT_ERROR = (
    "Gmail is not connected — connect your Google account in Settings ▸ Connectors."
)
EXPIRED_ERROR = "Gmail session expired — sign in again in Settings ▸ Connectors."
NO_CLIENT_ERROR = (
    "Google OAuth client not configured. Open Settings ▸ Connectors ▸ Gmail "
    "and click 'Configure Google OAuth' to add your client ID and secret."
)
PORT_BUSY_ERROR = (
    f"Port {CALLBACK_PORT} is already in use. Close other apps using this port "
    "and try again."
)


class GmailAuthError(RuntimeError):
    """A Gmail-auth failure with a user-readable message."""


class GmailSignInRequired(GmailAuthError):
    """No usable tokens — the fix is an explicit sign-in, never a silent browser."""


# -- PKCE / JWT helpers --------------------------------------------------------


def create_pkce() -> tuple[str, str]:
    """(verifier, S256 challenge) per RFC 7636."""
    verifier = pysecrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url(
    client_id: str, state: str, challenge: str, *, login_hint: str = ""
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",  # get refresh token
        "prompt": "consent",  # always show consent to ensure refresh token
    }
    if login_hint:
        params["login_hint"] = login_hint
    return AUTHORIZE_URL + "?" + urlencode(params)


# -- Client config persistence -------------------------------------------------


def get_client_config(secrets: Any) -> Optional[dict[str, str]]:
    """Return the stored Google OAuth client config, or None if not set."""
    if secrets is None:
        return None
    data = secrets.get(CLIENT_CONFIG_KEY) or {}
    client_id = data.get("client_id", "").strip()
    client_secret = data.get("client_secret", "").strip()
    if not client_id or not client_secret:
        return None
    return {"client_id": client_id, "client_secret": client_secret}


def set_client_config(secrets: Any, client_id: str, client_secret: str) -> dict[str, Any]:
    """Store the Google OAuth client config."""
    if not client_id or not client_secret:
        return {"ok": False, "error": "Client ID and secret are required."}
    secrets.put(CLIENT_CONFIG_KEY, {
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
    })
    return {"ok": True}


def clear_client_config(secrets: Any) -> bool:
    """Remove the stored Google OAuth client config."""
    if secrets is None:
        return False
    return bool(secrets.delete(CLIENT_CONFIG_KEY))


# -- Token exchange and refresh ------------------------------------------------


def _token_post(data: dict[str, str], timeout: float = 30.0) -> Any:
    """One POST to the token endpoint (module-level so tests stub the wire here)."""
    import httpx

    return httpx.post(TOKEN_URL, data=data, timeout=timeout)


def exchange_code(
    code: str, verifier: str, client_secret: str, timeout: float = 30.0
) -> dict[str, Any]:
    """authorization_code + PKCE verifier → the token set. Blocking (httpx sync)."""
    resp = _token_post(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": get_client_config.__self__.get(CLIENT_CONFIG_KEY, {}).get("client_id", ""),
            "client_secret": client_secret,
            "code_verifier": verifier,
        },
        timeout,
    )
    if resp.status_code >= 300:
        raise GmailAuthError(
            f"Sign-in failed — token exchange returned HTTP {resp.status_code}."
        )
    return resp.json()


def refresh_access_token(
    refresh_token: str, client_id: str, client_secret: str, timeout: float = 30.0
) -> dict[str, Any]:
    """Refresh the access token using the refresh token."""
    resp = _token_post(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout,
    )
    if resp.status_code >= 300:
        raise GmailAuthError(
            f"Token refresh failed (HTTP {resp.status_code})."
        )
    return resp.json()


# -- Token persistence + refresh -----------------------------------------------


class GmailTokenStore:
    """Token set in the `gmail:account:<email>` SecretStore profile.

    Stores the access token, refresh token, and expiry. `access_token()` returns
    a live token, refreshing proactively near expiry.
    """

    def __init__(self, secrets: Any, email: str = "") -> None:
        self._secrets = secrets
        self._email = email.lower().strip() if email else ""

    def _profile_key(self) -> str:
        if not self._email:
            return ""
        return PROFILE_PREFIX + self._email

    def _data(self) -> dict[str, Any]:
        if self._secrets is None or not self._profile_key():
            return {}
        return self._secrets.get(self._profile_key()) or {}

    def _merge(self, patch: dict[str, Any]) -> None:
        if self._profile_key():
            self._secrets.put(self._profile_key(), {**self._data(), **patch})

    def signed_in(self) -> bool:
        return bool(self._data().get("access_token"))

    def account_email(self) -> str:
        return self._email or self._data().get("account", "")

    def save_tokens(self, tokens: dict[str, Any], email: str = "") -> None:
        """Persist a token response."""
        email = email or self._email
        if not email:
            # Try to get email from the ID token or userinfo
            email = tokens.get("email", "")
        if not email:
            email = "unknown"
        
        self._email = email.lower().strip()
        patch: dict[str, Any] = {
            "access_token": tokens.get("access_token", ""),
            "refresh_token": tokens.get("refresh_token") or self._data().get("refresh_token", ""),
            "token_expiry": int(time.time()) + int(tokens.get("expires_in", 3600)),
            "account": email,
            "type": "oauth",
            "enabled": True,
        }
        # Remove any empty values
        patch = {k: v for k, v in patch.items() if v}
        self._merge(patch)

    def clear(self) -> bool:
        if self._secrets is None or not self._profile_key():
            return False
        return bool(self._secrets.delete(self._profile_key()))

    def access_token(self) -> str:
        """Live access token — refreshing first when stale/absent."""
        data = self._data()
        access = data.get("access_token", "")
        expiry = data.get("token_expiry", 0)
        
        if not access and not data.get("refresh_token"):
            raise GmailSignInRequired(SIGNED_OUT_ERROR)
        
        stale = not access or (
            isinstance(expiry, (int, float)) and expiry - time.time() < REFRESH_MARGIN_SECONDS
        )
        if stale:
            return self.refresh()
        return access

    def refresh(self) -> str:
        """Refresh the access token using the stored refresh token."""
        data = self._data()
        refresh = data.get("refresh_token", "")
        if not refresh:
            self.clear()
            raise GmailSignInRequired(EXPIRED_ERROR)
        
        client_config = get_client_config(self._secrets)
        if not client_config:
            raise GmailAuthError(NO_CLIENT_ERROR)
        
        try:
            tokens = refresh_access_token(
                refresh,
                client_config["client_id"],
                client_config["client_secret"],
            )
        except GmailAuthError:
            self.clear()
            raise GmailSignInRequired(EXPIRED_ERROR)
        
        # Merge with existing data (refresh may not return all fields)
        merged = {**data, **tokens}
        merged["token_expiry"] = int(time.time()) + int(tokens.get("expires_in", 3600))
        self._merge(merged)
        return tokens.get("access_token", "")


# -- Interactive sign-in flow --------------------------------------------------

# The last authorize URL, surfaced over REST so the GUI can offer "reopen sign-in page"
last_authorize_url: Optional[str] = None
_active_server: Optional[asyncio.AbstractServer] = None

_PAGE = """<!doctype html><meta charset="utf-8"><title>OpenWorker</title>
<body style="font-family: system-ui; margin: 4rem auto; max-width: 28rem; text-align: center;">
<h2>{title}</h2><p>{body}</p></body>"""


def _http_response(status: str, title: str, body: str) -> bytes:
    html = _PAGE.format(title=title, body=body).encode("utf-8")
    head = (
        f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(html)}\r\nConnection: close\r\n\r\n"
    )
    return head.encode("ascii") + html


async def _start_callback_server(
    expected_state: str,
) -> tuple[asyncio.AbstractServer, "asyncio.Future[str]"]:
    """Bind the loopback port and resolve the future with the auth code."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            parts = request_line.decode("ascii", errors="replace").split()
            target = urlsplit(parts[1] if len(parts) > 1 else "/")
            if target.path != CALLBACK_PATH:
                writer.write(_http_response("404 Not Found", "Not found", ""))
                return
            query = parse_qs(target.query)
            error = (query.get("error") or [""])[0]
            code = (query.get("code") or [""])[0]
            state = (query.get("state") or [""])[0]
            if error:
                writer.write(
                    _http_response(
                        "400 Bad Request",
                        "Sign-in failed",
                        f"Google reported: {error}. Return to OpenWorker and try again.",
                    )
                )
                if not future.done():
                    future.set_exception(
                        GmailAuthError(f"Sign-in failed — Google returned: {error}")
                    )
                return
            if not code or not pysecrets.compare_digest(state, expected_state):
                writer.write(
                    _http_response(
                        "400 Bad Request",
                        "Nothing waiting for this sign-in",
                        "The sign-in may have timed out. Return to OpenWorker and start it again.",
                    )
                )
                return
            writer.write(
                _http_response(
                    "200 OK",
                    "Signed in",
                    "You can close this tab and return to OpenWorker.",
                )
            )
            if not future.done():
                future.set_result(code)
        finally:
            try:
                await writer.drain()
                writer.close()
            except Exception:
                pass

    try:
        server = await asyncio.start_server(handle, "127.0.0.1", CALLBACK_PORT)
    except OSError as exc:
        raise GmailAuthError(PORT_BUSY_ERROR) from exc
    return server, future


async def sign_in(
    secrets: Any,
    *,
    timeout: float = FLOW_TIMEOUT_SECONDS,
    open_browser: bool = True,
) -> dict[str, Any]:
    """Run the full interactive flow: loopback server → browser → code → tokens.

    Explicit-action only (a Settings button) — never called from an engine turn.
    """
    global last_authorize_url, _active_server
    
    # Check client config first
    client_config = get_client_config(secrets)
    if not client_config:
        raise GmailAuthError(NO_CLIENT_ERROR)
    
    if _active_server is not None:
        _active_server.close()
        await _active_server.wait_closed()
        _active_server = None
    
    verifier, challenge = create_pkce()
    state = pysecrets.token_urlsafe(24)
    url = build_authorize_url(client_config["client_id"], state, challenge)
    last_authorize_url = url
    server, code_future = await _start_callback_server(state)
    _active_server = server
    
    try:
        if open_browser:
            import webbrowser
            logger.info("gmail auth: opening browser for sign-in")
            await asyncio.get_running_loop().run_in_executor(None, webbrowser.open, url)
        try:
            code = await asyncio.wait_for(code_future, timeout)
        except asyncio.TimeoutError:
            raise GmailAuthError(
                f"Sign-in timed out — the browser window was not completed in "
                f"{int(timeout) // 60} minutes."
            )
    finally:
        server.close()
        await server.wait_closed()
        if _active_server is server:
            _active_server = None
    
    # Exchange code for tokens
    import httpx
    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_config["client_id"],
            "client_secret": client_config["client_secret"],
            "code_verifier": verifier,
        },
        timeout=30.0,
    )
    if resp.status_code >= 300:
        raise GmailAuthError(
            f"Sign-in failed — token exchange returned HTTP {resp.status_code}."
        )
    tokens = resp.json()
    
    if not tokens.get("access_token"):
        raise GmailAuthError("Sign-in failed — the token response had no access token.")
    
    # Get user email
    email = tokens.get("email", "")
    if not email:
        # Fetch from userinfo endpoint
        try:
            userinfo_resp = httpx.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
                timeout=10.0,
            )
            if userinfo_resp.status_code < 300:
                email = userinfo_resp.json().get("email", "")
        except Exception:
            pass
    
    if not email:
        email = "unknown"
    
    # Store tokens
    store = GmailTokenStore(secrets, email)
    store.save_tokens(tokens, email)
    
    # Update the default account pointer
    from . import gmail_accounts
    gmail_accounts.managed_connect_account(secrets, {
        "account": email,
        "access_token": tokens.get("access_token", ""),
        "refresh_token": tokens.get("refresh_token", ""),
        "token_expiry": int(time.time()) + int(tokens.get("expires_in", 3600)),
        "type": "oauth",
        "enabled": True,
    })
    
    return {"ok": True, "account": email}


# -- Verify probe --------------------------------------------------------------


def verify(secrets: Any, email: str = "", timeout: float = 10.0) -> dict[str, Any]:
    """Test-button probe: one cheap authenticated request against Gmail API.

    Distinguishes signed-out vs expired vs OK. Never raises.
    """
    import httpx
    
    store = GmailTokenStore(secrets, email)
    if not store.signed_in():
        return {"ok": False, "error": SIGNED_OUT_ERROR, "state": "signed_out"}
    try:
        token = store.access_token()
    except GmailSignInRequired as exc:
        return {"ok": False, "error": str(exc), "state": "signed_out"}
    except GmailAuthError as exc:
        return {"ok": False, "error": str(exc)}
    
    try:
        resp = httpx.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/labels",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Couldn't reach Gmail API ({exc.__class__.__name__}).",
        }
    
    if resp.status_code < 300:
        return {"ok": True, "account": store.account_email()}
    if resp.status_code in (401, 403):
        return {"ok": False, "error": EXPIRED_ERROR, "state": "expired"}
    return {
        "ok": False,
        "error": f"Gmail API returned HTTP {resp.status_code}.",
    }
