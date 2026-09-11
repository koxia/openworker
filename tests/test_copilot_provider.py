"""GitHub Copilot provider (`github-copilot`): device flow, token storage +
refresh, backend request shape, failure modes, and the REST surface. No live
network — the GitHub token endpoint, the Copilot token endpoint, the SDK client,
and the verify probe are all faked."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from coworker.providers import copilot_auth
from coworker.providers.copilot_auth import (
    COPILOT_API_BASE,
    CopilotAuthError,
    CopilotSignInRequired,
    CopilotTokenStore,
    api_headers,
    fetch_copilot_token,
)
from coworker.providers.copilot_provider import CopilotProvider
from coworker.secrets import SecretStore
from coworker.server.app import create_app
from coworker.server.manager import SessionManager


def _token_response(status: int = 200, body: dict | None = None):
    return SimpleNamespace(status_code=status, json=lambda: body or {})


def _seed(secrets, copilot_token: str = "copilot-token-1", expires_offset: float = 7200) -> None:
    store = CopilotTokenStore(secrets)
    store.save_github_token("github-token-1", login="testuser", github_id="12345")
    store.save_copilot_token({"token": copilot_token, "expires_in": expires_offset})


# -- Token store: basic operations ------------------------------------------------


def test_signed_out_when_no_tokens(tmp_path):
    store = CopilotTokenStore(SecretStore(tmp_path / "s.json"))
    assert not store.signed_in()
    with pytest.raises(CopilotSignInRequired, match="Not signed in"):
        store.access_token()


def test_signed_in_after_save(tmp_path):
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets)
    store = CopilotTokenStore(secrets)
    assert store.signed_in()
    assert store.account_label() == "testuser"


def test_access_token_returns_live_token(tmp_path, monkeypatch):
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets)
    # Should not trigger refresh since token is fresh
    monkeypatch.setattr(
        copilot_auth, "fetch_copilot_token",
        lambda *a, **k: pytest.fail("refresh not needed")
    )
    token = CopilotTokenStore(secrets).access_token()
    assert token == "copilot-token-1"


def test_access_token_refreshes_near_expiry(tmp_path, monkeypatch):
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets, expires_offset=30)  # inside the refresh margin
    fresh_token = "fresh-copilot-token"

    def fake_fetch(github_token, timeout=30.0):
        return {"token": fresh_token, "expires_in": 7200}

    monkeypatch.setattr(copilot_auth, "fetch_copilot_token", fake_fetch)
    token = CopilotTokenStore(secrets).access_token()
    assert token == fresh_token


def test_rejected_refresh_blanks_to_signed_out(tmp_path, monkeypatch):
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets, expires_offset=-10)  # expired

    def fake_fetch(github_token, timeout=30.0):
        raise CopilotAuthError("No Copilot subscription")

    monkeypatch.setattr(copilot_auth, "fetch_copilot_token", fake_fetch)
    store = CopilotTokenStore(secrets)
    with pytest.raises(CopilotSignInRequired):
        store.access_token()
    assert not store.signed_in()


def test_clear_removes_all_tokens(tmp_path):
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets)
    assert CopilotTokenStore(secrets).clear()
    assert not CopilotTokenStore(secrets).signed_in()


# -- fetch_copilot_token ----------------------------------------------------------


def test_fetch_copilot_token_success(tmp_path, monkeypatch):
    def fake_post(url, headers=None, timeout=None):
        assert url == copilot_auth.COPILOT_TOKEN_URL
        assert headers["Authorization"] == "Bearer github-token-1"
        return _token_response(body={"token": "new-copilot-token", "expires_in": 7200})

    monkeypatch.setattr(copilot_auth, "_post_form", fake_post)
    result = fetch_copilot_token("github-token-1")
    assert result["token"] == "new-copilot-token"


def test_fetch_copilot_token_no_subscription(tmp_path, monkeypatch):
    def fake_post(url, headers=None, timeout=None):
        return _token_response(status=403)

    monkeypatch.setattr(copilot_auth, "_post_form", fake_post)
    with pytest.raises(CopilotAuthError, match="Copilot subscription"):
        fetch_copilot_token("github-token-1")


# -- api_headers ------------------------------------------------------------------


def test_api_headers_shape():
    headers = api_headers("my-token")
    assert headers["Authorization"] == "Bearer my-token"
    assert headers["Editor-Version"] == copilot_auth.EDITOR_VERSION
    assert headers["Editor-Plugin-Version"] == copilot_auth.EDITOR_PLUGIN_VERSION
    assert headers["Copilot-Integration-Id"] == copilot_auth.COPILOT_INTEGRATION_ID
    assert headers["Content-Type"] == "application/json"


# -- provider: request shape to the backend ---------------------------------------


class _FakeSDKClient:
    def __init__(self, events=None, errors=None):
        self.kwargs: dict = {}
        errors = list(errors or [])

        def create(**kwargs):
            self.kwargs = kwargs
            if errors:
                raise errors.pop(0)
            return iter(events or [])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _chat_response(text="hello"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _status_error(status: int, message: str = ""):
    exc = Exception(message or f"HTTP {status}")
    exc.status_code = status
    return exc


def _provider(tmp_path, monkeypatch, clients):
    """A CopilotProvider over seeded tokens whose SDK clients come from `clients`."""
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets)
    built: list[dict] = []

    def fake_openai(**kwargs):
        built.append(kwargs)
        return clients.pop(0)

    monkeypatch.setattr("openai.OpenAI", fake_openai)
    return CopilotProvider(secrets=secrets), secrets, built


def test_stream_request_headers_and_body(tmp_path, monkeypatch):
    fake = _FakeSDKClient(events=[_chat_response()])
    provider, secrets, built = _provider(tmp_path, monkeypatch, [fake])

    out = list(
        provider.stream(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert built[0]["api_key"] == "copilot-token-1"
    assert built[0]["base_url"] == COPILOT_API_BASE
    headers = built[0]["default_headers"]
    assert headers["Authorization"] == "Bearer copilot-token-1"
    assert headers["Editor-Version"] == copilot_auth.EDITOR_VERSION
    assert fake.kwargs["model"] == "gpt-5.5"
    assert out[-1].turn.text == "hello"


def test_401_refreshes_once_and_retries(tmp_path, monkeypatch):
    first = _FakeSDKClient(errors=[_status_error(401, "Unauthorized")])
    second = _FakeSDKClient(events=[_chat_response("after refresh")])
    provider, secrets, built = _provider(tmp_path, monkeypatch, [first, second])

    def fake_fetch(github_token, timeout=30.0):
        return {"token": "fresh-token", "expires_in": 7200}

    monkeypatch.setattr(copilot_auth, "fetch_copilot_token", fake_fetch)

    turn = provider.complete(
        model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
    )
    assert turn.text == "after refresh"
    assert len(built) == 2
    assert built[1]["api_key"] == "fresh-token"


def test_signed_out_provider_raises_typed_error(tmp_path):
    provider = CopilotProvider(secrets=SecretStore(tmp_path / "s.json"))
    with pytest.raises(CopilotSignInRequired, match="Not signed in"):
        provider.complete(
            model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
        )


# -- registry / matrix ---------------------------------------------------------------


def test_registry_builds_copilot_provider():
    from coworker.providers.registry import build_provider_client, get_descriptor

    assert isinstance(build_provider_client("github-copilot", {}, None), CopilotProvider)
    d = get_descriptor("github-copilot")
    assert d.auth == "oauth" and d.fields == []
    assert d.to_dict()["auth"] == "oauth"


def test_descriptor_configured_means_tokens_present():
    from coworker.providers.registry import descriptor_configured, get_descriptor

    d = get_descriptor("github-copilot")
    assert not descriptor_configured(d, {})
    assert descriptor_configured(d, {"tokens": {"github_token": "g"}})


def test_matrix_curates_copilot_models():
    from coworker.providers.capabilities import capabilities_for
    from coworker.providers.matrix import models_for_provider

    copilot_models = models_for_provider("github-copilot")
    assert "gpt-5.5" in copilot_models
    assert "claude-sonnet-5" in copilot_models
    assert "gemini-3.6-flash" in copilot_models

    caps = capabilities_for("github-copilot:gpt-5.5")
    assert caps.tools and caps.vision and caps.streaming


# -- verify probe --------------------------------------------------------------------


def _verify_with_backend(tmp_path, monkeypatch, status: int):
    secrets = SecretStore(tmp_path / "s.json")
    _seed(secrets)
    probes: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        probes.update(url=url, headers=headers, body=json)
        return SimpleNamespace(status_code=status)

    monkeypatch.setattr("httpx.post", fake_post)
    return copilot_auth.verify(secrets), probes


def test_verify_signed_out(tmp_path):
    result = copilot_auth.verify(SecretStore(tmp_path / "s.json"))
    assert result["ok"] is False and result["state"] == "signed_out"


def test_verify_ok_probes_backend(tmp_path, monkeypatch):
    result, probes = _verify_with_backend(tmp_path, monkeypatch, 200)
    assert result == {"ok": True, "account": "testuser"}
    assert probes["url"] == COPILOT_API_BASE + "/chat/completions"
    assert probes["headers"]["Authorization"].startswith("Bearer ")


def test_verify_expired(tmp_path, monkeypatch):
    result, _ = _verify_with_backend(tmp_path, monkeypatch, 401)
    assert result["ok"] is False and result["state"] == "expired"


# -- REST surface --------------------------------------------------------------------


def _rest(tmp_path):
    manager = SessionManager(data_dir=tmp_path / "data")
    return manager, TestClient(create_app(manager))


def test_providers_list_shows_copilot_oauth_state(tmp_path):
    manager, client = _rest(tmp_path)
    rows = {p["name"]: p for p in client.get("/v1/providers").json()}
    row = rows["github-copilot"]
    assert row["auth"] == "oauth"
    assert row["signed_in"] is False and row["configured"] is False
    assert "gpt-5.5" in row["suggested_models"]

    manager.secrets.put(
        "provider:github-copilot",
        {
            "tokens": {"github_token": "g"},
            "github_token": "g",
            "github_login": "testuser",
        },
    )
    row = {p["name"]: p for p in client.get("/v1/providers").json()}["github-copilot"]
    assert row["signed_in"] is True and row["configured"] is True
    assert row["account"] == "testuser"


def test_signin_route_starts_background_flow(tmp_path, monkeypatch):
    manager, _ = _rest(tmp_path)
    seen = {}

    async def fake_signin():
        seen["called"] = True
        return {"ok": True}

    monkeypatch.setattr(manager, "copilot_signin", fake_signin)
    client = TestClient(create_app(manager))
    assert client.post("/v1/providers/github-copilot/signin").json() == {
        "ok": True,
        "started": True,
    }
    assert seen["called"]
    assert manager._copilot_authorizing is True


def test_status_and_signout_routes(tmp_path):
    manager, client = _rest(tmp_path)
    status = client.get("/v1/providers/github-copilot/status").json()
    assert status["signed_in"] is False and status["authorizing"] is False

    manager.secrets.put(
        "provider:github-copilot",
        {
            "tokens": {"github_token": "g"},
            "github_token": "g",
            "github_login": "testuser",
        },
    )
    status = client.get("/v1/providers/github-copilot/status").json()
    assert status["signed_in"] is True and status["account"] == "testuser"

    assert client.post("/v1/providers/github-copilot/signout").json() == {
        "ok": True,
        "had_tokens": True,
    }
    assert client.get("/v1/providers/github-copilot/status").json()["signed_in"] is False


def test_verify_route_reports_signed_out(tmp_path):
    _, client = _rest(tmp_path)
    result = client.post("/v1/providers/verify", json={"name": "github-copilot"}).json()
    assert result["ok"] is False and result["state"] == "signed_out"


async def test_manager_signin_stores_and_promotes_model(tmp_path, monkeypatch):
    manager, _ = _rest(tmp_path)

    async def fake_sign_in(secrets, **kwargs):
        store = CopilotTokenStore(secrets)
        store.save_github_token("github-token-1", login="testuser", github_id="12345")
        store.save_copilot_token({"token": "copilot-token-1", "expires_in": 7200})
        return {"ok": True, "account": "testuser"}

    monkeypatch.setattr(copilot_auth, "sign_in", fake_sign_in)
    result = await manager.copilot_signin()
    assert result["ok"] is True
    assert manager._copilot_authorizing is False
    settings = manager.get_settings()
    assert "github-copilot:gpt-5.5" in settings["models"]


async def test_manager_signin_failure_lands_in_status(tmp_path, monkeypatch):
    manager, client = _rest(tmp_path)

    async def fake_sign_in(secrets, **kwargs):
        raise CopilotAuthError("GitHub returned an invalid device code response.")

    monkeypatch.setattr(copilot_auth, "sign_in", fake_sign_in)
    result = await manager.copilot_signin()
    assert result["ok"] is False
    status = client.get("/v1/providers/github-copilot/status").json()
    assert "invalid device code" in status["last_error"]
    assert status["authorizing"] is False
