"""The curated model matrix — the only models we actively suggest, label, and vouch for.

Keyed by the FULL routed id, exactly as the ProviderRouter receives it — including reseller
"ugly names" like ``together:zai-org/GLM-5.2`` (bare ids route to the OpenAI default). Each
entry carries the UI display label and the model's capabilities, making this the single
source of truth the capability probe and the GUI's pickers read from.

Deliberately SMALL (owner call, 2026-07-04): current-generation, agent-capable (tool-calling)
models only. It is not user-editable — users can still add any custom model string, which
falls back to the conservative heuristics in ``capabilities.py`` at their own risk of
degraded results. Ids verified against vendor/reseller catalogs on 2026-07-04; refresh the
reseller rows when catalogs rotate (they rename on every model generation).

Context windows (``context_window``, tokens) feed the GUI's context-fill meter. Entries
where the vendor spec wasn't re-checked stay ``None`` — the meter simply hides rather than
showing a made-up denominator. Values entered 2026-07-28 from vendor docs; verify alongside
the id refresh.

Resellers: Together + Fireworks + OpenRouter. TODO: add Groq entries here AND its
descriptor in ``registry.py`` once the current provider surface is tested — deliberately
deferred to bound how much needs verifying at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import ModelCapabilities

_AGENTIC = ModelCapabilities(
    tools=True, vision=False, parallel_tool_calls=True, streaming=True
)
# The native three (OpenAI, Anthropic, Gemini) all take PDFs directly; every
# OpenAI-compatible vendor and reseller in the matrix does not (their chat APIs have
# no inline file part — checked 2026-07-17), so those fall back via pdf_support.py.
_AGENTIC_VISION = ModelCapabilities(
    tools=True, vision=True, pdf=True, parallel_tool_calls=True, streaming=True
)


@dataclass(frozen=True)
class ModelEntry:
    label: str  # UI display name, e.g. "GLM-5.2 · via Together"
    caps: ModelCapabilities = _AGENTIC
    # Max context length in tokens (prompt side), for the GUI's context-fill meter.
    # None = not verified against the vendor spec yet; the meter hides.
    context_window: Optional[int] = None


MATRIX: dict[str, ModelEntry] = {
    # -- first-party ------------------------------------------------------------
    # GPT-5.6 (2026-07-09): number = generation, Sol/Terra/Luna = capability tiers.
    # Bare "gpt-5.6" aliases to Sol server-side; we list the explicit tier ids only.
    # Rolling out — accounts without access get a friendly error (providers/errors.py).
    "gpt-5.6-sol": ModelEntry("GPT-5.6 Sol · OpenAI", _AGENTIC_VISION, 400_000),
    "gpt-5.6-terra": ModelEntry("GPT-5.6 Terra · OpenAI", _AGENTIC_VISION, 400_000),
    "gpt-5.6-luna": ModelEntry("GPT-5.6 Luna · OpenAI", _AGENTIC_VISION, 400_000),
    "gpt-5.5": ModelEntry("GPT-5.5 · OpenAI", _AGENTIC_VISION, 400_000),
    # ChatGPT-subscription catalog (the `openai-codex` OAuth provider). Curated to the
    # ids the subscription backend actually serves; vision per the vendor's model docs,
    # PDF unverified over this backend → local fallback via pdf_support.py.
    # 5.6 tiers (Sol flagship / Terra balanced / Luna fast) serve over the subscription
    # backend by plan — Sol is rate-limited on Plus, full on Pro.
    "openai-codex:gpt-5.6-sol": ModelEntry(
        "GPT-5.6 Sol · ChatGPT plan",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "openai-codex:gpt-5.6-terra": ModelEntry(
        "GPT-5.6 Terra · ChatGPT plan",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "openai-codex:gpt-5.6-luna": ModelEntry(
        "GPT-5.6 Luna · ChatGPT plan",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "openai-codex:gpt-5.2-codex": ModelEntry(
        "GPT-5.2 Codex · ChatGPT plan",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "openai-codex:gpt-5.2": ModelEntry(
        "GPT-5.2 · ChatGPT plan",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "openai-codex:gpt-5.1-codex": ModelEntry(
        "GPT-5.1 Codex · ChatGPT plan",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "openai-codex:gpt-5.1-codex-mini": ModelEntry(
        "GPT-5.1 Codex Mini · ChatGPT plan", _AGENTIC, 400_000
    ),
    # GitHub Copilot catalog (the `github-copilot` OAuth provider). Models available
    # through a Copilot subscription via GitHub's device flow OAuth.
    # Source: https://docs.github.com/en/copilot/reference/ai-models/supported-models
    # Only GA (non-retired) models are listed. Vision per the vendor's model docs;
    # PDFs unverified over this backend → local fallback via pdf_support.py.
    # -- OpenAI models --
    "github-copilot:gpt-5.6-sol": ModelEntry(
        "GPT-5.6 Sol · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "github-copilot:gpt-5.6-terra": ModelEntry(
        "GPT-5.6 Terra · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "github-copilot:gpt-5.6-luna": ModelEntry(
        "GPT-5.6 Luna · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "github-copilot:gpt-5.5": ModelEntry(
        "GPT-5.5 · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "github-copilot:gpt-5.4": ModelEntry(
        "GPT-5.4 · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "github-copilot:gpt-5.4-mini": ModelEntry(
        "GPT-5.4 mini · GitHub Copilot",
        _AGENTIC,
        400_000,
    ),
    "github-copilot:gpt-5.3-codex": ModelEntry(
        "GPT-5.3 Codex · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    "github-copilot:gpt-5-mini": ModelEntry(
        "GPT-5 mini · GitHub Copilot",
        _AGENTIC,
        400_000,
    ),
    "github-copilot:gpt-6-astra": ModelEntry(
        "GPT-6 Astra · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        400_000,
    ),
    # -- Anthropic models --
    "github-copilot:claude-sonnet-5": ModelEntry(
        "Claude Sonnet 5 · GitHub Copilot",
        _AGENTIC_VISION,
        200_000,
    ),
    "github-copilot:claude-sonnet-4.6": ModelEntry(
        "Claude Sonnet 4.6 · GitHub Copilot",
        _AGENTIC_VISION,
        200_000,
    ),
    "github-copilot:claude-opus-5": ModelEntry(
        "Claude Opus 5 · GitHub Copilot",
        _AGENTIC_VISION,
        200_000,
    ),
    "github-copilot:claude-opus-4.8": ModelEntry(
        "Claude Opus 4.8 · GitHub Copilot",
        _AGENTIC_VISION,
        200_000,
    ),
    "github-copilot:claude-haiku-4.5": ModelEntry(
        "Claude Haiku 4.5 · GitHub Copilot",
        _AGENTIC_VISION,
        200_000,
    ),
    "github-copilot:claude-fable-5": ModelEntry(
        "Claude Fable 5 · GitHub Copilot",
        _AGENTIC_VISION,
        200_000,
    ),
    # -- Google models --
    "github-copilot:gemini-3.8-flash": ModelEntry(
        "Gemini 3.8 Flash · GitHub Copilot",
        _AGENTIC_VISION,
        1_048_576,
    ),
    "github-copilot:gemini-3.7-flash": ModelEntry(
        "Gemini 3.7 Flash · GitHub Copilot",
        _AGENTIC_VISION,
        1_048_576,
    ),
    "github-copilot:gemini-3.6-flash": ModelEntry(
        "Gemini 3.6 Flash · GitHub Copilot",
        _AGENTIC_VISION,
        1_048_576,
    ),
    "github-copilot:gemini-3.5-flash": ModelEntry(
        "Gemini 3.5 Flash · GitHub Copilot",
        _AGENTIC_VISION,
        1_048_576,
    ),
    # -- xAI models --
    "github-copilot:grok-4.6": ModelEntry(
        "Grok 4.6 · GitHub Copilot",
        _AGENTIC,
        256_000,
    ),
    "github-copilot:grok-4.5": ModelEntry(
        "Grok 4.5 · GitHub Copilot",
        _AGENTIC,
        256_000,
    ),
    # -- Moonshot AI models --
    "github-copilot:kimi-k3": ModelEntry(
        "Kimi K3 · GitHub Copilot",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        1_000_000,
    ),
    "github-copilot:kimi-k2.7-code": ModelEntry(
        "Kimi K2.7 Code · GitHub Copilot",
        _AGENTIC,
        256_000,
    ),
    # -- Microsoft models --
    "github-copilot:mai-code-1.1-flash": ModelEntry(
        "MAI-Code 1.1 Flash · GitHub Copilot",
        _AGENTIC,
        128_000,
    ),
    # Fable 5 (2026-06-09) is GA; its Mythos 5 sibling is approved-orgs-only, so it
    # stays out of a picker meant for the public.
    "anthropic:claude-fable-5": ModelEntry(
        "Claude Fable 5 · Anthropic", _AGENTIC_VISION, 1_000_000
    ),
    "anthropic:claude-opus-4-8": ModelEntry(
        "Claude Opus 4.8 · Anthropic", _AGENTIC_VISION, 200_000
    ),
    "anthropic:claude-sonnet-4-6": ModelEntry(
        "Claude Sonnet 4.6 · Anthropic", _AGENTIC_VISION, 200_000
    ),
    "anthropic:claude-haiku-4-5": ModelEntry(
        "Claude Haiku 4.5 · Anthropic", _AGENTIC_VISION, 200_000
    ),
    # Gemini 3 (thought signatures required in tool loops — carried via the `_gemini`
    # message sidecar, see gemini_provider.py; ids from the vendor catalog 2026-07-22).
    "gemini:gemini-3.1-pro-preview": ModelEntry(
        "Gemini 3.1 Pro · Google", _AGENTIC_VISION, 1_048_576
    ),
    "gemini:gemini-3.6-flash": ModelEntry(
        "Gemini 3.6 Flash · Google", _AGENTIC_VISION, 1_048_576
    ),
    "gemini:gemini-2.5-pro": ModelEntry(
        "Gemini 2.5 Pro · Google", _AGENTIC_VISION, 1_048_576
    ),
    "gemini:gemini-2.5-flash": ModelEntry(
        "Gemini 2.5 Flash · Google", _AGENTIC_VISION, 1_048_576
    ),
    # Ark Responses API providers (verified 2026-08-14). BytePlus pay-as-you-go and
    # Volcengine Agent Plan intentionally use separate provider prefixes because their
    # endpoints, credentials, regions, and model catalogs are not interchangeable.
    "ark:dola-seed-evolving-latest-version": ModelEntry(
        "Dola Seed Evolving · BytePlus Ark", context_window=256_000
    ),
    "ark:dola-seed-2-1-turbo-260628": ModelEntry(
        "Dola Seed 2.1 Turbo · BytePlus Ark", context_window=256_000
    ),
    "ark-agent-plan-cn:doubao-seed-evolving": ModelEntry(
        "Doubao Seed Evolving · Volcengine Agent Plan", context_window=256_000
    ),
    "ark-agent-plan-cn:doubao-seed-2.1-turbo": ModelEntry(
        "Doubao Seed 2.1 Turbo · Volcengine Agent Plan", context_window=256_000
    ),
    # -- direct OpenAI-compatible vendors ----------------------------------------
    # Muse Spark (Meta Model API, public preview 2026-07-09): multimodal + tools via
    # their OpenAI-compat surface. Vision yes; PDFs unverified over compat — falls
    # back via pdf_support.py like the other compat vendors.
    "meta:muse-spark-1.1": ModelEntry(
        "Muse Spark 1.1 · Meta",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
    ),
    "zai:glm-5.2": ModelEntry("GLM-5.2 · Z AI", _AGENTIC, 128_000),
    "deepseek:deepseek-v4-flash": ModelEntry(
        "DeepSeek V4 Flash · DeepSeek", _AGENTIC, 128_000
    ),
    "deepseek:deepseek-v4-pro": ModelEntry(
        "DeepSeek V4 Pro · DeepSeek", _AGENTIC, 128_000
    ),
    "kimi:kimi-k2.6": ModelEntry("Kimi K2.6 · Moonshot", _AGENTIC, 256_000),
    "minimax:MiniMax-M2.5": ModelEntry("MiniMax M2.5 · MiniMax"),
    "qwen:qwen3-max": ModelEntry("Qwen3 Max · Alibaba", _AGENTIC, 256_000),
    "xai:grok-4.3": ModelEntry("Grok 4.3 · xAI", _AGENTIC, 256_000),
    "mistral:mistral-large-latest": ModelEntry(
        "Mistral Large · Mistral", _AGENTIC, 128_000
    ),
    # -- resellers (their model namespaces, verbatim) -----------------------------
    "together:thinkingmachines/Inkling": ModelEntry("Inkling · via Together"),
    "together:zai-org/GLM-5.2": ModelEntry("GLM-5.2 · via Together", _AGENTIC, 128_000),
    # Kimi K3 on Together (landed late July 2026): 1M window, native vision; PDFs
    # unverified over the compat surface (falls back via pdf_support.py, like Muse Spark).
    "together:moonshotai/Kimi-K3": ModelEntry(
        "Kimi K3 · via Together",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        1_000_000,
    ),
    "together:moonshotai/Kimi-K2.7-Code": ModelEntry(
        "Kimi K2.7 Code · via Together", _AGENTIC, 256_000
    ),
    "together:moonshotai/Kimi-K2.6": ModelEntry(
        "Kimi K2.6 · via Together", _AGENTIC, 256_000
    ),
    "together:deepseek-ai/DeepSeek-V4-Pro": ModelEntry(
        "DeepSeek V4 Pro · via Together", _AGENTIC, 128_000
    ),
    "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelEntry(
        "Llama 4 Maverick · via Together", _AGENTIC, 1_000_000
    ),
    "fireworks:accounts/fireworks/models/glm-5p2": ModelEntry(
        "GLM-5.2 · via Fireworks", _AGENTIC, 128_000
    ),
    "fireworks:accounts/fireworks/models/kimi-k2p6": ModelEntry(
        "Kimi K2.6 · via Fireworks", _AGENTIC, 256_000
    ),
    "fireworks:accounts/fireworks/models/deepseek-v4-pro": ModelEntry(
        "DeepSeek V4 Pro · via Fireworks", _AGENTIC, 128_000
    ),
    "fireworks:accounts/fireworks/models/llama4-maverick-instruct-basic": ModelEntry(
        "Llama 4 Maverick · via Fireworks", _AGENTIC, 1_000_000
    ),
    # OpenRouter slugs are lowercase `<lab>/<model>` (checked against their catalog
    # 2026-07-25); same labs as above, one key for all of them.
    "openrouter:z-ai/glm-5.2": ModelEntry("GLM-5.2 · via OpenRouter", _AGENTIC, 128_000),
    "openrouter:moonshotai/kimi-k2.6": ModelEntry(
        "Kimi K2.6 · via OpenRouter", _AGENTIC, 256_000
    ),
    "openrouter:deepseek/deepseek-v4-pro": ModelEntry(
        "DeepSeek V4 Pro · via OpenRouter", _AGENTIC, 128_000
    ),
    "openrouter:meta-llama/llama-4-maverick": ModelEntry(
        "Llama 4 Maverick · via OpenRouter", _AGENTIC, 1_000_000
    ),
    # Stealth/cloaked alpha (catalog-checked 2026-08-24: 1,048,576 ctx, tool calling).
    # These are temporary lab previews — expect the slug to vanish when the lab ships
    # the real model; keep it until OpenRouter retires it.
    "openrouter:stealth/ox-alpha": ModelEntry(
        "Ox Alpha · via OpenRouter", _AGENTIC, 1_048_576
    ),
    # -- cloud accounts (models running in the user's own AWS/GCP) ----------------
    # Bedrock ids carry a family segment (claude/ → native Anthropic path, other/ →
    # Converse) plus AWS's own `-v<n>:<m>` version suffix. Some regions require the
    # `us.`/`eu.` cross-region inference-profile prefix — custom add-model accepts those.
    "bedrock:claude/anthropic.claude-sonnet-4-6-v1:0": ModelEntry(
        "Claude Sonnet 4.6 · AWS Bedrock", _AGENTIC_VISION, 200_000
    ),
    "bedrock:claude/anthropic.claude-haiku-4-5-v1:0": ModelEntry(
        "Claude Haiku 4.5 · AWS Bedrock", _AGENTIC_VISION, 200_000
    ),
    "bedrock:other/amazon.nova-2-pro-v1:0": ModelEntry(
        "Nova 2 Pro · AWS Bedrock", _AGENTIC, 300_000
    ),
    "bedrock:other/meta.llama4-maverick-17b-instruct-v1:0": ModelEntry(
        "Llama 4 Maverick · AWS Bedrock", _AGENTIC, 1_000_000
    ),
    "bedrock:other/mistral.mistral-large-3-v1:0": ModelEntry(
        "Mistral Large 3 · AWS Bedrock", _AGENTIC, 128_000
    ),
    # Live-verified on Converse 2026-07-26 (complete/stream/tool round trip); asked for
    # two tool calls it emits them one at a time, so parallel stays off.
    "bedrock:other/nvidia.nemotron-super-3-120b": ModelEntry(
        "Nemotron Super 3 120B · AWS Bedrock",
        ModelCapabilities(
            tools=True, vision=False, parallel_tool_calls=False, streaming=True
        ),
    ),
    # Vertex ids carry a family segment too (gemini/ and claude/ → native paths,
    # openweight/ → the MaaS OpenAI-compat endpoint, keeping the publisher segment).
    "vertex:gemini/gemini-3.1-pro-preview": ModelEntry(
        "Gemini 3.1 Pro · Vertex AI", _AGENTIC_VISION, 1_048_576
    ),
    "vertex:gemini/gemini-3.6-flash": ModelEntry(
        "Gemini 3.6 Flash · Vertex AI", _AGENTIC_VISION, 1_048_576
    ),
    "vertex:claude/claude-sonnet-4-6": ModelEntry(
        "Claude Sonnet 4.6 · Vertex AI", _AGENTIC_VISION, 200_000
    ),
    "vertex:claude/claude-haiku-4-5": ModelEntry(
        "Claude Haiku 4.5 · Vertex AI", _AGENTIC_VISION, 200_000
    ),
    "vertex:openweight/meta/llama-4-maverick-17b-128e-instruct-maas": ModelEntry(
        "Llama 4 Maverick · Vertex AI", _AGENTIC, 1_000_000
    ),
    "vertex:openweight/qwen/qwen3-coder-480b-a35b-instruct-maas": ModelEntry(
        "Qwen3 Coder · Vertex AI", _AGENTIC, 256_000
    ),
    # -- OpenCode Go subscription models ----------------------------------------
    # OpenCode Go is a $10/month subscription for open coding models. All models
    # require the x-opencode-session header for routing optimization.
    # Source: https://opencode.ai/docs/go
    "opencode-go:glm-5.3-flash": ModelEntry(
        "GLM-5.3 Flash · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:glm-5.3": ModelEntry(
        "GLM-5.3 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:glm-5.2": ModelEntry(
        "GLM-5.2 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:glm-5.1": ModelEntry(
        "GLM-5.1 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:kimi-k3": ModelEntry(
        "Kimi K3 · OpenCode Go",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        1_000_000,
    ),
    "opencode-go:kimi-k2.7-code": ModelEntry(
        "Kimi K2.7 Code · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:kimi-k2.6": ModelEntry(
        "Kimi K2.6 · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:longcat-2.0": ModelEntry(
        "LongCat 2.0 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:deepseek-v4.1-flash": ModelEntry(
        "DeepSeek V4.1 Flash · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:deepseek-v4-pro": ModelEntry(
        "DeepSeek V4 Pro · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:deepseek-v4-flash": ModelEntry(
        "DeepSeek V4 Flash · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:deepseek-v4-flash-vision-exp": ModelEntry(
        "DeepSeek V4 Flash Vision Exp · OpenCode Go",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        128_000,
    ),
    "opencode-go:mimo-v2.5": ModelEntry(
        "MiMo V2.5 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:mimo-v2.5-pro": ModelEntry(
        "MiMo V2.5 Pro · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:minimax-m3": ModelEntry(
        "MiniMax M3 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:minimax-m2.7": ModelEntry(
        "MiniMax M2.7 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:minimax-m2.5": ModelEntry(
        "MiniMax M2.5 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:muse-spark-1.3-contributor": ModelEntry(
        "Muse Spark 1.3 Contributor · OpenCode Go",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        128_000,
    ),
    "opencode-go:muse-spark-1.2-contributor": ModelEntry(
        "Muse Spark 1.2 Contributor · OpenCode Go",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        128_000,
    ),
    "opencode-go:qwen3.8-max": ModelEntry(
        "Qwen3.8 Max · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:qwen3.8-flash": ModelEntry(
        "Qwen3.8 Flash · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:qwen3.7-max": ModelEntry(
        "Qwen3.7 Max · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:qwen3.7-plus": ModelEntry(
        "Qwen3.7 Plus · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:qwen3.6-plus": ModelEntry(
        "Qwen3.6 Plus · OpenCode Go", _AGENTIC, 256_000
    ),
    "opencode-go:hy4-preview": ModelEntry(
        "Hy4 Preview · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:hy3": ModelEntry(
        "Hy3 · OpenCode Go", _AGENTIC, 128_000
    ),
    "opencode-go:grok-4.6": ModelEntry(
        "Grok 4.6 · OpenCode Go", _AGENTIC, 200_000
    ),
    "opencode-go:gpt-5.6-luna": ModelEntry(
        "GPT 5.6 Luna · OpenCode Go",
        ModelCapabilities(
            tools=True, vision=True, parallel_tool_calls=True, streaming=True
        ),
        272_000,
    ),
}


def entry_for(model: str) -> ModelEntry | None:
    return MATRIX.get(model)


def model_labels() -> dict[str, str]:
    """Full-id → display-label map, shipped to the GUI so every picker shows human names."""
    return {mid: e.label for mid, e in MATRIX.items()}


def model_context_windows() -> dict[str, int]:
    """Full-id → context-window map (verified entries only), for the GUI's fill meter."""
    return {
        mid: e.context_window for mid, e in MATRIX.items() if e.context_window
    }


def models_for_provider(provider: str) -> list[str]:
    """BARE model ids (prefix stripped) the matrix curates for a provider — feeds the
    Settings pane's suggestions and the composer picker so both stay in lockstep with the
    matrix. OpenAI entries are stored without a prefix (bare ids route to the OpenAI
    default), so its list is every un-prefixed id."""
    if provider == "openai":
        return [mid for mid in MATRIX if ":" not in mid]
    prefix = provider + ":"
    return [mid[len(prefix) :] for mid in MATRIX if mid.startswith(prefix)]
