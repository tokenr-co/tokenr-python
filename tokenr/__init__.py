"""
Tokenr Python SDK - Automatic LLM cost tracking

Install:
    pip install tokenr

Usage:
    import tokenr
    tokenr.init(token="your-token")

    # Use OpenAI/Anthropic as normal - tracking is automatic!
    import openai
    response = openai.chat.completions.create(...)
"""

import os
import threading
import time
from typing import Optional, Dict, Any
from functools import wraps

try:
    import requests
except ImportError:
    requests = None

__version__ = "0.1.2"

# Global configuration
_config = {
    "token": None,
    "url": "https://tokenr.co/api/v1/track",
    "enabled": True,
    "timeout": 2,
    "debug": False,
    "default_agent_id": None,
    "default_tags": {},
}

# Track original methods for restoration
_original_methods = {}


def init(
    token: Optional[str] = None,
    url: Optional[str] = None,
    agent_id: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    debug: bool = False,
):
    """
    Initialize Tokenr tracking.

    Args:
        token: Your Tokenr API token (or set TOKENR_TOKEN env var)
        url: Tokenr API URL (default: https://tokenr.co/api/v1/track)
        agent_id: Default agent ID for all requests
        tags: Default tags to include with all requests
        enabled: Enable/disable tracking (useful for dev/prod switching)
        debug: Print debug information

    Example:
        import tokenr
        tokenr.init(token="your-token", agent_id="my-app")

        # Now use OpenAI/Anthropic as normal
        import openai
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        # Usage is automatically tracked!
    """
    _config["token"] = token or os.getenv("TOKENR_TOKEN")
    _config["url"] = url or os.getenv("TOKENR_URL", _config["url"])
    _config["enabled"] = enabled
    _config["debug"] = debug
    _config["default_agent_id"] = agent_id
    _config["default_tags"] = tags or {}

    if not _config["token"]:
        if debug:
            print("[Tokenr] Warning: No API token provided. Set TOKENR_TOKEN or pass token to init()")
        return

    if not requests:
        if debug:
            print("[Tokenr] Warning: requests library not found. Install with: pip install requests")
        return

    # Auto-patch OpenAI and Anthropic if available
    _patch_openai()
    _patch_anthropic()

    if debug:
        print(f"[Tokenr] Initialized with token: {_config['token'][:8]}...")


def configure(**kwargs):
    """Update configuration after initialization"""
    _config.update(kwargs)


def track(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    agent_id: Optional[str] = None,
    feature_name: Optional[str] = None,
    team_id: Optional[str] = None,
    status: str = "success",
    latency_ms: Optional[int] = None,
    tags: Optional[Dict[str, Any]] = None,
    requested_at: Optional[str] = None,
):
    """
    Manually track an LLM request.

    Use this if you want explicit control over tracking,
    or if you're using a provider that isn't auto-patched.

    Args:
        provider: Provider name (openai, anthropic, google, etc.)
        model: Model name (gpt-4, claude-sonnet-4-5, etc.)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        agent_id: Agent identifier (overrides default)
        feature_name: Feature name for attribution
        team_id: Team ID for multi-tenant tracking
        status: Request status (success, error)
        latency_ms: Request latency in milliseconds
        tags: Additional metadata tags
        requested_at: ISO timestamp (defaults to now)
    """
    if not _config["enabled"] or not _config["token"]:
        return

    merged_tags = {**_config["default_tags"], **(tags or {})}

    data = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens if cache_read_tokens else None,
        "cache_write_tokens": cache_write_tokens if cache_write_tokens else None,
        "agent_id": agent_id or _config["default_agent_id"],
        "feature_name": feature_name,
        "team_id": team_id,
        "status": status,
        "latency_ms": latency_ms,
        "tags": merged_tags if merged_tags else None,
        "requested_at": requested_at,
    }

    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}

    _send_tracking(data)


def _send_tracking(data: Dict[str, Any]):
    """Send tracking data asynchronously"""
    def _send():
        try:
            response = requests.post(
                _config["url"],
                headers={
                    "Authorization": f"Bearer {_config['token']}",
                    "Content-Type": "application/json",
                },
                json=data,
                timeout=_config["timeout"],
            )

            if _config["debug"]:
                if response.ok:
                    result = response.json()
                    print(f"[Tokenr] Tracked: {data.get('model')} - ${result.get('cost', 0)}")
                else:
                    print(f"[Tokenr] Error: {response.status_code} - {response.text}")
        except Exception as e:
            if _config["debug"]:
                print(f"[Tokenr] Failed to track: {e}")

    # Send in background thread
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()


# Maps base URL substrings to Tokenr provider slugs for OpenAI-compatible APIs
_OPENAI_COMPAT_PROVIDERS = {
    "minimax": "minimax",
    "anthropic": "anthropic",
    "googleapis": "google",
    "mistral": "mistral",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "x.ai": "xai",
    "xai": "xai",
    "azure": "azure_openai",
}


def _detect_provider(completions_self) -> str:
    """Detect the real provider slug from an OpenAI Completions instance's base URL."""
    try:
        client = getattr(completions_self, "_client", None)
        base_url = str(getattr(client, "base_url", "") or "").lower()
        for keyword, slug in _OPENAI_COMPAT_PROVIDERS.items():
            if keyword in base_url:
                return slug
    except Exception:
        pass
    return "openai"


def _patch_openai():
    """Automatically patch OpenAI client to track usage"""
    try:
        import openai
        from openai import OpenAI
        from openai.resources.chat import completions as chat_completions
    except ImportError:
        return

    # Patch the chat completions create method
    original_create = chat_completions.Completions.create

    if original_create in _original_methods:
        return  # Already patched

    _original_methods[original_create] = True

    @wraps(original_create)
    def tracked_create(self, *args, **kwargs):
        # Extract metadata from kwargs
        agent_id = kwargs.pop("tokenr_agent_id", None)
        feature_name = kwargs.pop("tokenr_feature", None)
        team_id = kwargs.pop("tokenr_team_id", None)
        tags = kwargs.pop("tokenr_tags", None)

        start_time = time.time()

        # Call original method
        response = original_create(self, *args, **kwargs)

        latency_ms = int((time.time() - start_time) * 1000)

        # Track usage if available
        if hasattr(response, 'usage') and response.usage:
            provider = _detect_provider(self)

            # Extract cache token counts from prompt_tokens_details if present.
            # prompt_tokens = non-cached input + cached reads (all billed together by the
            # provider but at different rates). We separate them so Tokenr can price each
            # category correctly.
            details = getattr(response.usage, 'prompt_tokens_details', None)
            cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
            non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

            track(
                provider=provider,
                model=response.model,
                input_tokens=max(non_cached_input, 0),
                output_tokens=response.usage.completion_tokens,
                cache_read_tokens=cache_read,
                agent_id=agent_id,
                feature_name=feature_name,
                team_id=team_id,
                latency_ms=latency_ms,
                tags=tags,
            )

        return response

    chat_completions.Completions.create = tracked_create


def _patch_anthropic():
    """Automatically patch Anthropic client to track usage"""
    try:
        from anthropic.resources import messages as anthropic_messages
    except ImportError:
        return

    original_create = anthropic_messages.Messages.create

    if original_create in _original_methods:
        return  # Already patched

    _original_methods[original_create] = True

    @wraps(original_create)
    def tracked_create(self, *args, **kwargs):
        # Extract metadata from kwargs
        agent_id = kwargs.pop("tokenr_agent_id", None)
        feature_name = kwargs.pop("tokenr_feature", None)
        team_id = kwargs.pop("tokenr_team_id", None)
        tags = kwargs.pop("tokenr_tags", None)

        start_time = time.time()

        # Call original method
        response = original_create(self, *args, **kwargs)

        latency_ms = int((time.time() - start_time) * 1000)

        # Track usage if available
        if hasattr(response, 'usage') and response.usage:
            # Anthropic reports cache tokens explicitly and separately from input_tokens.
            # input_tokens = non-cached input only (already excludes cache hits/writes).
            cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
            cache_read  = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

            track(
                provider="anthropic",
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                agent_id=agent_id,
                feature_name=feature_name,
                team_id=team_id,
                latency_ms=latency_ms,
                tags=tags,
            )

        return response

    anthropic_messages.Messages.create = tracked_create


# Convenience exports
__all__ = ["init", "track", "configure", "__version__"]
