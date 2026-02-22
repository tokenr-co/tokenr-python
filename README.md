# Tokenr Python SDK

**Automatic LLM cost tracking in one line of code.**

Track costs from OpenAI, Anthropic, and other LLM providers with zero code changes. Get real-time visibility into spending by agent, feature, team, or any dimension you need.

## Features

- **One-line setup** — Add `tokenr.init()` and you're done
- **Zero code changes** — Use OpenAI/Anthropic SDK as normal
- **Automatic tracking** — Token counts and costs tracked automatically
- **Async by default** — Never slows down your app
- **Multi-provider** — OpenAI, Anthropic (more coming)
- **Rich attribution** — Track by agent, feature, team, or custom tags
- **Production-ready** — Handles errors gracefully, never crashes your app

## Installation

```bash
pip install tokenr
```

## Quickstart

### OpenAI

```python
import tokenr
import openai

# One line to enable tracking
tokenr.init(token="your-tokenr-token")

# Use OpenAI exactly as you normally would
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# That's it — costs are automatically tracked to Tokenr
```

### Anthropic

```python
import tokenr
from anthropic import Anthropic

tokenr.init(token="your-tokenr-token")

client = Anthropic(api_key="your-anthropic-key")
response = client.messages.create(
    model="claude-opus-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Automatically tracked!
```

## Configuration

### Environment Variables

```bash
export TOKENR_TOKEN="your-token"
```

```python
import tokenr
tokenr.init()  # Reads from TOKENR_TOKEN
```

### All Options

```python
tokenr.init(
    token="your-token",           # API token (or TOKENR_TOKEN env var)
    url="https://tokenr.co/...",  # Override API URL (optional)
    agent_id="my-app",            # Default agent ID for all requests
    tags={"environment": "prod"}, # Default tags applied to every request
    enabled=True,                 # Set False to disable tracking entirely
    debug=False,                  # Print tracking info to console
)
```

### Disable in Development

```python
import os
tokenr.init(
    token=os.getenv("TOKENR_TOKEN"),
    enabled=os.getenv("ENV") == "production"
)
```

## Advanced Usage

### Track by Agent

```python
# Option 1: Set a default agent ID at init
tokenr.init(token="...", agent_id="customer-support-bot")

# Option 2: Override per request
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_agent_id="sales-assistant"
)
```

### Track by Feature

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_feature="chat",
    tokenr_agent_id="support-bot"
)
```

### Multi-Tenant Tracking

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_team_id="team-123",
    tokenr_agent_id="shared-bot"
)
```

### Custom Tags

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_tags={
        "customer_id": "cust_123",
        "conversation_id": "conv_456",
        "language": "en",
        "priority": "high"
    }
)
```

### Manual Tracking

For providers not yet auto-patched, or when you need full control:

```python
import tokenr

tokenr.track(
    provider="custom-provider",
    model="custom-model-v1",
    input_tokens=1000,
    output_tokens=500,
    agent_id="my-agent",
    feature_name="translation",
    tags={"language": "es"}
)
```

## Per-Request Parameters

Add any of these to OpenAI or Anthropic calls:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tokenr_agent_id` | `str` | Agent identifier |
| `tokenr_feature` | `str` | Feature name |
| `tokenr_team_id` | `str` | Team or customer ID |
| `tokenr_tags` | `dict` | Arbitrary metadata |

## `tokenr.track()` Reference

```python
tokenr.track(
    provider="openai",    # required
    model="gpt-4",        # required
    input_tokens=100,     # required
    output_tokens=50,     # required
    agent_id=None,
    feature_name=None,
    team_id=None,
    status="success",     # "success" or "error"
    latency_ms=None,
    tags=None,
    requested_at=None,    # ISO 8601 timestamp; defaults to now
)
```

## Supported Providers

| Provider | Auto-Tracking | Manual Tracking |
|----------|:-------------:|:---------------:|
| OpenAI   | Yes           | Yes             |
| Anthropic| Yes           | Yes             |
| Google AI| Coming soon   | Yes             |
| Cohere   | Coming soon   | Yes             |
| Custom   | —             | Yes             |

## How It Works

1. `tokenr.init()` wraps the `create` method on OpenAI and Anthropic clients
2. After each call, token counts are read from the response `usage` field
3. A background daemon thread sends the data to Tokenr asynchronously
4. If tracking fails for any reason, your app is completely unaffected

## Examples

See the [`examples/`](examples/) directory:

- [`basic_openai.py`](examples/basic_openai.py) — Simple OpenAI setup
- [`multi_agent.py`](examples/multi_agent.py) — Multiple agents, multiple providers
- [`saas_multitenant.py`](examples/saas_multitenant.py) — Per-team cost attribution

## Getting Your API Token

1. Sign up at [tokenr.co](https://tokenr.co)
2. Go to **API Tokens** and create a token
3. Copy it — it's shown only once

```bash
export TOKENR_TOKEN="your-token-here"
```

## Troubleshooting

**No data showing up?**

```python
tokenr.init(token="your-token", debug=True)
# Prints: [Tokenr] Tracked: gpt-4 - $0.0012
```

1. Verify `echo $TOKENR_TOKEN` returns your token
2. Confirm you're making actual LLM API calls
3. Check network access to `tokenr.co`

**Errors?** The SDK never raises — it always fails silently. Use `debug=True` to see what's happening.

## Security

This SDK is open source so you can audit exactly what data is sent and when. The short version:

- Only token counts, model names, and your attribution metadata are transmitted
- No prompt content or response content ever leaves your application
- All requests use HTTPS
- The tracker runs in a daemon thread and cannot block your process

## License

MIT — see [LICENSE](LICENSE)

## Support

- Issues: [github.com/tokenr-co/tokenr-python/issues](https://github.com/tokenr-co/tokenr-python/issues)
- Email: support@tokenr.co
