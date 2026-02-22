# Tokenr Quickstart

Get cost tracking running in under 2 minutes.

## 1. Install

```bash
pip install tokenr
```

## 2. Get Your API Token

1. Sign up at [tokenr.co](https://tokenr.co)
2. Go to **API Tokens** and create a new token
3. Copy it (shown only once)

## 3. Set Environment Variable

```bash
export TOKENR_TOKEN="your-token-here"
```

Or add to your `.env` file:

```
TOKENR_TOKEN=your-token-here
```

## 4. Add One Line to Your Code

### OpenAI

```python
import tokenr
import openai

tokenr.init()  # reads TOKENR_TOKEN automatically

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic

```python
import tokenr
from anthropic import Anthropic

tokenr.init()

client = Anthropic()
response = client.messages.create(
    model="claude-opus-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 5. Check Your Dashboard

Go to [tokenr.co/dashboard](https://tokenr.co/dashboard)

You should see your request with token counts, cost, model, and timestamp.

---

## Next Steps

### Track Different Agents

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_agent_id="customer-support-bot"
)
```

### Track by Feature

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_feature="chat"
)
```

### Track by Team

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tokenr_team_id="team-123"
)
```

### Debug Mode

```python
tokenr.init(debug=True)
# Prints: [Tokenr] Tracked: gpt-4 - $0.0012
```

---

## Troubleshooting

**Not seeing data?**
1. Check `echo $TOKENR_TOKEN` returns your token
2. Enable `debug=True`
3. Make sure you're making actual LLM API calls after calling `tokenr.init()`

**Errors in my app?**
The SDK never raises exceptions from tracking. Your app always runs normally.

---

Full docs: [README.md](README.md)
