"""
Live integration tests — make real API calls and verify token extraction.

These tests require real API keys and cost real money (fractions of a cent).
They are SKIPPED by default. Run them explicitly:

    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python -m pytest tests/test_live_integration.py -v

Each test captures the payload that would be sent to Tokenr (without actually
sending it) and verifies it against the raw provider response.
"""

import os
import unittest
from unittest.mock import patch
import tokenr


OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")


def _init_tokenr():
    """Initialize tokenr in a testable state (enabled, but we mock the send)."""
    tokenr._config.update({
        "token": "test-live",
        "url": "https://tokenr.co/api/v1/track",
        "enabled": True,
        "timeout": 2,
        "debug": True,
        "default_agent_id": None,
        "default_tags": {},
    })
    tokenr._original_methods.clear()


@unittest.skipUnless(OPENAI_KEY, "OPENAI_API_KEY not set")
class TestLiveOpenAI(unittest.TestCase):

    def setUp(self):
        _init_tokenr()

    @patch.object(tokenr, "_send_tracking")
    def test_real_openai_chat_completion(self, mock_send):
        """Make a real OpenAI call and verify extracted tokens match the response."""
        import openai

        client = openai.OpenAI(api_key=OPENAI_KEY)

        # Patch and call
        tokenr._patch_openai()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say exactly: hello"}],
            max_tokens=10,
        )

        # Verify tracking was called
        self.assertTrue(mock_send.called, "track() was not called")
        data = mock_send.call_args[0][0]

        # The raw response has the ground truth
        raw_prompt = response.usage.prompt_tokens
        raw_completion = response.usage.completion_tokens
        details = getattr(response.usage, 'prompt_tokens_details', None)
        raw_cached = int(getattr(details, 'cached_tokens', 0) or 0)

        # Verify SDK extracted correctly
        self.assertEqual(data["input_tokens"], raw_prompt - raw_cached)
        self.assertEqual(data["output_tokens"], raw_completion)
        self.assertEqual(data.get("cache_read_tokens", 0), raw_cached)
        self.assertEqual(data["model"], response.model)
        self.assertEqual(data["provider"], "openai")

        print(f"\n  OpenAI response: prompt={raw_prompt}, completion={raw_completion}, "
              f"cached={raw_cached}")
        print(f"  Tokenr payload:  input={data['input_tokens']}, output={data['output_tokens']}, "
              f"cache_read={data.get('cache_read_tokens', 0)}")


@unittest.skipUnless(ANTHROPIC_KEY, "ANTHROPIC_API_KEY not set")
class TestLiveAnthropic(unittest.TestCase):

    def setUp(self):
        _init_tokenr()

    @patch.object(tokenr, "_send_tracking")
    def test_real_anthropic_message(self, mock_send):
        """Make a real Anthropic call and verify extracted tokens match the response."""
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        # Patch and call
        tokenr._patch_anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say exactly: hello"}],
        )

        self.assertTrue(mock_send.called, "track() was not called")
        data = mock_send.call_args[0][0]

        # Ground truth from response
        raw_input = response.usage.input_tokens
        raw_output = response.usage.output_tokens
        raw_cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
        raw_cache_read = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

        # Verify SDK extracted correctly
        self.assertEqual(data["input_tokens"], raw_input)
        self.assertEqual(data["output_tokens"], raw_output)
        self.assertEqual(data.get("cache_write_tokens", 0), raw_cache_write)
        self.assertEqual(data.get("cache_read_tokens", 0), raw_cache_read)
        self.assertEqual(data["provider"], "anthropic")

        print(f"\n  Anthropic response: input={raw_input}, output={raw_output}, "
              f"cache_write={raw_cache_write}, cache_read={raw_cache_read}")
        print(f"  Tokenr payload:    input={data['input_tokens']}, output={data['output_tokens']}, "
              f"cache_write={data.get('cache_write_tokens', 0)}, "
              f"cache_read={data.get('cache_read_tokens', 0)}")

    @patch.object(tokenr, "_send_tracking")
    def test_real_anthropic_with_prompt_caching(self, mock_send):
        """Make an Anthropic call with a system prompt large enough to trigger caching."""
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        tokenr._patch_anthropic()

        # System prompt must be >1024 tokens to be cache-eligible.
        # We use a large block of text to ensure caching is triggered.
        long_system = "You are a helpful assistant. " * 200  # ~1000 words

        # First call: should write to cache
        response1 = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=[{"type": "text", "text": long_system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "Say hi"}],
        )

        if mock_send.called:
            data1 = mock_send.call_args[0][0]
            cache_write_1 = data1.get("cache_write_tokens", 0)
            cache_read_1 = data1.get("cache_read_tokens", 0)
            print(f"\n  Call 1 (cache write): input={data1['input_tokens']}, "
                  f"cache_write={cache_write_1}, cache_read={cache_read_1}")

        mock_send.reset_mock()

        # Second call with same system prompt: should read from cache
        response2 = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=[{"type": "text", "text": long_system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "Say bye"}],
        )

        if mock_send.called:
            data2 = mock_send.call_args[0][0]
            cache_write_2 = data2.get("cache_write_tokens", 0)
            cache_read_2 = data2.get("cache_read_tokens", 0)
            print(f"  Call 2 (cache read):  input={data2['input_tokens']}, "
                  f"cache_write={cache_write_2}, cache_read={cache_read_2}")

            # The second call should have cache reads where the first had writes
            raw_cache_read = int(getattr(response2.usage, 'cache_read_input_tokens', 0) or 0)
            self.assertEqual(data2.get("cache_read_tokens", 0), raw_cache_read)
            if raw_cache_read > 0:
                self.assertGreater(cache_read_2, 0, "Expected cache_read_tokens > 0 on second call")


if __name__ == "__main__":
    unittest.main()
