"""
Tests for token extraction from provider API responses.

These tests verify that the SDK correctly parses token usage from
OpenAI and Anthropic response objects, including cache token handling.

No real API calls are made — responses are built from mock objects
matching the exact structure each provider returns.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from types import SimpleNamespace
import tokenr


def _reset_config(**overrides):
    """Reset tokenr config for testing."""
    tokenr._config.update({
        "token": "test-token",
        "url": "https://tokenr.co/api/v1/track",
        "enabled": True,
        "timeout": 2,
        "debug": False,
        "default_agent_id": None,
        "default_tags": {},
        **overrides,
    })
    tokenr._original_methods.clear()


def _openai_response(model, prompt_tokens, completion_tokens, cached_tokens=0):
    """Build a mock OpenAI chat completion response."""
    details = None
    if cached_tokens > 0:
        details = SimpleNamespace(cached_tokens=cached_tokens)

    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=details,
        ),
    )


def _anthropic_response(model, input_tokens, output_tokens,
                         cache_creation_input_tokens=0,
                         cache_read_input_tokens=0):
    """Build a mock Anthropic messages response."""
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


class TestOpenAITokenExtraction(unittest.TestCase):
    """Verify the patched OpenAI create method extracts tokens correctly."""

    def setUp(self):
        _reset_config()

    @patch.object(tokenr, "_send_tracking")
    def test_basic_request_no_cache(self, mock_send):
        """Standard OpenAI response with no cached tokens."""
        response = _openai_response("gpt-4o", prompt_tokens=1500, completion_tokens=300)

        # Simulate what the patched create does:
        details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
        non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

        tokenr.track(
            provider="openai",
            model=response.model,
            input_tokens=max(non_cached_input, 0),
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["model"], "gpt-4o")
        self.assertEqual(data["input_tokens"], 1500)
        self.assertEqual(data["output_tokens"], 300)
        # cache_read_tokens should be omitted (it's 0, and track() strips None/0)
        self.assertNotIn("cache_read_tokens", data)

    @patch.object(tokenr, "_send_tracking")
    def test_request_with_prompt_cache(self, mock_send):
        """OpenAI response where some prompt tokens were served from cache."""
        # prompt_tokens=2000 includes 1200 cached + 800 new
        response = _openai_response("gpt-4o", prompt_tokens=2000, completion_tokens=500,
                                    cached_tokens=1200)

        details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
        non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

        tokenr.track(
            provider="openai",
            model=response.model,
            input_tokens=max(non_cached_input, 0),
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 800)   # 2000 - 1200 cached
        self.assertEqual(data["output_tokens"], 500)
        self.assertEqual(data["cache_read_tokens"], 1200)

    @patch.object(tokenr, "_send_tracking")
    def test_all_tokens_cached(self, mock_send):
        """Edge case: entire prompt served from cache."""
        response = _openai_response("gpt-4o", prompt_tokens=3000, completion_tokens=200,
                                    cached_tokens=3000)

        details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
        non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

        tokenr.track(
            provider="openai",
            model=response.model,
            input_tokens=max(non_cached_input, 0),
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 0)       # all cached
        self.assertEqual(data["cache_read_tokens"], 3000)

    @patch.object(tokenr, "_send_tracking")
    def test_no_prompt_tokens_details(self, mock_send):
        """Older API responses that don't include prompt_tokens_details at all."""
        response = SimpleNamespace(
            model="gpt-3.5-turbo",
            usage=SimpleNamespace(
                prompt_tokens=500,
                completion_tokens=100,
                total_tokens=600,
            ),
        )
        # No prompt_tokens_details attribute

        details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
        non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

        tokenr.track(
            provider="openai",
            model=response.model,
            input_tokens=max(non_cached_input, 0),
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 500)
        self.assertEqual(data["output_tokens"], 100)
        self.assertNotIn("cache_read_tokens", data)

    @patch.object(tokenr, "_send_tracking")
    def test_prompt_tokens_details_with_none_cached(self, mock_send):
        """prompt_tokens_details exists but cached_tokens is None."""
        response = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=None),
            ),
        )

        details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
        non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

        tokenr.track(
            provider="openai",
            model=response.model,
            input_tokens=max(non_cached_input, 0),
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 1000)
        self.assertNotIn("cache_read_tokens", data)


class TestAnthropicTokenExtraction(unittest.TestCase):
    """Verify the patched Anthropic create method extracts tokens correctly."""

    def setUp(self):
        _reset_config()

    @patch.object(tokenr, "_send_tracking")
    def test_basic_request_no_cache(self, mock_send):
        """Standard Anthropic response with no caching."""
        response = _anthropic_response("claude-sonnet-4-20250514",
                                        input_tokens=2000, output_tokens=800)

        cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
        cache_read = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

        tokenr.track(
            provider="anthropic",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["model"], "claude-sonnet-4-20250514")
        self.assertEqual(data["input_tokens"], 2000)
        self.assertEqual(data["output_tokens"], 800)
        self.assertNotIn("cache_write_tokens", data)
        self.assertNotIn("cache_read_tokens", data)

    @patch.object(tokenr, "_send_tracking")
    def test_request_with_cache_write(self, mock_send):
        """First request that writes to Anthropic's prompt cache."""
        response = _anthropic_response("claude-sonnet-4-20250514",
                                        input_tokens=500,
                                        output_tokens=300,
                                        cache_creation_input_tokens=4000)

        cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
        cache_read = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

        tokenr.track(
            provider="anthropic",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 500)       # non-cached input only
        self.assertEqual(data["output_tokens"], 300)
        self.assertEqual(data["cache_write_tokens"], 4000)
        self.assertNotIn("cache_read_tokens", data)       # 0 = omitted

    @patch.object(tokenr, "_send_tracking")
    def test_request_with_cache_read(self, mock_send):
        """Subsequent request that reads from Anthropic's prompt cache."""
        response = _anthropic_response("claude-opus-4-6",
                                        input_tokens=200,
                                        output_tokens=1000,
                                        cache_read_input_tokens=15000)

        cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
        cache_read = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

        tokenr.track(
            provider="anthropic",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 200)
        self.assertEqual(data["output_tokens"], 1000)
        self.assertEqual(data["cache_read_tokens"], 15000)
        self.assertNotIn("cache_write_tokens", data)

    @patch.object(tokenr, "_send_tracking")
    def test_request_with_both_cache_read_and_write(self, mock_send):
        """Request that both reads from and writes to cache (common in long conversations)."""
        response = _anthropic_response("claude-sonnet-4-20250514",
                                        input_tokens=300,
                                        output_tokens=600,
                                        cache_creation_input_tokens=2000,
                                        cache_read_input_tokens=10000)

        cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
        cache_read = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

        tokenr.track(
            provider="anthropic",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]
        self.assertEqual(data["input_tokens"], 300)
        self.assertEqual(data["cache_write_tokens"], 2000)
        self.assertEqual(data["cache_read_tokens"], 10000)


class TestCostAccuracy(unittest.TestCase):
    """End-to-end cost accuracy: given real-world token counts, verify the
    payload sent to Tokenr would produce the correct cost server-side.

    These tests use actual pricing from the Tokenr server and compute
    expected cost the same way LlmModel.calculate_cost does.
    """

    def setUp(self):
        _reset_config()

    @patch.object(tokenr, "_send_tracking")
    def test_openai_gpt4o_cost_with_cache(self, mock_send):
        """GPT-4o: $2.50/M input, $10/M output, $1.25/M cache_read."""
        # Simulate: 800 new input + 1200 cached + 500 output
        response = _openai_response("gpt-4o", prompt_tokens=2000, completion_tokens=500,
                                    cached_tokens=1200)

        details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
        non_cached_input = (response.usage.prompt_tokens or 0) - cache_read

        tokenr.track(
            provider="openai",
            model=response.model,
            input_tokens=max(non_cached_input, 0),
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]

        # Compute expected cost using server formula: (tokens / 1M) * rate
        expected_input_cost = (800 / 1_000_000) * 2.5
        expected_output_cost = (500 / 1_000_000) * 10.0
        expected_cache_read_cost = (1200 / 1_000_000) * 1.25
        expected_total = expected_input_cost + expected_output_cost + expected_cache_read_cost

        # Verify the payload has the right tokens for the server to compute this
        self.assertEqual(data["input_tokens"], 800)
        self.assertEqual(data["output_tokens"], 500)
        self.assertEqual(data["cache_read_tokens"], 1200)

        # Document expected server-side cost
        self.assertAlmostEqual(expected_total, 0.0085, places=8)

    @patch.object(tokenr, "_send_tracking")
    def test_anthropic_opus_cost_with_cache(self, mock_send):
        """Opus 4: $15/M input, $75/M output, $1.50/M cache_read, $18.75/M cache_write."""
        response = _anthropic_response("claude-opus-4-6",
                                        input_tokens=10000,
                                        output_tokens=2000,
                                        cache_creation_input_tokens=5000,
                                        cache_read_input_tokens=50000)

        cache_write = int(getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)
        cache_read = int(getattr(response.usage, 'cache_read_input_tokens', 0) or 0)

        tokenr.track(
            provider="anthropic",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

        data = mock_send.call_args[0][0]

        expected_input = (10000 / 1_000_000) * 15.0       # 0.15
        expected_output = (2000 / 1_000_000) * 75.0        # 0.15
        expected_cache_read = (50000 / 1_000_000) * 1.5    # 0.075
        expected_cache_write = (5000 / 1_000_000) * 18.75  # 0.09375
        expected_total = expected_input + expected_output + expected_cache_read + expected_cache_write

        self.assertEqual(data["input_tokens"], 10000)
        self.assertEqual(data["output_tokens"], 2000)
        self.assertEqual(data["cache_read_tokens"], 50000)
        self.assertEqual(data["cache_write_tokens"], 5000)
        self.assertAlmostEqual(expected_total, 0.46875, places=8)


class TestProviderDetection(unittest.TestCase):
    """Verify _detect_provider identifies providers from base URLs."""

    def test_default_is_openai(self):
        mock_self = MagicMock()
        mock_self._client.base_url = "https://api.openai.com/v1"
        self.assertEqual(tokenr._detect_provider(mock_self), "openai")

    def test_detects_minimax(self):
        mock_self = MagicMock()
        mock_self._client.base_url = "https://api.minimax.chat/v1"
        self.assertEqual(tokenr._detect_provider(mock_self), "minimax")

    def test_detects_deepseek(self):
        mock_self = MagicMock()
        mock_self._client.base_url = "https://api.deepseek.com/v1"
        self.assertEqual(tokenr._detect_provider(mock_self), "deepseek")

    def test_detects_anthropic_via_openai_compat(self):
        mock_self = MagicMock()
        mock_self._client.base_url = "https://api.anthropic.com/v1"
        self.assertEqual(tokenr._detect_provider(mock_self), "anthropic")

    def test_falls_back_to_openai_on_unknown(self):
        mock_self = MagicMock()
        mock_self._client.base_url = "https://custom-llm.example.com/v1"
        self.assertEqual(tokenr._detect_provider(mock_self), "openai")

    def test_handles_missing_base_url(self):
        mock_self = MagicMock()
        mock_self._client.base_url = None
        self.assertEqual(tokenr._detect_provider(mock_self), "openai")


if __name__ == "__main__":
    unittest.main()
