"""
Tests for the Tokenr Python SDK
"""

import json
import threading
import unittest
from unittest.mock import patch, MagicMock, call
import tokenr
from tokenr import __version__


class TestInit(unittest.TestCase):

    def setUp(self):
        # Reset config before each test
        tokenr._config.update({
            "token": None,
            "url": "https://tokenr.co/api/v1/track",
            "enabled": True,
            "timeout": 2,
            "debug": False,
            "default_agent_id": None,
            "default_tags": {},
        })
        tokenr._original_methods.clear()

    def test_version_exists(self):
        self.assertIsNotNone(__version__)
        self.assertIsInstance(__version__, str)

    def test_init_with_token(self):
        tokenr.init(token="test-token-123", enabled=False)
        self.assertEqual(tokenr._config["token"], "test-token-123")

    def test_init_reads_env_token(self):
        with patch.dict("os.environ", {"TOKENR_TOKEN": "env-token-456"}):
            tokenr.init(enabled=False)
        self.assertEqual(tokenr._config["token"], "env-token-456")

    def test_init_explicit_token_overrides_env(self):
        with patch.dict("os.environ", {"TOKENR_TOKEN": "env-token"}):
            tokenr.init(token="explicit-token", enabled=False)
        self.assertEqual(tokenr._config["token"], "explicit-token")

    def test_init_sets_agent_id(self):
        tokenr.init(token="tok", agent_id="my-agent", enabled=False)
        self.assertEqual(tokenr._config["default_agent_id"], "my-agent")

    def test_init_sets_tags(self):
        tokenr.init(token="tok", tags={"env": "prod"}, enabled=False)
        self.assertEqual(tokenr._config["default_tags"], {"env": "prod"})

    def test_init_disabled(self):
        tokenr.init(token="tok", enabled=False)
        self.assertFalse(tokenr._config["enabled"])

    def test_init_no_token_does_not_crash(self):
        # Should return silently, not raise
        tokenr.init(enabled=False)
        self.assertIsNone(tokenr._config["token"])

    def test_configure_updates_config(self):
        tokenr.configure(debug=True, timeout=5)
        self.assertTrue(tokenr._config["debug"])
        self.assertEqual(tokenr._config["timeout"], 5)


class TestTrack(unittest.TestCase):

    def setUp(self):
        tokenr._config.update({
            "token": "test-token",
            "url": "https://tokenr.co/api/v1/track",
            "enabled": True,
            "timeout": 2,
            "debug": False,
            "default_agent_id": None,
            "default_tags": {},
        })

    def test_track_does_nothing_when_disabled(self):
        tokenr._config["enabled"] = False
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=10, output_tokens=5)
            mock_send.assert_not_called()

    def test_track_does_nothing_without_token(self):
        tokenr._config["token"] = None
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=10, output_tokens=5)
            mock_send.assert_not_called()

    def test_track_sends_required_fields(self):
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=100, output_tokens=50)
            mock_send.assert_called_once()
            data = mock_send.call_args[0][0]
            self.assertEqual(data["provider"], "openai")
            self.assertEqual(data["model"], "gpt-4")
            self.assertEqual(data["input_tokens"], 100)
            self.assertEqual(data["output_tokens"], 50)

    def test_track_uses_default_agent_id(self):
        tokenr._config["default_agent_id"] = "default-agent"
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=10, output_tokens=5)
            data = mock_send.call_args[0][0]
            self.assertEqual(data["agent_id"], "default-agent")

    def test_track_per_call_agent_id_overrides_default(self):
        tokenr._config["default_agent_id"] = "default-agent"
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=10, output_tokens=5,
                         agent_id="override-agent")
            data = mock_send.call_args[0][0]
            self.assertEqual(data["agent_id"], "override-agent")

    def test_track_merges_default_tags(self):
        tokenr._config["default_tags"] = {"env": "prod"}
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=10, output_tokens=5,
                         tags={"feature": "chat"})
            data = mock_send.call_args[0][0]
            self.assertEqual(data["tags"], {"env": "prod", "feature": "chat"})

    def test_track_omits_none_values(self):
        with patch.object(tokenr, "_send_tracking") as mock_send:
            tokenr.track(provider="openai", model="gpt-4", input_tokens=10, output_tokens=5)
            data = mock_send.call_args[0][0]
            self.assertNotIn("agent_id", data)
            self.assertNotIn("feature_name", data)
            self.assertNotIn("team_id", data)
            self.assertNotIn("tags", data)


class TestSendTracking(unittest.TestCase):

    def setUp(self):
        tokenr._config.update({
            "token": "test-token",
            "url": "https://tokenr.co/api/v1/track",
            "enabled": True,
            "timeout": 2,
            "debug": False,
            "default_agent_id": None,
            "default_tags": {},
        })

    def test_send_tracking_uses_background_thread(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"cost": 0.002}

        threads_before = threading.active_count()

        with patch("tokenr.requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            tokenr._send_tracking({"model": "gpt-4", "provider": "openai"})
            # Give the thread a moment to start
            import time; time.sleep(0.05)

        # Should have started a background thread
        mock_requests.post.assert_called_once()

    def test_send_tracking_uses_bearer_auth(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {}

        with patch("tokenr.requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            tokenr._send_tracking({"model": "gpt-4"})
            import time; time.sleep(0.05)

        call_kwargs = mock_requests.post.call_args[1]
        self.assertIn("Authorization", call_kwargs["headers"])
        self.assertTrue(call_kwargs["headers"]["Authorization"].startswith("Bearer "))

    def test_send_tracking_does_not_raise_on_error(self):
        with patch("tokenr.requests") as mock_requests:
            mock_requests.post.side_effect = Exception("network error")
            # Should not raise
            tokenr._send_tracking({"model": "gpt-4"})
            import time; time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
