import os
import unittest
from unittest.mock import patch

import live_injuries


class LiveInjuryCredentialTests(unittest.TestCase):
    def test_missing_api_key_skips_network_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                live_injuries.requests,
                "get",
                side_effect=AssertionError("HTTP must not be called without a key"),
            ):
                self.assertEqual(live_injuries.fetch_live_injuries(), {})

    def test_api_key_is_read_from_environment_for_request(self):
        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": [
                        {
                            "player": {
                                "first_name": "Test",
                                "last_name": "Player",
                            },
                            "team": {"full_name": "Boston Celtics"},
                            "status": "Out",
                            "description": "Test injury",
                        }
                    ]
                }

        with patch.dict(os.environ, {"BALLDONTLIE_API_KEY": "test-key"}, clear=True):
            with patch.object(
                live_injuries.requests, "get", return_value=FakeResponse()
            ) as mocked:
                result = live_injuries.fetch_live_injuries()

        self.assertEqual(result["Boston Celtics"][0]["player"], "Test Player")
        self.assertEqual(result["Boston Celtics"][0]["status"], "Out")
        self.assertEqual(mocked.call_args.kwargs["headers"]["Authorization"], "test-key")


if __name__ == "__main__":
    unittest.main(verbosity=2)
