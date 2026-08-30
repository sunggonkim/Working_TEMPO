import hashlib
import json
import unittest
from unittest import mock

from eval.sota_4node import run_tempo_pd_stream_metrics_forced_drain_salted_v296 as salted


class SaltedPayloadTest(unittest.TestCase):
    def test_cache_salt_is_stable_and_request_unique(self):
        item = mock.Mock(request_id="request-7")
        seen = {}

        def fake_forced(value, *args, opener, **kwargs):
            request = mock.Mock(
                data=json.dumps({"prompt": "x"}).encode(),
                full_url="http://example.invalid/v1/completions",
                headers={},
            )
            request.get_method.return_value = "POST"
            opener(request)
            return {"ok": True}

        def fake_open(request, **kwargs):
            seen.update(json.loads(request.data))
            return mock.Mock()

        with mock.patch.object(salted.forced, "execute_request", fake_forced):
            salted.execute_request(item, opener=fake_open)
        expected = "tempo-cold-" + hashlib.sha256(b"request-7").hexdigest()
        self.assertEqual(seen["cache_salt"], expected)


if __name__ == "__main__":
    unittest.main()
