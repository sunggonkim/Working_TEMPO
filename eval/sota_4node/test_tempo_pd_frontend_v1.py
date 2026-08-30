from __future__ import annotations

import unittest
from unittest import mock

from eval.sota_4node import tempo_pd_frontend_v1 as frontend
from eval.sota_4node.tempo_pd_frontend_v1 import build_app, pair_index


class TempoPDFrontendTests(unittest.TestCase):
    def test_numeric_suffix_balances_replicas(self) -> None:
        self.assertEqual([pair_index(f"request-{i}", 2) for i in range(6)],
                         [0, 1, 0, 1, 0, 1])

    def test_fallback_hash_is_stable(self) -> None:
        self.assertEqual(pair_index("request-alpha", 2), pair_index("request-alpha", 2))

    def test_requires_two_unique_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "two unique"):
            build_app(["http://a", "http://a"])

    def test_pair_client_retries_connect_only_and_expires_idle_socket(self) -> None:
        transport_value = object()
        client_value = object()
        with (
            mock.patch.object(
                frontend.httpx,
                "AsyncHTTPTransport",
                return_value=transport_value,
            ) as transport,
            mock.patch.object(
                frontend.httpx,
                "AsyncClient",
                return_value=client_value,
            ) as client,
        ):
            self.assertIs(frontend._pair_client("http://pair"), client_value)
        self.assertEqual(transport.call_args.kwargs["retries"], 2)
        self.assertEqual(
            transport.call_args.kwargs["limits"].keepalive_expiry,
            1.0,
        )
        client.assert_called_once_with(
            base_url="http://pair",
            timeout=None,
            transport=transport_value,
        )


if __name__ == "__main__":
    unittest.main()
