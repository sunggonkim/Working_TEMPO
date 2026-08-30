import io
import unittest
from eval.sota_4node import run_tempo_pd_stream_metrics_forced_drain_v38 as client
from eval.sota_4node.test_run_tempo_pd_stream_metrics_v3 import _Clock, _event


class ForcedDrainV39Tests(unittest.TestCase):
    def test_done_then_http_tail_is_drained(self):
        stream = io.BytesIO(b"".join((
            _event({"id":"x","model":"m","choices":[{
                "text":" A","finish_reason":"length",
                "logprobs":{"tokens":[" A"]}}]}),
            _event({"id":"x","model":"m","choices":[],"usage":{
                "prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}),
            b"data: [DONE]\n\nignored-http-tail",
        )))
        record = client._stream_record(
            stream, dispatch_ns=100, run_start_ns=100, expected_tokens=1,
            route="decoder_local_recompute_or_cache", clock_ns=_Clock())
        self.assertTrue(record["http_eof_drained_after_done"])
        self.assertEqual(stream.read(), b"")


if __name__ == "__main__":
    unittest.main()
