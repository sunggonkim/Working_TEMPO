import unittest

from eval.sota_4node import analyze_tempo_pd_c4_semantic_load as analysis


def _decision(request_id, *, pair, active, decode, capacity=16):
    return {
        "request_id": request_id,
        "frontend_pair_index": pair,
        "frontend_semantic_load_schema": analysis.LOAD_SCHEMA,
        "frontend_semantic_load_source": analysis.LOAD_SOURCE,
        "frontend_semantic_pair_index": pair,
        "frontend_semantic_active_requests_before": active,
        "frontend_semantic_decode_tokens_before": decode,
        "frontend_semantic_max_num_seqs": capacity,
        "frontend_semantic_occupancy_ratio_before": active / capacity,
    }


class C4SemanticLoadAnalysisTest(unittest.TestCase):
    def test_block_rows_validate_and_summary_capacity_events(self):
        foreground = "epd-tempo-c0-foreground"
        background = "epd-tempo-c0-decoder"
        raw = {
            "c4_phase_screen_contract": {
                "arm": "tempo",
                "replicate": 0,
                "block_sequence_index": 0,
                "request_index": {
                    foreground: {
                        "phase": "c0_cool", "tenant": "foreground"},
                    background: {
                        "phase": "c0_cool", "tenant": "decoder_hot"},
                },
            },
            "router_decisions": [
                _decision(foreground, pair=0, active=8, decode=512),
                _decision(background, pair=1, active=16, decode=1024),
            ],
        }
        contract, rows = analysis._block_rows(raw)
        self.assertEqual(contract["arm"], "tempo")
        self.assertEqual(len(rows), 2)
        summary = analysis._summary(rows)
        self.assertEqual(summary["active_requests_before"]["median"], 8.0)
        self.assertEqual(summary["active_requests_before"]["maximum"], 16.0)
        self.assertEqual(
            summary["capacity_event_fraction"]["at_least_half"], 1.0)
        self.assertEqual(
            summary["capacity_event_fraction"]["at_or_above_max_num_seqs"],
            0.5,
        )
        self.assertEqual(summary["pair_counts"], {0: 1, 1: 1})

    def test_partial_or_inconsistent_semantic_evidence_fails(self):
        request_id = "epd-tempo-c1-foreground"
        raw = {
            "c4_phase_screen_contract": {
                "arm": "tempo",
                "replicate": 0,
                "block_sequence_index": 0,
                "request_index": {
                    request_id: {
                        "phase": "c1_decoder_hot", "tenant": "foreground"},
                },
            },
            "router_decisions": [
                {
                    **_decision(
                        request_id, pair=0, active=4, decode=256),
                    "frontend_semantic_pair_index": 1,
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "pair assignment differs"):
            analysis._block_rows(raw)

    def test_independent_contract_layout_is_normalized(self):
        request_id = "epd-tempo-independent"
        raw = {
            "independent_validation_contract": {
                "arm": "tempo",
                "replicate": 4,
                "sequence": 11,
                "request_index": {
                    request_id: {
                        "phase": "c3_both_hot", "tenant": "foreground"},
                },
            },
            "router_decisions": [
                _decision(request_id, pair=1, active=7, decode=768),
            ],
        }
        contract, rows = analysis._block_rows(raw)
        self.assertEqual(contract["arm"], "tempo")
        self.assertEqual(contract["replicate"], 4)
        self.assertEqual(contract["block_sequence_index"], 11)
        self.assertEqual(
            contract["semantic_contract_name"],
            "independent_validation_contract")
        self.assertEqual(rows[0]["decode_tokens_before"], 768)


if __name__ == "__main__":
    unittest.main()
