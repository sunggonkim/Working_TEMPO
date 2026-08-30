from __future__ import annotations

import hashlib
import unittest

from tempo.pd_cache_state_protocol import (
    CacheProtocolItem,
    build_cache_preparation_plan,
)
from tempo.pd_contention_workload import CacheState


def _item(
    index: int, state: CacheState, *, arm: str = "tempo",
    prompt_key: str | None = None,
) -> CacheProtocolItem:
    key = prompt_key or hashlib.sha256(
        f"prompt-{index}".encode()).hexdigest()
    marker = state.value.replace("_", "-")
    return CacheProtocolItem(
        request_id=(
            f"epd-{arm}-cache-{marker}-measured-occ-{index:06d}-"
            f"item-{index:06d}"
        ),
        prompt=f"prompt {index}",
        prompt_token_sha256=key,
        prompt_tokens=(512, 2048, 4094, 512)[index % 4],
        output_tokens=(16, 128, 256, 128)[index % 4],
        cache_state=state,
        terminal_item=index,
    )


class CacheStateProtocolTest(unittest.TestCase):
    def test_four_states_generate_exact_ordered_preparation(self):
        items = tuple(_item(index, state) for index, state in enumerate(
            (CacheState.MISS, CacheState.P_ONLY,
             CacheState.D_ONLY, CacheState.BOTH)
        ))
        plan = build_cache_preparation_plan(items)
        self.assertEqual(len(plan.source_probe_rows), 2)
        self.assertEqual(len(plan.decoder_prepare_rows), 4)
        self.assertIn(
            "-warm-cache-p-probe-",
            plan.source_probe_rows[0]["request_id"],
        )
        self.assertIn(
            "-warm-seed-o256-cache-d-seed-",
            plan.decoder_prepare_rows[0]["request_id"],
        )
        self.assertIn(
            "-warm-cache-d-probe-",
            plan.decoder_prepare_rows[1]["request_id"],
        )
        manifest = plan.manifest_dict()
        self.assertEqual(manifest["preparation_order"], [
            "source_probe_rows",
            "quiescent_decoder_apc_reset_preserving_external_lmcache",
            "decoder_prepare_rows",
            "measured_rows",
        ])
        self.assertFalse(manifest["request_id_labels_establish_residency"])

    def test_p_only_and_both_namespaces_can_be_reused_once_prepared(self):
        p = _item(1, CacheState.P_ONLY)
        p_repeat = CacheProtocolItem(
            request_id=(
                "epd-tempo-cache-p-only-measured-occ-000101-item-000001"),
            prompt=p.prompt,
            prompt_token_sha256=p.prompt_token_sha256,
            prompt_tokens=p.prompt_tokens,
            output_tokens=p.output_tokens,
            cache_state=p.cache_state,
            terminal_item=p.terminal_item,
        )
        plan = build_cache_preparation_plan((p, p_repeat))
        self.assertEqual(len(plan.source_probe_rows), 1)
        self.assertEqual(len(plan.decoder_prepare_rows), 0)

    def test_miss_and_d_only_reuse_or_conflicting_state_fails(self):
        for state in (CacheState.MISS, CacheState.D_ONLY):
            first = _item(0, state)
            repeated = CacheProtocolItem(
                request_id=(
                    f"epd-tempo-cache-{state.value.replace('_', '-')}-"
                    "measured-occ-000100-item-000000"),
                prompt=first.prompt,
                prompt_token_sha256=first.prompt_token_sha256,
                prompt_tokens=first.prompt_tokens,
                output_tokens=first.output_tokens,
                cache_state=state,
                terminal_item=0,
            )
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "cannot be measured"):
                    build_cache_preparation_plan((first, repeated))

        shared = hashlib.sha256(b"shared").hexdigest()
        with self.assertRaisesRegex(ValueError, "conflicting cache states"):
            build_cache_preparation_plan((
                _item(0, CacheState.P_ONLY, prompt_key=shared),
                _item(0, CacheState.BOTH, prompt_key=shared),
            ))

    def test_terminal_item_and_state_marker_are_exact(self):
        with self.assertRaisesRegex(ValueError, "terminal item differs"):
            CacheProtocolItem(
                request_id=(
                    "epd-local-cache-miss-measured-occ-000000-item-000001"),
                prompt="x",
                prompt_token_sha256=hashlib.sha256(b"x").hexdigest(),
                prompt_tokens=512,
                output_tokens=16,
                cache_state=CacheState.MISS,
                terminal_item=0,
            )


if __name__ == "__main__":
    unittest.main()
