import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from eval.sota_4node import vllm_lmcache_elastic_pd_node as node


def _inherited_command(*_args, **_kwargs):
    return [
        "vllm", "serve",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--max-num-seqs", "8",
        "--max-num-batched-tokens", "8192",
        "--kv-transfer-config", json.dumps({
            "kv_connector_extra_config": {},
        }),
    ]


def _inherited_config(*_args, **_kwargs):
    return (
        "chunk_size: 256\n"
        "local_cpu: False\n"
        "pd_buffer_size: 2147483648\n"
        "nixl_backends: [UCX]\n"
    )


def _inherited_router_command(*_args, **_kwargs):
    return [
        "python", "-m", "eval.sota_4node.tempo_pd_elastic_router_v445",
        "--remote-backend", "official-lmcacheconnectorv1-nixl-ucx",
        "--allow-screen-profile",
        "--queue-wait-ms", "250",
    ]


def _inherited_client_command(*_args, **_kwargs):
    return [
        "python", "-m", "eval.sota_4node.run_tempo_pd_stream_metrics_v1",
    ]


class CanonicalElasticPDNodeTest(unittest.TestCase):
    def command(self, *, is_prefill, decoder_tokens, prefix_caching="0"):
        environment = {
            "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS": str(decoder_tokens),
            "TEMPO_VLLM_DECODER_PREFIX_CACHING": prefix_caching,
        }
        with (
            mock.patch.object(
                node, "_ORIGINAL_VLLM_COMMAND", _inherited_command),
            mock.patch.dict(os.environ, environment, clear=False),
        ):
            return node._vllm_command(is_prefill=is_prefill)

    def test_scheduler_token_budget_is_role_specific(self):
        producer = self.command(is_prefill=True, decoder_tokens=8192)
        decoder = self.command(is_prefill=False, decoder_tokens=8192)
        marker = "--max-num-batched-tokens"
        self.assertEqual(producer[producer.index(marker) + 1], "32768")
        self.assertEqual(decoder[decoder.index(marker) + 1], "8192")

    def test_invalid_decoder_token_budget_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must be 8192"):
            self.command(is_prefill=False, decoder_tokens=4096)

    def test_prefix_caching_is_decoder_only_and_exports_hit_details(self):
        producer = self.command(
            is_prefill=True, decoder_tokens=8192, prefix_caching="1")
        decoder = self.command(
            is_prefill=False, decoder_tokens=8192, prefix_caching="1")
        self.assertIn("--no-enable-prefix-caching", producer)
        self.assertNotIn("--enable-prompt-tokens-details", producer)
        self.assertIn("--enable-prefix-caching", decoder)
        self.assertIn("--enable-prompt-tokens-details", decoder)
        self.assertEqual(
            decoder[:3],
            ["python", "-m", node._CACHE_CONTROL_MODULE],
        )
        self.assertEqual(
            decoder[decoder.index("--block-size") + 1], "16")

    def test_invalid_prefix_caching_setting_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.command(
                is_prefill=False, decoder_tokens=8192, prefix_caching="yes")

    def config(self, *, backend=None, is_prefill=False, pd_buffer=None):
        environment = {"TEMPO_LMCACHE_LOCAL_CPU_GB": "8"}
        if backend is not None:
            environment["TEMPO_LMCACHE_NIXL_BACKEND"] = backend
        if pd_buffer is not None:
            environment["TEMPO_LMCACHE_PD_BUFFER_BYTES"] = str(pd_buffer)
        with (
            mock.patch.object(
                node, "_ORIGINAL_CONFIG_TEXT", _inherited_config),
            mock.patch.dict(os.environ, environment, clear=True),
        ):
            return node._config_text(is_prefill=is_prefill)

    def test_nixl_backend_defaults_to_ucx(self):
        config = self.config()
        self.assertIn("nixl_backends: [UCX]", config)
        self.assertNotIn("nixl_backends: [LIBFABRIC]", config)

    def test_nixl_backend_can_select_libfabric(self):
        config = self.config(backend="LIBFABRIC", is_prefill=True)
        self.assertIn("nixl_backends: [LIBFABRIC]", config)
        self.assertNotIn("nixl_backends: [UCX]", config)
        self.assertIn("local_cpu: True", config)

    def test_invalid_nixl_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must be UCX or LIBFABRIC"):
            self.config(backend="TCP")

    def test_pd_buffer_defaults_to_two_gib(self):
        self.assertIn(
            "pd_buffer_size: 2147483648", self.config())

    def test_pd_buffer_can_be_reduced_to_one_gib(self):
        config = self.config(pd_buffer=1073741824)
        self.assertIn("pd_buffer_size: 1073741824", config)
        self.assertNotIn("pd_buffer_size: 2147483648", config)

    def test_invalid_pd_buffer_fails_closed(self):
        for value in (0, 805306368, "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "TEMPO_LMCACHE_PD_BUFFER_BYTES",
            ):
                self.config(pd_buffer=value)

    def router_command(self, environment):
        with (
            mock.patch.object(
                node, "_ORIGINAL_ROUTER_COMMAND",
                _inherited_router_command),
            mock.patch.dict(os.environ, environment, clear=True),
        ):
            return node._router_command()

    def test_router_profile_identity_tracks_ucx_transport(self):
        command = self.router_command({})
        marker = "--remote-backend"
        self.assertEqual(
            command[command.index(marker) + 1],
            "official-lmcacheconnectorv1-nixl-ucx",
        )

    def test_router_profile_identity_tracks_libfabric_cxi_transport(self):
        command = self.router_command({
            "TEMPO_LMCACHE_NIXL_BACKEND": "LIBFABRIC",
            "FI_PROVIDER": "cxi",
        })
        marker = "--remote-backend"
        self.assertEqual(
            command[command.index(marker) + 1],
            "official-lmcacheconnectorv1-nixl-libfabric-cxi",
        )

    def test_libfabric_profile_identity_requires_cxi_provider(self):
        with self.assertRaisesRegex(ValueError, "FI_PROVIDER=cxi"):
            self.router_command({
                "TEMPO_LMCACHE_NIXL_BACKEND": "LIBFABRIC",
                "FI_PROVIDER": "tcp",
            })

    def test_background_traffic_starts_only_for_measured_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "result"
            stage_dir = result_dir / "tempo_elastic_pd_v445"
            stage_dir.mkdir(parents=True)
            start_file = result_dir / "cxi-background.start"
            environment = {
                "TEMPO_CXI_BACKGROUND_START_FILE": str(start_file),
            }
            with (
                mock.patch.object(
                    node.v445, "_ORIGINAL_CLIENT", _inherited_client_command),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                node._client_command(
                    output=stage_dir / "warmup.raw.json",
                    run_id="tempo_elastic_pd_v445-warmup",
                )
                self.assertFalse(start_file.exists())
                node._client_command(
                    output=stage_dir / "raw.json",
                    run_id="tempo_elastic_pd_v445",
                )
            self.assertEqual(start_file.read_text(encoding="utf-8"), "start\n")

    def test_background_start_file_must_match_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_dir = root / "result" / "tempo_elastic_pd_v445"
            stage_dir.mkdir(parents=True)
            environment = {
                "TEMPO_CXI_BACKGROUND_START_FILE": str(root / "wrong.start"),
            }
            with (
                mock.patch.object(
                    node.v445, "_ORIGINAL_CLIENT", _inherited_client_command),
                mock.patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(ValueError, "share the client result"),
            ):
                node._client_command(
                    output=stage_dir / "raw.json",
                    run_id="tempo_elastic_pd_v445",
                )

    def test_launcher_reserves_all_perlmutter_logical_cpus(self):
        launcher = (
            Path(__file__).resolve().parent
            / "run_tempo_pd_elastic_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("MAIN_CPUS=128", launcher)
        self.assertIn("MAIN_CPUS=120", launcher)
        self.assertIn("SRUN_OVERLAP=(--overlap)", launcher)
        self.assertIn('--cpus-per-task="${MAIN_CPUS}"', launcher)
        self.assertNotIn("--cpus-per-task=64", launcher)

    def test_launcher_gates_cxi_load_until_measured_phase(self):
        root = Path(__file__).resolve().parent
        launcher = (root / "run_tempo_pd_elastic_in_allocation.sh").read_text(
            encoding="utf-8")
        source = (root / "cxi_background_traffic.c").read_text(
            encoding="utf-8")
        self.assertIn("TEMPO_CXI_BACKGROUND_START_FILE", launcher)
        self.assertIn('--start-file "${BACKGROUND_START}"', launcher)
        self.assertIn('"start_observed":true', launcher)
        self.assertIn("--start-file", source)
        self.assertIn("tempo-cxi-background-traffic-3", source)
        self.assertIn('\\"start_observed\\":%s', source)
        self.assertIn("pd-2p2d-incast", source)
        self.assertIn("pd-3p1d-incast", source)
        self.assertIn("node_received_gbps", source)
        self.assertIn("MPICH_OFI_NIC_POLICY=ROUND-ROBIN", launcher)

    def test_launcher_exposes_request_rate_with_canonical_default(self):
        launcher = (Path(__file__).resolve().parent / "run_tempo_pd_elastic_in_allocation.sh").read_text(encoding="utf-8")
        self.assertIn("REQUEST_RATE=${TEMPO_ELASTIC_PD_REQUEST_RATE:-48}", launcher)
        self.assertIn('"${PORT_SLOT}" "${REQUEST_RATE}" 32 128 8 3000 250 16000', launcher)

    def test_canonical_runner_has_explicit_cold_measured_contract(self):
        runner = (
            Path(__file__).resolve().parent / "run_tempo_pd_elastic.py"
        ).read_text(encoding="utf-8")
        self.assertIn("TEMPO_PD_BENCHMARK_COLD_MEASURED", runner)
        self.assertIn('"phase_arm_replicate_and_item"', runner)
        self.assertIn('"cold_disjoint_prompt_keys"', runner)
        self.assertIn(
            'cache_keys_stable_across_phases"] = not cold_measured', runner)
        self.assertIn(
            "marker_id = (1 << 17) | (replicate << 10) | (arm_index << 8) | item",
            runner)
        self.assertNotIn("marker_id = 1_000_000", runner)


if __name__ == "__main__":
    unittest.main()
