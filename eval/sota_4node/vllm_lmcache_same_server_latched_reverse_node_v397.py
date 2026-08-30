from eval.sota_4node import vllm_lmcache_same_server_latched_node_v384 as generic
from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base


def main():
    old = generic.install(
        "eval.sota_4node.run_tempo_pd_same_server_reverse_phasechange_prefixswap_v395",
        "phasechange",
        ("phasechange_paired_v353", "measured.raw.json"),
    )
    real_run = base._bounded_run

    def run(command, *args, **kwargs):
        if (isinstance(command, list)
                and "eval.sota_4node.analyze_tempo_pd_latched_controller_v383" in command):
            command = list(command)
            command[command.index("eval.sota_4node.analyze_tempo_pd_latched_controller_v383")] = (
                "eval.sota_4node.analyze_tempo_pd_latched_reverse_v396")
            class_at = command.index("--workload-class")
            del command[class_at:class_at + 2]
        return real_run(command, *args, **kwargs)

    base._bounded_run = run
    try:
        return base.main()
    finally:
        base._client_command, base._router_command, base._bounded_run = old


if __name__ == "__main__":
    raise SystemExit(main())
