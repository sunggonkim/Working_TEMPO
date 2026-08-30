"""Reverse lifecycle order for the long-context saturated validation."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as impl
from eval.sota_4node import vllm_lmcache_live_pd_node_v24 as latest_node


REVERSED_MODES = ("tempo_admission", "lmcache_always_remote")


def _reversed_impl_main() -> int:
    args = impl._parse()
    hosts = args.hosts.split(",")
    impl.common._require(
        len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required"
    )
    impl.common._require(
        args.repo_root.resolve() in args.result_dir.resolve().parents,
        "result directory must be below repository",
    )
    for lifecycle, mode in enumerate(REVERSED_MODES):
        impl._lifecycle(args, lifecycle=lifecycle, mode=mode, hosts=hosts)
    final = args.result_dir / "result.json"
    if args.node_index == 0:
        impl.subprocess.run(
            [
                str(args.repo_root / ".vllm_venv/bin/python"),
                "-m",
                "eval.sota_4node.live_pd_controller_lmcache_v3",
                "combine",
                "--baseline",
                str(args.result_dir / "lmcache_always_remote" / "result.json"),
                "--tempo",
                str(args.result_dir / "tempo_admission" / "result.json"),
                "--output",
                str(final),
            ],
            cwd=args.repo_root,
            check=True,
            timeout=60.0,
        )
    else:
        impl.common._wait_file(final, [])
    return 0


def main() -> int:
    old = impl.main
    impl.main = _reversed_impl_main
    try:
        return latest_node.main()
    finally:
        impl.main = old


if __name__ == "__main__":
    raise SystemExit(main())
