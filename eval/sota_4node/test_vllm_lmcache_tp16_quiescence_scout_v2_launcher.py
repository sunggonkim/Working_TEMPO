from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).parent
NODE = HERE / "vllm_lmcache_tp16_quiescence_scout_node_v2.py"
LAUNCHER = HERE / "run_vllm_lmcache_tp16_quiescence_scout_v2_in_allocation.sh"


def test_v2_static_wiring_and_single_bounded_step() -> None:
    node = NODE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    ast.parse(node, filename=str(NODE))
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert "eval.sota_4node.run_vllm_lmcache_tp16_quiescence_scout_v2" in node
    assert "vllm_lmcache_tp16_quiescence_scout_node_v2.py" in launcher
    assert "TEMPO_TP16_QUIESCENCE_V2_PORT_STRIDE" in launcher
    assert "vllm_lmcache_tp16_quiescence_scout_v2_" in launcher
    assert "real_tp16_quiescence_scout_v1.json" in launcher
    assert '--plan "${PLAN_PATH}"' in launcher
    assert len(re.findall(r"\bsrun\b", launcher)) == 1
    assert "--nodes=4 --ntasks=4 --ntasks-per-node=1" in launcher
    assert re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", launcher) is None
