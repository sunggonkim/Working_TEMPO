"""
TEMPO: Temporal Emulation and Masking for Predictable I/O in Large-Scale AI Training
OSDI/SC 2025 Systems Paper — NERSC Perlmutter Experiment Codebase

Core Problem:
    Even in hardware-isolated network topologies (separate storage NIC vs. GPU NIC),
    aggressive checkpoint flushing (NVMe -> RAM -> Slingshot 11 NIC -> Lustre) causes
    PCIe Root Complex contention on the AMD EPYC CPU, degrading NCCL All-Reduce
    bandwidth by up to 40%.

    PCIe Contention Path (Perlmutter GPU node):
        NVMe (PCIe 4.0) ──► AMD EPYC I/O Die ──► Slingshot NIC (PCIe)
        GPU NCCL        ──► AMD EPYC I/O Die ──► Slingshot NIC (PCIe)
                                  ▲
                          CONTENTION POINT
                    (PCIe Root Complex + DRAM BW)

Solution — TEMPO Pacing Scheduler:
    1. PhaseMonitor:      Detects current training phase (NCCL vs. Compute)
    2. CheckpointManager: O(1) local NVMe save + background Lustre flush
    3. TEMPOScheduler:    Pauses/throttles flush during NCCL, resumes during matmul

TEMPO v4 (SOSP-level) additional components:
    4. SparseTransferFilter: InfiniGen-style attention-probe sparse KV selection
       — reduces checkpoint payload ~8.5× (only hot ~12% tokens transferred)
    5. P2PCacheStore: Mooncake-style DHT P2P DRAM/NVMe pool
       — eliminates Lustre metadata latency for cache-hot checkpoints
    6. NanoOverlapController: NanoFlow-style per-layer CUDA stream pipelining
       — eliminates I/O bubble by overlapping KV I/O with next-layer compute

Usage:
    >>> from tempo import TEMPOSchedulerV4
    >>> ctrl = TEMPOSchedulerV4(rank=rank, world_size=ws,
    ...            lustre_dir=os.environ["PSCRATCH"]+"/ckpts")
    >>> for step in range(n_steps):
    ...     ctrl.on_step_begin(step)
    ...     for layer in range(32):
    ...         ctrl.on_layer_event(layer, "start")
    ...         # ... forward pass ...
    ...         ctrl.on_layer_event(layer, "end")
    ...     if step % ckpt_every == 0:
    ...         ctrl.checkpoint(model.state_dict(), step)
    >>> ctrl.shutdown()
"""

from tempo.phase_monitor import PhaseMonitor, TrainingPhase
from tempo.checkpoint_manager import CheckpointManager
from tempo.scheduler import TEMPOScheduler, TEMPOSchedulerV2, TEMPOSchedulerV3, TEMPOSchedulerV4
from tempo.network_monitor import NetworkMonitor, CassiniHWCounters
from tempo.service_gain import ServiceGainScheduler, TokenBucket, FlushPriority
from tempo.interleaving_engine import InterleavingEngine, PhaseDurationPredictor
from tempo.topology_router import TopologyRouter, PlacementDecision, PlacementTier
from tempo.qos_mapper import QoSMapper, TC, TrafficClass
from tempo.sparse_transfer import SparseTransferFilter, SparseKVBlock
from tempo.p2p_cache import P2PCacheStore
from tempo.nano_overlap import NanoOverlapController, LayerTiming, StepMetrics
from tempo.trace_loader import TraceLoader, Request, TraceStats

__version__ = "0.4.0"
__all__ = [
    # Core (v1)
    "PhaseMonitor", "TrainingPhase", "CheckpointManager", "TEMPOScheduler",
    # V2: Communication & I/O-Aware Co-Scheduling
    "TEMPOSchedulerV2",
    "NetworkMonitor",
    "ServiceGainScheduler", "TokenBucket", "FlushPriority",
    "InterleavingEngine", "PhaseDurationPredictor",
    # V3: Topology-Aware + Hardware QoS Co-Design
    "TEMPOSchedulerV3",
    "TopologyRouter", "PlacementDecision", "PlacementTier",
    "QoSMapper", "TC", "TrafficClass",
    # V4: Sparse Transfer + P2P Cache + Nano-Overlap (SOSP-level)
    "TEMPOSchedulerV4",
    "SparseTransferFilter", "SparseKVBlock",
    "P2PCacheStore",
    "NanoOverlapController", "LayerTiming", "StepMetrics",
    # Real hardware + workload (OSDI AE requirements)
    "CassiniHWCounters",
    "TraceLoader", "Request", "TraceStats",
]
