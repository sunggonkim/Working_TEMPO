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

Usage:
    >>> from tempo import TEMPOScheduler
    >>> tempo = TEMPOScheduler(model, optimizer, rank=rank, world_size=ws)
    >>> for step, batch in enumerate(loader):
    ...     with tempo.compute_phase():
    ...         loss = model(batch); loss.backward()
    ...     with tempo.nccl_phase():
    ...         optimizer.step()           # DDP all_reduce inside
    ...     if step % ckpt_every == 0:
    ...         tempo.save_checkpoint(step, model.state_dict())
    >>> tempo.shutdown()
"""

from tempo.phase_monitor import PhaseMonitor, TrainingPhase
from tempo.checkpoint_manager import CheckpointManager
from tempo.scheduler import TEMPOScheduler, TEMPOSchedulerV2, TEMPOSchedulerV3
from tempo.network_monitor import NetworkMonitor
from tempo.service_gain import ServiceGainScheduler, TokenBucket, FlushPriority
from tempo.interleaving_engine import InterleavingEngine, PhaseDurationPredictor
from tempo.topology_router import TopologyRouter, PlacementDecision, PlacementTier
from tempo.qos_mapper import QoSMapper, TC, TrafficClass

__version__ = "0.3.0"
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
]
