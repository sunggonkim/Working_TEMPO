"""Small PyTorch-native KV movement adapter for the TEMPO-RD contract.

This is a framework-neutral reference backend, not a vLLM/SGLang replacement.
It moves real ``torch.Tensor`` buffers between an HBM-side cache and an
endpoint-side cache while delegating version identity, exact completion, route
overlap, residuals, and deadline admission to :mod:`tempo.kv_flow` and the
shared :class:`~tempo.domain_admission.DomainAdmissionController`.

The endpoint store is an in-process CPU tensor in this adapter.  A production
backend replaces only the endpoint copy with host/NIC/storage operations and
must retain the same request/version/completion contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tempo.domain_admission import DomainAdmissionController
from tempo.kv_flow import (
    KVAdmissionDecision,
    KVFlowLedger,
    KVOperation,
    KVTransferRequest,
    KVVersion,
)
from tempo.resource_domain import ResourceDomain


@dataclass(frozen=True)
class TorchKVCompletion:
    request_id: str
    version: KVVersion
    destination: ResourceDomain
    bytes: int
    content_digest: str


class TorchKVBackend:
    """Reference tensor backend with the same lifecycle as a native adapter."""

    def __init__(self, ledger: KVFlowLedger | None = None) -> None:
        self.ledger = ledger or KVFlowLedger()
        self._hbm: dict[str, tuple[KVVersion, Any]] = {}
        self._endpoint: dict[str, tuple[KVVersion, Any]] = {}
        self._hbm_devices: dict[str, Any] = {}
        self._pending: dict[str, tuple[KVTransferRequest, Any]] = {}

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("TorchKVBackend requires PyTorch") from exc
        return torch

    @classmethod
    def _tensor_bytes(cls, tensor: Any) -> tuple[Any, bytes]:
        torch = cls._torch()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("KV payload must be a torch.Tensor")
        if tensor.layout is not torch.strided:
            raise ValueError("KV payload must use strided tensor layout")
        # Canonicalize before hashing so non-contiguous views cannot produce a
        # version whose bytes differ between HBM and endpoint copies.
        cpu = tensor.detach().contiguous().to(device="cpu")
        raw = cpu.view(torch.uint8).numpy().tobytes()
        header = f"dtype={cpu.dtype};shape={tuple(cpu.shape)};".encode("utf-8")
        return cpu, header + raw

    @classmethod
    def _tensor_nbytes(cls, tensor: Any) -> int:
        cls._torch()
        value = int(tensor.numel()) * int(tensor.element_size())
        if value <= 0:
            raise ValueError("KV payload must contain positive bytes")
        return value

    @classmethod
    def _version_for_tensor(cls, session_id: str, sequence: int, tensor: Any) -> KVVersion:
        _cpu, identity_bytes = cls._tensor_bytes(tensor)
        return KVVersion.from_bytes(session_id, sequence, identity_bytes)

    def publish_hbm(self, session_id: str, sequence: int, tensor: Any) -> KVVersion:
        """Publish a new HBM KV version and return its exact identity."""

        cpu, identity_bytes = self._tensor_bytes(tensor)
        version = KVVersion.from_bytes(session_id, sequence, identity_bytes)
        self.ledger.publish(version)
        self._hbm[session_id] = (version, tensor.detach().contiguous().clone())
        self._hbm_devices[session_id] = tensor.device
        return version

    def seed_endpoint(self, version: KVVersion, tensor: Any) -> None:
        """Install an exact version at the endpoint side for a prefetch test."""

        current = self.ledger.published.get(version.session_id)
        if current != version:
            raise ValueError("endpoint seed is stale or unpublished")
        _cpu, identity_bytes = self._tensor_bytes(tensor)
        expected = KVVersion.from_bytes(version.session_id, version.sequence, identity_bytes)
        if expected != version:
            raise ValueError("endpoint seed bytes do not match KV version")
        self._endpoint[version.session_id] = (version, tensor.detach().contiguous().to(device="cpu"))

    def make_request(
        self,
        *,
        request_id: str,
        session_id: str,
        operation: KVOperation,
        route: tuple[ResourceDomain, ...],
        deadline_ns: int,
        max_residual_bytes: int = 0,
        tail_budget_ns: int = 0,
    ) -> KVTransferRequest:
        """Build an exact-byte request from the currently published buffer."""

        if not isinstance(operation, KVOperation):
            raise TypeError("operation must be a KVOperation")
        if not route:
            raise ValueError("route must be non-empty")
        if operation is KVOperation.PREFETCH and (
            route[0] is not ResourceDomain.PERSISTENT_ENDPOINT
            or route[-1] is not ResourceDomain.GPU_LOCAL
        ):
            raise ValueError("prefetch route must run from persistent endpoint to GPU local")
        if operation is KVOperation.EVICT and (
            route[0] is not ResourceDomain.GPU_LOCAL
            or route[-1] is not ResourceDomain.PERSISTENT_ENDPOINT
        ):
            raise ValueError("evict route must run from GPU local to persistent endpoint")
        if operation is KVOperation.PREFETCH:
            source = self._endpoint.get(session_id)
            if source is None:
                raise ValueError("no endpoint KV version is available for prefetch")
            version, tensor = source
        else:
            source = self._hbm.get(session_id)
            if source is None:
                raise ValueError("no HBM KV version is available for transfer")
            version, tensor = source
        return KVTransferRequest(
            request_id=request_id,
            version=version,
            operation=operation,
            bytes=self._tensor_nbytes(tensor),
            source=route[0],
            destination=route[-1],
            route=route,
            deadline_ns=deadline_ns,
            max_residual_bytes=max_residual_bytes,
            tail_budget_ns=tail_budget_ns,
        )

    def admit(
        self,
        request: KVTransferRequest,
        controller: DomainAdmissionController,
        *,
        now_ns: int,
        foreground_domains: tuple[ResourceDomain, ...] | None = None,
        control_overhead_ns: int = 0,
    ) -> KVAdmissionDecision:
        """Admit a real tensor transfer through the shared domain controller."""

        if request.operation is KVOperation.PREFETCH:
            source = self._endpoint.get(request.version.session_id)
        else:
            source = self._hbm.get(request.version.session_id)
        if source is None or source[0] != request.version:
            raise ValueError("request source does not match the published KV version")
        if self._tensor_nbytes(source[1]) != request.bytes:
            raise ValueError("request bytes do not match the source tensor")
        decision = self.ledger.admit_via_domain_controller(
            request,
            controller,
            now_ns=now_ns,
            foreground_domains=foreground_domains,
            control_overhead_ns=control_overhead_ns,
        )
        if decision.status == "admitted":
            self._pending[request.request_id] = (request, source[1].detach().contiguous().clone())
        return decision

    def complete(
        self,
        request_id: str,
        completed_tensor: Any | None = None,
    ) -> TorchKVCompletion:
        """Commit an exact tensor and release the controller reservation."""

        pending = self._pending.get(request_id)
        if pending is None:
            raise ValueError("unknown or already completed KV request")
        request, source_tensor = pending
        tensor = source_tensor if completed_tensor is None else completed_tensor
        if self._tensor_nbytes(tensor) != request.bytes:
            raise ValueError("completed tensor bytes do not match admitted bytes")
        expected = self._version_for_tensor(
            request.version.session_id, request.version.sequence, tensor
        )
        if expected != request.version:
            raise ValueError("completed tensor does not match KV version")
        try:
            self.ledger.complete(request_id, request.bytes)
        except Exception:
            # The ledger releases stale reservations itself; retain no pending
            # transport object after any terminal completion decision.
            self._pending.pop(request_id, None)
            raise
        output = tensor.detach().contiguous()
        if request.operation is KVOperation.PREFETCH:
            device = self._hbm_devices.get(request.version.session_id, "cpu")
            output = output.to(device=device).clone()
            self._hbm[request.version.session_id] = (request.version, output)
        else:
            output = output.to(device="cpu").clone()
            self._endpoint[request.version.session_id] = (request.version, output)
        self._pending.pop(request_id, None)
        return TorchKVCompletion(
            request_id=request_id,
            version=request.version,
            destination=request.destination,
            bytes=request.bytes,
            content_digest=request.version.content_digest,
        )

    def drop_hbm(self, session_id: str, version: KVVersion) -> None:
        """Evict HBM only after the matching endpoint version is present."""

        endpoint = self._endpoint.get(session_id)
        current = self._hbm.get(session_id)
        if endpoint is None or endpoint[0] != version:
            raise ValueError("cannot evict before matching endpoint commit")
        if current is None or current[0] != version:
            raise ValueError("HBM version is not current")
        del self._hbm[session_id]

    def read_hbm(self, session_id: str) -> Any:
        current = self._hbm.get(session_id)
        if current is None:
            raise KeyError(session_id)
        return current[1].clone()

    def read_endpoint(self, session_id: str) -> Any:
        current = self._endpoint.get(session_id)
        if current is None:
            raise KeyError(session_id)
        return current[1].clone()
