"""Frozen, evidence-backed workload policy for the TEMPO P/D controller."""

from __future__ import annotations

from dataclasses import dataclass

from tempo.pd_regime_controller import PairArrivalRegimeController


@dataclass(frozen=True)
class FrozenPDPolicy:
    high_pair_interval_ns: int = 70_000_000
    mid_pair_interval_ns: int = 110_000_000
    calibration_requests: int = 3

    def direct_local(self, prompt_tokens: int, output_tokens: int) -> bool:
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive int")
        if output_tokens in (16, 128, 256):
            if prompt_tokens <= 4096:
                return True
            raise ValueError(
                f"output{output_tokens} is GPU-validated only for prompts up to "
                "4096 tokens")
        return False

    def force_local(self, prompt_tokens: int, output_tokens: int) -> bool:
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive int")
        self.high_local_credit(output_tokens)
        return output_tokens == 64 and prompt_tokens <= 512

    def validate_controller_workload(self, prompt_tokens: int,
                                     output_tokens: int) -> None:
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive int")
        self.high_local_credit(output_tokens)
        if prompt_tokens > 2048:
            raise ValueError(
                "output32/output64 controllers are GPU-validated only for "
                "prompts up to 2048 tokens")

    def high_pair_interval(self, output_tokens: int) -> int:
        if output_tokens == 32:
            return 58_000_000
        if output_tokens == 64:
            return self.high_pair_interval_ns
        return self.high_local_credit(output_tokens)

    def high_local_credit(self, output_tokens: int) -> int:
        if output_tokens == 32:
            return 8
        if output_tokens == 64:
            return 9
        raise ValueError(
            "only the GPU-validated 32- and 64-output-token policies are frozen"
        )

    def controller(self, output_tokens: int) -> PairArrivalRegimeController:
        return PairArrivalRegimeController(
            high_pair_interval_ns=self.high_pair_interval(output_tokens),
            mid_pair_interval_ns=self.mid_pair_interval_ns,
            calibration_requests=self.calibration_requests,
            high_local_inflight_cap=self.high_local_credit(output_tokens),
        )
