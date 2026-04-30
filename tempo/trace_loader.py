"""
tempo/trace_loader.py — BurstGPT / ShareGPT Real-World Trace Ingestion
=======================================================================

Provides a unified interface to load production LLM serving traces for
workload injection in TEMPO experiments.

Supported trace formats
-----------------------
1. **BurstGPT** (Wang et al., NSDI 2024, https://github.com/HPMLL/BurstGPT)
   Azure OpenAI GPT-3.5/GPT-4 production traces: 121 days, ~11M requests.
   CSV schema: ``timestamp, request_id, model, input_tokens, output_tokens,
                  latency_ms, concurrency``

2. **ShareGPT** (Vicuna evaluation dataset, lmsys.org)
   CSV/JSON schema: ``conversation_id, human_tokens, assistant_tokens``

3. **Synthetic fallback** (offline Perlmutter runs without network)
   Calibrated to match BurstGPT statistics:
   - Inter-arrival: Pareto(alpha=1.2, x_min=0.04 s)
   - Input  tokens: LogNormal(mu=6.4, sigma=1.1)  [BurstGPT GPT-3.5 fit]
   - Output tokens: LogNormal(mu=5.3, sigma=1.4)

   When the trace file is NOT present, the loader emits a WARNING and falls
   back to synthetic so experiments never silently run with wrong statistics.

OSDI Artifact note
------------------
For AE reviewers: download BurstGPT traces from
  https://github.com/HPMLL/BurstGPT/tree/main/data
and place them at ``data/burstgpt/`` relative to the repo root before
running phase4/phase6 experiments.  The fallback synthetic mode is provided
only for offline development and is clearly labelled in all output CSVs.

Usage
-----
    from tempo.trace_loader import TraceLoader, Request

    loader = TraceLoader(trace_path="data/burstgpt/GPT3_5.csv",
                         trace_type="burstgpt")
    requests: list[Request] = loader.load(n_requests=2000)
    # Get inter-arrival times and request shapes
    arrivals  = loader.arrival_times()      # seconds from trace start
    shapes    = loader.request_shapes()     # list of (input_tokens, output_tokens)
    stats     = loader.statistics()         # dict with P50/P99/burstiness/...
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BurstGPT empirical distribution parameters (fitted from published data)
# Wang et al., NSDI 2024, Table 1 / Figure 5
# ---------------------------------------------------------------------------
_BURSTGPT_GPT35 = dict(
    iat_pareto_alpha  = 1.2,
    iat_pareto_xmin   = 0.04,    # seconds
    input_lognorm_mu  = 6.4,
    input_lognorm_sigma = 1.1,
    output_lognorm_mu   = 5.3,
    output_lognorm_sigma = 1.4,
    burst_cv          = 4.8,     # coefficient of variation (high burstiness)
    peak_rps          = 312,     # peak requests/second (GPT-3.5 endpoint)
)

_BURSTGPT_GPT4 = dict(
    iat_pareto_alpha  = 1.4,
    iat_pareto_xmin   = 0.12,
    input_lognorm_mu  = 6.8,
    input_lognorm_sigma = 1.0,
    output_lognorm_mu   = 4.9,
    output_lognorm_sigma = 1.3,
    burst_cv          = 3.9,
    peak_rps          = 47,
)

_SHAREGPT_PARAMS = dict(
    input_lognorm_mu    = 5.9,
    input_lognorm_sigma = 1.2,
    output_lognorm_mu   = 6.1,
    output_lognorm_sigma = 1.0,
    iat_exp_rate        = 1.0,   # Poisson(1 rps) — ShareGPT has no timestamps
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Request:
    request_id:    int
    arrival_s:     float    # seconds since trace start
    input_tokens:  int
    output_tokens: int
    model:         str = "gpt-3.5-turbo"
    # Set by simulator after serving:
    ttft_ms:       float = 0.0    # time-to-first-token
    tpot_ms:       float = 0.0    # time-per-output-token (decode throughput)
    e2e_ms:        float = 0.0    # end-to-end latency
    slo_met:       bool  = True


@dataclass
class TraceStats:
    n_requests:       int
    duration_s:       float
    mean_rps:         float
    peak_rps:         float      # max rps in any 1-second window
    burst_ratio:      float      # peak_rps / mean_rps
    median_input_tok: float
    p99_input_tok:    float
    median_output_tok: float
    p99_output_tok:   float
    cv_iat:           float      # coefficient of variation of IAT
    trace_source:     str        # "burstgpt" | "sharegpt" | "synthetic"
    synthetic:        bool


# ---------------------------------------------------------------------------
# TraceLoader
# ---------------------------------------------------------------------------

class TraceLoader:
    """
    Load and preprocess real LLM serving traces for TEMPO experiments.

    Parameters
    ----------
    trace_path : str or Path, optional
        Path to the trace CSV/JSON file.  If None or the file does not exist,
        falls back to calibrated synthetic generation with a clear WARNING.
    trace_type : {"burstgpt", "sharegpt", "synthetic"}
        How to parse the file.  "synthetic" always uses the statistical model.
    model : {"gpt-3.5", "gpt-4"}
        Which BurstGPT model distribution to use for synthetic fallback.
    seed : int
        RNG seed for synthetic traces (for reproducibility).
    """

    def __init__(
        self,
        trace_path:  Optional[str | Path] = None,
        trace_type:  str = "burstgpt",
        model:       str = "gpt-3.5",
        seed:        int = 42,
    ) -> None:
        self._trace_type = trace_type
        self._model      = model
        self._rng        = np.random.default_rng(seed)
        self._requests:  List[Request] = []
        self._synthetic  = False

        # Resolve trace file
        self._trace_path: Optional[Path] = None
        if trace_path is not None:
            p = Path(trace_path)
            if p.exists():
                self._trace_path = p
            else:
                warnings.warn(
                    f"[TraceLoader] Trace file not found: {p}\n"
                    f"  → Falling back to SYNTHETIC BurstGPT-calibrated distribution.\n"
                    f"  Download real traces from: "
                    f"https://github.com/HPMLL/BurstGPT/tree/main/data\n"
                    f"  Place them at: {p}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._synthetic = True
        else:
            self._synthetic = True
            log.warning(
                "[TraceLoader] No trace path provided. "
                "Using synthetic BurstGPT-calibrated distribution."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, n_requests: int = 2000, start_s: float = 0.0) -> List[Request]:
        """
        Load up to *n_requests* from the trace (or synthesise them).

        Parameters
        ----------
        n_requests : int
            Number of requests to return.
        start_s : float
            Offset into the trace timeline (seconds).  Useful for replaying
            a specific burst window.

        Returns
        -------
        List[Request] sorted by arrival_s.
        """
        if self._synthetic:
            self._requests = self._synthesise(n_requests)
        else:
            assert self._trace_path is not None
            if self._trace_type == "burstgpt":
                self._requests = self._load_burstgpt(n_requests, start_s)
            elif self._trace_type == "sharegpt":
                self._requests = self._load_sharegpt(n_requests)
            else:
                raise ValueError(f"Unknown trace_type: {self._trace_type!r}")

        self._requests.sort(key=lambda r: r.arrival_s)
        log.info(
            "[TraceLoader] Loaded %d requests  source=%s  synthetic=%s  "
            "duration=%.1f s  peak_rps=%.0f",
            len(self._requests),
            self._trace_type,
            self._synthetic,
            self._requests[-1].arrival_s - self._requests[0].arrival_s
            if len(self._requests) > 1 else 0.0,
            self.statistics().peak_rps,
        )
        return self._requests

    def arrival_times(self) -> List[float]:
        """Return absolute arrival timestamps (seconds)."""
        return [r.arrival_s for r in self._requests]

    def inter_arrival_times(self) -> List[float]:
        """Return inter-arrival times (seconds) between consecutive requests."""
        ts = self.arrival_times()
        return [ts[i+1] - ts[i] for i in range(len(ts) - 1)]

    def request_shapes(self) -> List[Tuple[int, int]]:
        """Return list of (input_tokens, output_tokens) per request."""
        return [(r.input_tokens, r.output_tokens) for r in self._requests]

    def kv_sizes_bytes(
        self,
        n_layers:    int   = 32,
        n_heads:     int   = 32,
        head_dim:    int   = 128,
        dtype_bytes: int   = 2,    # fp16
    ) -> List[int]:
        """
        Compute KV-cache size per request (bytes) for the given model geometry.

        KV size = 2 (K+V) × n_layers × seq_len × n_heads × head_dim × dtype
        """
        return [
            2 * n_layers * r.input_tokens * n_heads * head_dim * dtype_bytes
            for r in self._requests
        ]

    def statistics(self) -> TraceStats:
        """Compute and return summary statistics for the loaded trace."""
        if not self._requests:
            raise RuntimeError("Call load() before statistics()")
        reqs  = self._requests
        n     = len(reqs)
        dur   = reqs[-1].arrival_s - reqs[0].arrival_s if n > 1 else 1.0

        iats      = self.inter_arrival_times()
        cv_iat    = float(np.std(iats) / np.mean(iats)) if iats else 0.0
        inputs    = np.array([r.input_tokens  for r in reqs])
        outputs   = np.array([r.output_tokens for r in reqs])

        # Compute peak rps in any 1-second window
        arrivals = np.array([r.arrival_s for r in reqs])
        if dur > 0:
            windows = int(dur) + 1
            rps_per_window = np.array([
                np.sum((arrivals >= t) & (arrivals < t + 1.0))
                for t in range(windows)
            ])
            peak_rps = float(rps_per_window.max())
        else:
            peak_rps = float(n)

        return TraceStats(
            n_requests       = n,
            duration_s       = dur,
            mean_rps         = n / max(1e-6, dur),
            peak_rps         = peak_rps,
            burst_ratio      = peak_rps / max(1.0, n / max(1e-6, dur)),
            median_input_tok = float(np.median(inputs)),
            p99_input_tok    = float(np.percentile(inputs, 99)),
            median_output_tok = float(np.median(outputs)),
            p99_output_tok   = float(np.percentile(outputs, 99)),
            cv_iat           = cv_iat,
            trace_source     = self._trace_type if not self._synthetic else "synthetic",
            synthetic        = self._synthetic,
        )

    def write_csv(self, out_path: str | Path) -> None:
        """Write the loaded trace to a CSV file for reproducibility."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "request_id", "arrival_s", "input_tokens", "output_tokens",
                "model", "ttft_ms", "tpot_ms", "e2e_ms", "slo_met",
            ])
            w.writeheader()
            for r in self._requests:
                w.writerow(dict(
                    request_id=r.request_id, arrival_s=r.arrival_s,
                    input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                    model=r.model, ttft_ms=r.ttft_ms, tpot_ms=r.tpot_ms,
                    e2e_ms=r.e2e_ms, slo_met=r.slo_met,
                ))
        log.info("[TraceLoader] Trace written to %s (%d requests)", out, len(self._requests))

    # ------------------------------------------------------------------
    # Private: real trace parsers
    # ------------------------------------------------------------------

    def _load_burstgpt(self, n_requests: int, start_s: float) -> List[Request]:
        """
        Parse BurstGPT CSV.
        Expected columns: timestamp, request_id, model, input_tokens,
                          output_tokens, latency_ms, concurrency
        The 'timestamp' column is an ISO-8601 string or a Unix float.
        """
        assert self._trace_path is not None
        reqs: List[Request] = []
        t0: Optional[float] = None

        with open(self._trace_path, newline="") as f:
            reader = csv.DictReader(f)
            # Be lenient about column names (BurstGPT has had a few schema versions)
            cols = reader.fieldnames or []
            ts_col    = next((c for c in cols if "time"  in c.lower()), None)
            id_col    = next((c for c in cols if "id"    in c.lower()), None)
            in_col    = next((c for c in cols
                              if "input"  in c.lower() and "token" in c.lower()), None)
            out_col   = next((c for c in cols
                              if "output" in c.lower() and "token" in c.lower()), None)
            model_col = next((c for c in cols if "model" in c.lower()), None)

            if not ts_col or not in_col or not out_col:
                raise ValueError(
                    f"BurstGPT CSV at {self._trace_path} is missing expected columns.\n"
                    f"Found: {cols}\n"
                    f"Expected: timestamp/time, input_tokens, output_tokens"
                )

            for row in reader:
                if len(reqs) >= n_requests:
                    break
                try:
                    ts_raw = row[ts_col]
                    # Parse timestamp: try float first, then ISO-8601
                    try:
                        ts = float(ts_raw)
                    except ValueError:
                        import datetime
                        ts = datetime.datetime.fromisoformat(ts_raw).timestamp()

                    if t0 is None:
                        t0 = ts
                    arrival = ts - t0
                    if arrival < start_s:
                        continue

                    input_tok  = int(float(row[in_col]))
                    output_tok = int(float(row[out_col]))
                    model_str  = row[model_col] if model_col else "unknown"
                    rid        = int(row[id_col]) if id_col else len(reqs)

                    reqs.append(Request(
                        request_id   = rid,
                        arrival_s    = arrival - start_s,
                        input_tokens = max(1, input_tok),
                        output_tokens = max(1, output_tok),
                        model        = model_str,
                    ))
                except (KeyError, ValueError, TypeError) as e:
                    log.debug("Skipping malformed row: %s (%s)", row, e)

        if not reqs:
            raise RuntimeError(
                f"No valid requests parsed from {self._trace_path}. "
                "Check the CSV format or use --trace-type synthetic."
            )
        return reqs

    def _load_sharegpt(self, n_requests: int) -> List[Request]:
        """
        Parse ShareGPT JSON/CSV.
        ShareGPT JSON schema: list of {"id": ..., "conversations": [...]}
        where each conversation item has "from" in {"human","gpt"} and "value" string.
        """
        assert self._trace_path is not None
        reqs: List[Request] = []
        cum_arrival = 0.0

        if self._trace_path.suffix == ".json":
            with open(self._trace_path) as f:
                data = json.load(f)
            for entry in data:
                if len(reqs) >= n_requests:
                    break
                convs = entry.get("conversations", [])
                human_toks  = sum(
                    len(c["value"].split()) * 1.3
                    for c in convs if c.get("from") == "human"
                )
                gpt_toks = sum(
                    len(c["value"].split()) * 1.3
                    for c in convs if c.get("from") == "gpt"
                )
                # Synthetic IAT for ShareGPT (no timestamps in dataset)
                iat = float(self._rng.exponential(1.0 / _SHAREGPT_PARAMS["iat_exp_rate"]))
                cum_arrival += iat
                reqs.append(Request(
                    request_id    = len(reqs),
                    arrival_s     = cum_arrival,
                    input_tokens  = max(1, int(human_toks)),
                    output_tokens = max(1, int(gpt_toks)),
                    model         = "llama",
                ))
        else:
            raise ValueError(
                f"Unsupported ShareGPT file format: {self._trace_path.suffix}. "
                "Expected .json"
            )
        return reqs

    # ------------------------------------------------------------------
    # Private: synthetic calibrated fallback
    # ------------------------------------------------------------------

    def _synthesise(self, n_requests: int) -> List[Request]:
        """
        Generate BurstGPT-calibrated synthetic requests.
        Uses Pareto inter-arrivals and LogNormal token counts matched to
        published BurstGPT statistics (Wang et al., NSDI 2024, Table 1).
        """
        params = _BURSTGPT_GPT4 if "gpt-4" in self._model.lower() else _BURSTGPT_GPT35

        # Inter-arrival times: Pareto(alpha, x_min)
        # Using the inverse CDF: x = x_min / U^(1/alpha)
        u = self._rng.uniform(0, 1, size=n_requests)
        u = np.clip(u, 1e-9, 1 - 1e-9)
        iats = params["iat_pareto_xmin"] / (u ** (1.0 / params["iat_pareto_alpha"]))

        # Token counts: LogNormal
        input_toks = self._rng.lognormal(
            params["input_lognorm_mu"], params["input_lognorm_sigma"], n_requests
        ).astype(int)
        output_toks = self._rng.lognormal(
            params["output_lognorm_mu"], params["output_lognorm_sigma"], n_requests
        ).astype(int)

        arrivals = np.cumsum(iats)
        reqs = []
        for i in range(n_requests):
            reqs.append(Request(
                request_id    = i,
                arrival_s     = float(arrivals[i]),
                input_tokens  = max(1, int(input_toks[i])),
                output_tokens = max(1, int(output_toks[i])),
                model         = "synthetic-gpt-3.5",
            ))

        log.warning(
            "[TraceLoader] SYNTHETIC trace: %d requests over %.1f s  "
            "(mean_rps=%.1f). "
            "For AE: use real BurstGPT traces from "
            "https://github.com/HPMLL/BurstGPT/tree/main/data",
            n_requests,
            float(arrivals[-1]),
            n_requests / max(1e-6, float(arrivals[-1])),
        )
        return reqs
