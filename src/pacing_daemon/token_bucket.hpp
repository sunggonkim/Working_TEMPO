// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/pacing_daemon/token_bucket.hpp
//
// Non-blocking token bucket for byte-rate limiting.
//
// Prevents TEMPO from monopolising the Slingshot NIC even during long FFN
// windows (e.g., very wide MLP layers in 70B models). The rate cap ensures
// foreground NCCL AllReduce always has sufficient NIC bandwidth headroom.
//
// Thread safety: try_consume() uses relaxed atomics and is safe to call from
// a single consumer thread (PacingDaemon). refill() is idempotent and cheap.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace tempo {

class TokenBucket {
public:
    /// @param rate_bytes_per_sec  Steady-state flush ceiling (e.g. 5 GB/s on
    ///                            Slingshot 11 with 200 Gbps ≈ 25 GB/s total).
    ///                            Set to ~20% of NIC BW to leave 80% for NCCL.
    /// @param burst_bytes         Maximum burst (e.g. 256 MB).
    ///                            Should be ≥ one KV block flush unit.
    TokenBucket(uint64_t rate_bytes_per_sec,
                uint64_t burst_bytes) noexcept
        : rate_bpns_(static_cast<double>(rate_bytes_per_sec) / 1e9)
        , burst_     (static_cast<double>(burst_bytes))
        , tokens_    (static_cast<double>(burst_bytes))
        , last_ns_   (now_ns())
    {}

    /// Try to consume `want` bytes.
    /// Returns the number of bytes actually granted (may be 0 if empty,
    /// or < want if partially filled). Non-blocking.
    uint64_t try_consume(uint64_t want) noexcept {
        refill();
        double w     = static_cast<double>(want);
        double avail = tokens_.load(std::memory_order_relaxed);
        if (avail <= 0.0) return 0;
        double grant = (w <= avail) ? w : avail;
        tokens_.store(avail - grant, std::memory_order_relaxed);
        return static_cast<uint64_t>(grant);
    }

    /// Approximate tokens available (for metrics / logging).
    double available() const noexcept {
        return tokens_.load(std::memory_order_relaxed);
    }

    /// Hard-reset the bucket to full (e.g. after a long idle period).
    void reset() noexcept {
        tokens_.store(burst_, std::memory_order_relaxed);
        last_ns_.store(now_ns(), std::memory_order_relaxed);
    }

private:
    static uint64_t now_ns() noexcept {
        return static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void refill() noexcept {
        uint64_t now  = now_ns();
        uint64_t prev = last_ns_.load(std::memory_order_relaxed);
        if (now <= prev) return;  // clock skew guard

        double elapsed_ns = static_cast<double>(now - prev);
        double added      = elapsed_ns * rate_bpns_;
        double cur        = tokens_.load(std::memory_order_relaxed);
        double next       = std::min(cur + added, burst_);

        tokens_.store(next,  std::memory_order_relaxed);
        last_ns_.store(now,  std::memory_order_relaxed);
    }

    double              rate_bpns_;       ///< bytes per nanosecond
    double              burst_;           ///< max token capacity (bytes)
    std::atomic<double> tokens_;          ///< current token level
    std::atomic<uint64_t> last_ns_;       ///< last refill timestamp (ns)
};

} // namespace tempo
