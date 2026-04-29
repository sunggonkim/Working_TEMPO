// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/attention_monitor/monitor.hpp
//
// GPU-phase monitor based on CUPTI callback API.
//
// Problem: PCIe Root Complex is shared between HBM DMA (attention) and
// NVMe DMA (KV eviction). During attention, every PCIe byte of KV flush
// competes with HBM reads → decode latency spikes.
//
// Solution: Classify each CUDA kernel launch as ATTENTION or FFN.
// PacingDaemon reads the current phase atomically (O(1)) and suppresses
// Lustre/NVMe writes during ATTENTION windows.
//
// Supported backends:
//   vLLM v1  : paged_attention_v2, flash_fwd_kernel
//   SGLang   : context_attention_fwd, token_attention_fwd
//   FA2/FA3  : flash_fwd_kernel, FlashAttn (triton)
//   SDPA     : scaled_dot_product_fused_kernel

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string_view>

// CUPTI forward declaration to avoid pulling in cupti.h everywhere.
struct CUpti_SubscriberHandle_st;
using CUpti_Subscriber = CUpti_SubscriberHandle_st*;

namespace tempo {

// ---------------------------------------------------------------------------
// Phase enum
// ---------------------------------------------------------------------------

enum class Phase : uint8_t {
    UNKNOWN   = 0,
    ATTENTION = 1,  ///< HBM-bandwidth-bound, PCIe-sensitive  → NO background I/O
    FFN       = 2,  ///< GEMM/CUTLASS compute-bound, PCIe-tolerant → TEMPO flushes here
    OTHER     = 3,  ///< LayerNorm, embedding, misc
};

constexpr std::string_view phase_name(Phase p) noexcept {
    switch (p) {
        case Phase::ATTENTION: return "ATTENTION";
        case Phase::FFN:       return "FFN";
        case Phase::OTHER:     return "OTHER";
        default:               return "UNKNOWN";
    }
}

/// Classify a CUDA kernel name (demangled) into a Phase.
/// Called from the CUPTI callback — must be async-signal-safe and fast.
Phase classify_kernel(const char* mangled_name) noexcept;

// ---------------------------------------------------------------------------
// AttentionPhaseMonitor
// ---------------------------------------------------------------------------

/// Thread-safe GPU phase monitor.
///
/// Lifecycle:
///   1. Construct once per process.
///   2. Call start() after cudaInit — registers CUPTI subscriber.
///   3. Hot path: get() — single atomic load, zero overhead.
///   4. Blocking path: wait_for_ffn(timeout_us) — used by PacingDaemon.
///   5. Call stop() on shutdown.
///
/// Thread safety:
///   - get() is safe to call from any thread at any time.
///   - start()/stop() must not be called concurrently.
class AttentionPhaseMonitor {
public:
    AttentionPhaseMonitor()  = default;
    ~AttentionPhaseMonitor() { stop(); }

    AttentionPhaseMonitor(const AttentionPhaseMonitor&)            = delete;
    AttentionPhaseMonitor& operator=(const AttentionPhaseMonitor&) = delete;

    // --- Control ---

    /// Register CUPTI callbacks. Must be called once, after CUDA init.
    /// Throws std::runtime_error on CUPTI failure.
    void start();

    /// Deregister CUPTI callbacks. Idempotent.
    void stop() noexcept;

    // --- Hot-path accessors ---

    /// Current GPU phase. O(1) atomic load. Safe from any thread.
    [[nodiscard]] Phase get() const noexcept {
        return phase_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool is_attention() const noexcept {
        return get() == Phase::ATTENTION;
    }

    [[nodiscard]] bool is_ffn() const noexcept {
        return get() == Phase::FFN;
    }

    // --- Blocking interface (for PacingDaemon) ---

    /// Block until phase == FFN or timeout_us microseconds elapse.
    /// Returns true if an FFN window was observed before timeout.
    bool wait_for_ffn(uint64_t timeout_us) noexcept;

    // --- Metrics ---

    uint64_t attention_count() const noexcept {
        return attn_count_.load(std::memory_order_relaxed);
    }
    uint64_t ffn_count() const noexcept {
        return ffn_count_.load(std::memory_order_relaxed);
    }

    // --- Internal: called by CUPTI callback ---
    void _set_phase(Phase p) noexcept;

private:
    std::atomic<Phase>    phase_     {Phase::UNKNOWN};
    std::atomic<uint64_t> attn_count_{0};
    std::atomic<uint64_t> ffn_count_ {0};

    mutable std::mutex      cv_mtx_;
    std::condition_variable cv_;

    CUpti_Subscriber subscriber_{nullptr};
    bool             started_   {false};
};

} // namespace tempo
