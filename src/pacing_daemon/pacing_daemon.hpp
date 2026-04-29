// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/pacing_daemon/pacing_daemon.hpp

#pragma once

#include "../attention_monitor/monitor.hpp"
#include "../spike_absorber/absorber.hpp"
#include "token_bucket.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace tempo {

// ---------------------------------------------------------------------------
// PacingConfig  — tunables exposed via environment variables / YAML
// ---------------------------------------------------------------------------

struct PacingConfig {
    // I/O target
    std::string staging_dir   = "/tmp/tempo_stage";  ///< Local NVMe staging path
    std::string lustre_dir    = "";                  ///< Lustre flush target (from env PSCRATCH)

    // Rate limiting
    uint64_t rate_bytes_per_sec = 5ULL * 1024 * 1024 * 1024;  ///< 5 GB/s (≈20% of Slingshot NIC)
    uint64_t burst_bytes        = 256ULL * 1024 * 1024;        ///< 256 MB burst

    // Phase-gating
    uint64_t ffn_wait_us = 200;   ///< Max wait for FFN window before giving up (µs)
    bool     strict_gate = true;  ///< If false, flush even during ATTENTION if ring > threshold

    // io_uring
    uint32_t uring_depth = 128;   ///< io_uring SQ/CQ depth

    // Miscellaneous
    bool     verbose     = false;

    /// Load from environment variables (TEMPO_RATE_GBPS, TEMPO_BURST_MB, etc.)
    static PacingConfig from_env();
};

// ---------------------------------------------------------------------------
// PacingDaemon — core harmonious flush loop
//
// Lifecycle:
//   1. Construct with references to AttentionPhaseMonitor and SpikeAbsorber.
//   2. Call start() to spawn the background daemon thread.
//   3. Call stop()  on shutdown; waits for in-flight io_uring completions.
//
// Hot path (inside daemon thread):
//   while (running) {
//       wait for FFN window (attention_monitor.wait_for_ffn)
//       drain one block from SpikeAbsorber
//       consume token_bucket credits
//       submit io_uring write to Lustre
//       collect completions
//   }
// ---------------------------------------------------------------------------

class PacingDaemon {
public:
    PacingDaemon(AttentionPhaseMonitor& monitor,
                 SpikeAbsorber&         absorber,
                 PacingConfig           cfg = PacingConfig{});
    ~PacingDaemon();

    PacingDaemon(const PacingDaemon&)            = delete;
    PacingDaemon& operator=(const PacingDaemon&) = delete;

    void start();
    void stop() noexcept;

    // Metrics
    uint64_t bytes_flushed()    const noexcept { return bytes_flushed_.load(std::memory_order_relaxed); }
    uint64_t blocks_flushed()   const noexcept { return blocks_flushed_.load(std::memory_order_relaxed); }
    uint64_t attention_pauses() const noexcept { return attention_pauses_.load(std::memory_order_relaxed); }

private:
    void run_loop();       ///< Main daemon thread entry point
    int  open_lustre();    ///< Open (or create) the Lustre target file, O_DIRECT
    void flush_block(const KVBlock& blk, int lustre_fd);

    AttentionPhaseMonitor& monitor_;
    SpikeAbsorber&         absorber_;
    PacingConfig           cfg_;
    TokenBucket            bucket_;

    std::thread            thread_;
    std::atomic<bool>      running_{false};

    // io_uring handle (opaque pointer to avoid pulling in liburing headers here)
    void* uring_ptr_{nullptr};

    // Metrics
    std::atomic<uint64_t> bytes_flushed_    {0};
    std::atomic<uint64_t> blocks_flushed_   {0};
    std::atomic<uint64_t> attention_pauses_ {0};
};

} // namespace tempo
