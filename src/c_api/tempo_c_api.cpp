// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/c_api/tempo_c_api.cpp
//
// Clean C API exposed by libtempo.so so Python (ctypes) and other languages
// can use TEMPO without linking against C++ headers.

#include "../attention_monitor/monitor.hpp"
#include "../spike_absorber/absorber.hpp"
#include "../pacing_daemon/pacing_daemon.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

// ---------------------------------------------------------------------------
// Engine: bundles monitor + absorber + daemon
// ---------------------------------------------------------------------------

struct TEMPOEngine {
    tempo::AttentionPhaseMonitor monitor;
    tempo::SpikeAbsorber         absorber;
    tempo::PacingDaemon          daemon;

    TEMPOEngine(tempo::PacingConfig cfg)
        : daemon(monitor, absorber, std::move(cfg))
    {
        monitor.start();
        daemon.start();
    }

    ~TEMPOEngine() {
        daemon.stop();
        monitor.stop();
    }
};

// ---------------------------------------------------------------------------
// C API (extern "C" — no name mangling)
// ---------------------------------------------------------------------------

extern "C" {

/// Create a TEMPO engine.
/// Returns an opaque handle or NULL on failure.
void* tempo_create_engine(
    uint64_t    rate_bytes_per_sec,
    uint64_t    burst_bytes,
    uint64_t    ffn_wait_us,
    const char* staging_dir,
    const char* lustre_dir,
    int         strict_gate,
    int         verbose)
{
    try {
        tempo::PacingConfig cfg;
        cfg.rate_bytes_per_sec = rate_bytes_per_sec;
        cfg.burst_bytes        = burst_bytes;
        cfg.ffn_wait_us        = ffn_wait_us;
        if (staging_dir) cfg.staging_dir = staging_dir;
        if (lustre_dir)  cfg.lustre_dir  = lustre_dir;
        cfg.strict_gate = (strict_gate != 0);
        cfg.verbose     = (verbose     != 0);

        return static_cast<void*>(new TEMPOEngine(std::move(cfg)));
    } catch (...) {
        return nullptr;
    }
}

/// Submit a KV block for harmonious pacing. O(1) wait-free.
/// Returns 1 on success, 0 if the ring buffer is full (back-pressure signal).
int tempo_absorb(
    void*    handle,
    uint64_t block_id,
    void*    host_ptr,
    size_t   size_bytes)
{
    if (!handle || !host_ptr || size_bytes == 0) return 0;

    tempo::KVBlock blk{};
    blk.block_id   = block_id;
    blk.host_ptr   = host_ptr;
    blk.size_bytes = size_bytes;

    auto* engine = static_cast<TEMPOEngine*>(handle);
    return engine->absorber.absorb(blk) ? 1 : 0;
}

/// Destroy a TEMPO engine and wait for all in-flight I/O to complete.
void tempo_destroy_engine(void* handle) {
    delete static_cast<TEMPOEngine*>(handle);
}

/// Query metrics. All out-parameters are optional (pass NULL to skip).
void tempo_get_metrics(
    void*     handle,
    uint64_t* out_bytes_flushed,
    uint64_t* out_blocks_flushed,
    uint64_t* out_attention_pauses,
    uint64_t* out_ring_size_approx)
{
    if (!handle) return;
    auto* e = static_cast<TEMPOEngine*>(handle);
    if (out_bytes_flushed)      *out_bytes_flushed      = e->daemon.bytes_flushed();
    if (out_blocks_flushed)     *out_blocks_flushed     = e->daemon.blocks_flushed();
    if (out_attention_pauses)   *out_attention_pauses   = e->daemon.attention_pauses();
    if (out_ring_size_approx)   *out_ring_size_approx   = e->absorber.size_approx();
}

} // extern "C"
