// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/spike_absorber/absorber.hpp
//
// Lock-free MPSC (multi-producer, single-consumer) ring buffer.
//
// The absorb() call is on the critical path of vLLM's KV eviction:
//   vLLM KVCacheManager::evict() → TEMPOStorageBackend::put() → absorb()
//
// Requirements:
//   - O(1) wait-free: must never block the GPU scheduling thread
//   - MPSC: multiple CUDA streams may evict concurrently (producers)
//   - Single consumer: PacingDaemon drain loop (one thread)
//
// Design: Dmitry Vyukov's MPSC bounded queue
//   (https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue)
//   adapted for MPSC. Each slot has a sequence number; producers CAS on
//   enqueue_pos_ and readers check seq == dequeue_pos+1.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tempo {

// ---------------------------------------------------------------------------
// KVBlock: metadata for one KV cache block to be staged
// ---------------------------------------------------------------------------

struct alignas(64) KVBlock {
    uint64_t block_id;      ///< Content-addressed block ID (SHA-256 prefix or page index)
    uint32_t layer_idx;     ///< Transformer layer (for tiering heuristics)
    uint32_t num_tokens;    ///< Number of tokens in this block
    void*    host_ptr;      ///< Pointer to pinned CPU staging memory (already copied from HBM)
    size_t   size_bytes;    ///< Payload size in bytes
    int      dst_tier;      ///< Target tier: 0=NVMe, 1=DRAM, 2=Lustre
    int      _pad;          ///< Alignment padding
};
static_assert(sizeof(KVBlock) <= 64, "KVBlock must fit in one cache line");

// ---------------------------------------------------------------------------
// SpikeAbsorber: lock-free MPSC ring buffer
// ---------------------------------------------------------------------------

class SpikeAbsorber {
public:
    // 16 K slots × ~64 B metadata = 1 MB ring  →  headroom for a 70B model
    // checkpoint spike (typically < 8 K KV pages per layer per node)
    static constexpr uint32_t CAPACITY = 1u << 14;  // must be power of 2
    static constexpr uint32_t MASK     = CAPACITY - 1;

    SpikeAbsorber();
    ~SpikeAbsorber();

    SpikeAbsorber(const SpikeAbsorber&)            = delete;
    SpikeAbsorber& operator=(const SpikeAbsorber&) = delete;

    // -----------------------------------------------------------------------
    // Producer API  (called from vLLM eviction path — must be wait-free)
    // -----------------------------------------------------------------------

    /// Enqueue a KV block for deferred flush.
    /// Returns false if the ring is full (caller must apply back-pressure
    /// or fall back to synchronous flush).
    ///
    /// Complexity: O(1) amortized wait-free.
    bool absorb(const KVBlock& blk) noexcept;

    // -----------------------------------------------------------------------
    // Consumer API  (called exclusively from PacingDaemon thread)
    // -----------------------------------------------------------------------

    /// Dequeue one KV block. Returns false if ring is empty.
    bool drain(KVBlock& out) noexcept;

    /// eventfd(2) file descriptor.
    /// PacingDaemon polls/epoll-waits on this to be notified of new blocks
    /// without spinning. absorb() writes 1 to this fd.
    int eventfd() const noexcept { return efd_; }

    // -----------------------------------------------------------------------
    // Metrics (non-linearizable approximations)
    // -----------------------------------------------------------------------

    size_t size_approx() const noexcept;
    uint64_t absorbed_total()  const noexcept { return absorbed_total_.load(std::memory_order_relaxed); }
    uint64_t drained_total()   const noexcept { return drained_total_.load(std::memory_order_relaxed); }
    uint64_t overflow_drops()  const noexcept { return overflow_drops_.load(std::memory_order_relaxed); }

private:
    struct Slot {
        std::atomic<uint64_t> seq;
        KVBlock               blk;
    };

    // Producer and consumer indices on separate cache lines to avoid
    // false sharing (the most common perf pitfall in ring buffers).
    alignas(64) std::atomic<uint64_t> enqueue_pos_{0};
    alignas(64) std::atomic<uint64_t> dequeue_pos_{0};

    // The ring itself — 1 MB for CAPACITY=16K
    alignas(64) Slot ring_[CAPACITY];

    // Metrics
    alignas(64) std::atomic<uint64_t> absorbed_total_{0};
    std::atomic<uint64_t>             drained_total_ {0};
    std::atomic<uint64_t>             overflow_drops_{0};

    int efd_{-1};  ///< eventfd for PacingDaemon wakeup
};

} // namespace tempo
