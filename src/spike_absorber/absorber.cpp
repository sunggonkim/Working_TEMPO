// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/spike_absorber/absorber.cpp

#include "absorber.hpp"

#include <stdexcept>
#include <sys/eventfd.h>
#include <unistd.h>

namespace tempo {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

SpikeAbsorber::SpikeAbsorber() {
    // Initialise sequence numbers: slot i is "ready for producer" when
    // seq == i (i.e., it has been consumed i/CAPACITY times already).
    for (uint32_t i = 0; i < CAPACITY; ++i)
        ring_[i].seq.store(i, std::memory_order_relaxed);

    // EFD_SEMAPHORE-less eventfd: write(1) increments counter; read() returns
    // and resets it. EFD_NONBLOCK so PacingDaemon can drain without blocking.
    efd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (efd_ < 0)
        throw std::runtime_error("SpikeAbsorber: eventfd() failed");
}

SpikeAbsorber::~SpikeAbsorber() {
    if (efd_ >= 0) ::close(efd_);
}

// ---------------------------------------------------------------------------
// Producer: absorb()
//
// Implementation: Dmitry Vyukov's MPMC queue, single-consumer optimisation.
// Each producer:
//   1. Reads enqueue_pos_ (relaxed load).
//   2. Peeks the target slot's seq number.
//   3. If seq == pos → slot is free → CAS enqueue_pos_ pos→pos+1.
//   4. Write payload, then release-store seq = pos+1.
//
// No ABA problem because sequence numbers are monotonically increasing.
// ---------------------------------------------------------------------------

bool SpikeAbsorber::absorb(const KVBlock& blk) noexcept {
    uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
        Slot&    slot = ring_[pos & MASK];
        uint64_t seq  = slot.seq.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) -
                        static_cast<intptr_t>(pos);

        if (diff == 0) {
            // Slot is free. Try to claim it.
            if (enqueue_pos_.compare_exchange_weak(
                    pos, pos + 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
            {
                slot.blk = blk;
                // Release-store makes the write visible to the consumer.
                slot.seq.store(pos + 1, std::memory_order_release);

                absorbed_total_.fetch_add(1, std::memory_order_relaxed);

                // Wake PacingDaemon. Ignore EAGAIN (counter already nonzero).
                const uint64_t one = 1;
                (void)::write(efd_, &one, sizeof(one));

                return true;
            }
            // CAS lost — another producer raced. Reload pos and retry.
        } else if (diff < 0) {
            // Ring is full (consumer hasn't caught up).
            overflow_drops_.fetch_add(1, std::memory_order_relaxed);
            return false;
        } else {
            // pos is stale — another producer advanced it. Reload.
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Consumer: drain()  (called only from PacingDaemon — single thread)
// ---------------------------------------------------------------------------

bool SpikeAbsorber::drain(KVBlock& out) noexcept {
    uint64_t pos  = dequeue_pos_.load(std::memory_order_relaxed);
    Slot&    slot = ring_[pos & MASK];
    uint64_t seq  = slot.seq.load(std::memory_order_acquire);
    intptr_t diff = static_cast<intptr_t>(seq) -
                    static_cast<intptr_t>(pos + 1);

    if (diff != 0) return false;  // ring empty or slot not yet written

    out = slot.blk;
    // seq = pos + CAPACITY signals that the slot is free for reuse after
    // CAPACITY more enqueues (prevents the slow producer from overwriting
    // a slot the consumer just freed).
    slot.seq.store(pos + CAPACITY, std::memory_order_release);
    dequeue_pos_.store(pos + 1, std::memory_order_relaxed);

    drained_total_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

size_t SpikeAbsorber::size_approx() const noexcept {
    uint64_t e = enqueue_pos_.load(std::memory_order_relaxed);
    uint64_t d = dequeue_pos_.load(std::memory_order_relaxed);
    return (e >= d) ? static_cast<size_t>(e - d) : 0;
}

} // namespace tempo
