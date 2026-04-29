// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/pacing_daemon/pacing_daemon.cpp
//
// Core harmonious flush loop:
//
//   ┌──────────────────────────────────────────────────────────────────┐
//   │  while (running)                                                 │
//   │    ①  wait_for_ffn(ffn_wait_us)   ← gate: pause during ATTENTION │
//   │    ②  drain one KVBlock from SpikeAbsorber                       │
//   │    ③  token_bucket.try_consume(size) ← rate-limit Slingshot NIC  │
//   │    ④  io_uring_prep_write → Lustre   ← async, IOSQE_ASYNC        │
//   │    ⑤  io_uring_submit + collect CQEs                             │
//   └──────────────────────────────────────────────────────────────────┘
//
// io_uring is used instead of pwrite()/aio because:
//   - Zero syscall overhead for batched writes (submit multiple SQEs at once)
//   - IOSQE_ASYNC flag bypasses io_uring's internal "try-sync-first" path,
//     guaranteeing non-blocking async semantics on Lustre
//   - Works with O_DIRECT on Lustre (avoids double-buffering in page cache)

#include "pacing_daemon.hpp"

#include <liburing.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace tempo {

// ---------------------------------------------------------------------------
// PacingConfig helpers
// ---------------------------------------------------------------------------

PacingConfig PacingConfig::from_env() {
    PacingConfig c;
    // TEMPO_RATE_GBPS: flush rate cap in GB/s (default 5)
    if (const char* v = std::getenv("TEMPO_RATE_GBPS")) {
        double gbps = std::stod(v);
        c.rate_bytes_per_sec = static_cast<uint64_t>(gbps * 1e9);
    }
    // TEMPO_BURST_MB: token bucket burst in MB (default 256)
    if (const char* v = std::getenv("TEMPO_BURST_MB")) {
        uint64_t mb = static_cast<uint64_t>(std::stoul(v));
        c.burst_bytes = mb * 1024 * 1024;
    }
    // TEMPO_STAGE_DIR: NVMe staging directory
    if (const char* v = std::getenv("TEMPO_STAGE_DIR")) c.staging_dir = v;
    // TEMPO_LUSTRE_DIR: Lustre flush target
    if (const char* v = std::getenv("TEMPO_LUSTRE_DIR")) c.lustre_dir = v;
    else if (const char* v = std::getenv("PSCRATCH"))
        c.lustre_dir = std::string(v) + "/tempo_kvcache";
    // TEMPO_FFN_WAIT_US: max wait for FFN window in µs
    if (const char* v = std::getenv("TEMPO_FFN_WAIT_US"))
        c.ffn_wait_us = std::stoull(v);
    // TEMPO_STRICT_GATE: 0 to disable strict gating
    if (const char* v = std::getenv("TEMPO_STRICT_GATE"))
        c.strict_gate = (std::stoi(v) != 0);
    if (const char* v = std::getenv("TEMPO_VERBOSE"))
        c.verbose = (std::stoi(v) != 0);
    return c;
}

// ---------------------------------------------------------------------------
// PacingDaemon — ctor / dtor
// ---------------------------------------------------------------------------

PacingDaemon::PacingDaemon(AttentionPhaseMonitor& monitor,
                            SpikeAbsorber&         absorber,
                            PacingConfig           cfg)
    : monitor_(monitor)
    , absorber_(absorber)
    , cfg_(std::move(cfg))
    , bucket_(cfg_.rate_bytes_per_sec, cfg_.burst_bytes)
{}

PacingDaemon::~PacingDaemon() { stop(); }

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void PacingDaemon::start() {
    if (running_.exchange(true)) return;  // already running

    // Initialise io_uring
    auto* ring = new struct io_uring;
    if (io_uring_queue_init(cfg_.uring_depth, ring, 0) < 0) {
        running_.store(false);
        delete ring;
        throw std::runtime_error("PacingDaemon: io_uring_queue_init failed: " +
                                  std::string(std::strerror(errno)));
    }
    uring_ptr_ = ring;

    // Create staging and Lustre dirs
    std::filesystem::create_directories(cfg_.staging_dir);
    if (!cfg_.lustre_dir.empty())
        std::filesystem::create_directories(cfg_.lustre_dir);

    thread_ = std::thread(&PacingDaemon::run_loop, this);
}

void PacingDaemon::stop() noexcept {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    if (uring_ptr_) {
        io_uring_queue_exit(static_cast<struct io_uring*>(uring_ptr_));
        delete static_cast<struct io_uring*>(uring_ptr_);
        uring_ptr_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// open_lustre()  — open/create the Lustre flush target file
//
// O_DIRECT bypasses the Linux page cache — mandatory on Lustre to avoid
// double-buffering and to ensure we measure raw Slingshot NIC throughput.
// Stripe count should be set externally with `lfs setstripe -c 4`.
// ---------------------------------------------------------------------------

int PacingDaemon::open_lustre() {
    if (cfg_.lustre_dir.empty())
        throw std::runtime_error("PacingDaemon: TEMPO_LUSTRE_DIR not set");

    std::string path = cfg_.lustre_dir + "/kv_blocks.dat";
    int fd = ::open(path.c_str(),
                    O_WRONLY | O_CREAT | O_DIRECT | O_CLOEXEC,
                    0644);
    if (fd < 0)
        throw std::system_error(errno, std::system_category(),
                                 "PacingDaemon: open(" + path + ")");
    return fd;
}

// ---------------------------------------------------------------------------
// flush_block()  — submit one KV block to io_uring
// ---------------------------------------------------------------------------

void PacingDaemon::flush_block(const KVBlock& blk, int lustre_fd) {
    auto* ring = static_cast<struct io_uring*>(uring_ptr_);

    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        // SQ is full — collect completions first, then retry
        io_uring_submit(ring);
        struct io_uring_cqe* cqe;
        io_uring_wait_cqe(ring, &cqe);
        io_uring_cqe_seen(ring, cqe);
        sqe = io_uring_get_sqe(ring);
        if (!sqe) return;  // give up this block
    }

    // Write the KV payload. block_id encodes the file offset so different
    // blocks land at non-overlapping positions (page-aligned).
    constexpr off_t BLOCK_SIZE = 4096;  // must match O_DIRECT alignment
    off_t offset = static_cast<off_t>(blk.block_id) * BLOCK_SIZE;

    io_uring_prep_write(sqe, lustre_fd,
                        blk.host_ptr, static_cast<unsigned>(blk.size_bytes),
                        offset);
    // IOSQE_ASYNC: skip the "try sync first" optimisation — always async.
    // Essential for Lustre where sync paths can block for tens of ms.
    io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
    sqe->user_data = blk.block_id;  // echoed back in CQE for accounting

    io_uring_submit(ring);

    bytes_flushed_.fetch_add(blk.size_bytes, std::memory_order_relaxed);
    blocks_flushed_.fetch_add(1,             std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// run_loop()  — THE harmonious flush algorithm
// ---------------------------------------------------------------------------

void PacingDaemon::run_loop() {
    int lustre_fd = -1;
    try {
        lustre_fd = open_lustre();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[TEMPO PacingDaemon] FATAL: %s\n", e.what());
        running_.store(false);
        return;
    }

    auto* ring = static_cast<struct io_uring*>(uring_ptr_);

    if (cfg_.verbose)
        std::fprintf(stderr, "[TEMPO PacingDaemon] started  "
                              "rate=%.1f GB/s  burst=%zu MB  "
                              "strict_gate=%d\n",
                     static_cast<double>(cfg_.rate_bytes_per_sec) / 1e9,
                     cfg_.burst_bytes / (1024 * 1024),
                     static_cast<int>(cfg_.strict_gate));

    while (running_.load(std::memory_order_relaxed)) {

        // ── ① Phase gate ─────────────────────────────────────────────────
        // Suppress flushing during ATTENTION to eliminate PCIe contention.
        if (cfg_.strict_gate && monitor_.is_attention()) {
            bool got_ffn = monitor_.wait_for_ffn(cfg_.ffn_wait_us);
            if (!got_ffn) {
                attention_pauses_.fetch_add(1, std::memory_order_relaxed);
                continue;  // spin back to check running_ and re-gate
            }
        }

        // ── ② Drain one block from SpikeAbsorber ─────────────────────────
        KVBlock blk;
        if (!absorber_.drain(blk)) {
            // Ring empty — wait for absorber eventfd (up to 1 ms)
            // to avoid busy-polling when there's nothing to do.
            struct io_uring_cqe* cqe = nullptr;
            struct __kernel_timespec ts{0, 1'000'000};  // 1 ms
            io_uring_wait_cqe_timeout(ring, &cqe, &ts);
            if (cqe) io_uring_cqe_seen(ring, cqe);
            continue;
        }

        // ── ③ Token bucket: rate-limit Slingshot NIC usage ───────────────
        uint64_t granted = bucket_.try_consume(blk.size_bytes);
        if (granted == 0) {
            // Bucket empty — re-absorb block (push back to absorber) and wait
            // a short time for the bucket to refill.
            absorber_.absorb(blk);  // re-enqueue (tail of ring)
            struct __kernel_timespec ts{0, 100'000};  // 100 µs
            struct io_uring_cqe* cqe = nullptr;
            io_uring_wait_cqe_timeout(ring, &cqe, &ts);
            if (cqe) io_uring_cqe_seen(ring, cqe);
            continue;
        }

        // ── ④⑤ io_uring async write to Lustre ────────────────────────────
        flush_block(blk, lustre_fd);

        // Collect any available CQEs without blocking (best-effort)
        struct io_uring_cqe* cqe;
        while (io_uring_peek_cqe(ring, &cqe) == 0) {
            if (cqe->res < 0 && cfg_.verbose) {
                std::fprintf(stderr, "[TEMPO PacingDaemon] write error "
                                      "block_id=%llu: %s\n",
                             static_cast<unsigned long long>(cqe->user_data),
                             std::strerror(-cqe->res));
            }
            io_uring_cqe_seen(ring, cqe);
        }
    }

    // Drain remaining CQEs on shutdown
    unsigned pending = io_uring_cq_ready(ring);
    while (pending-- > 0) {
        struct io_uring_cqe* cqe;
        if (io_uring_peek_cqe(ring, &cqe) == 0)
            io_uring_cqe_seen(ring, cqe);
    }

    ::close(lustre_fd);

    if (cfg_.verbose)
        std::fprintf(stderr, "[TEMPO PacingDaemon] stopped  "
                              "flushed %llu blocks  %llu MB  "
                              "attention_pauses=%llu\n",
                     static_cast<unsigned long long>(blocks_flushed()),
                     static_cast<unsigned long long>(bytes_flushed() >> 20),
                     static_cast<unsigned long long>(attention_pauses()));
}

} // namespace tempo
