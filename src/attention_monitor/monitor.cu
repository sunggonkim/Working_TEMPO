// SPDX-License-Identifier: Apache-2.0
// TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
// src/attention_monitor/monitor.cu

#include "monitor.hpp"

#include <cupti.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

#define CUPTI_CHECK(call)                                                      \
    do {                                                                       \
        CUptiResult _err = (call);                                             \
        if (_err != CUPTI_SUCCESS) {                                           \
            const char* _msg = "unknown";                                      \
            cuptiGetResultString(_err, &_msg);                                 \
            throw std::runtime_error(std::string("CUPTI: ") + _msg +          \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

namespace tempo {

// ---------------------------------------------------------------------------
// Kernel name → Phase classification
//
// Patterns are matched as substring of the demangled kernel name.
// Ordered: most-specific first.
// ---------------------------------------------------------------------------

struct KernelPattern { const char* substr; Phase phase; };

static constexpr KernelPattern kPatterns[] = {
    // ── Attention kernels ─────────────────────────────────────────────────
    // vLLM PagedAttention (v1 & v2)
    { "paged_attention_v1",           Phase::ATTENTION },
    { "paged_attention_v2",           Phase::ATTENTION },
    // FlashAttention-2 / FA3 (fwd + bwd)
    { "flash_fwd_kernel",             Phase::ATTENTION },
    { "flash_bwd_kernel",             Phase::ATTENTION },
    { "flash_fwd_splitkv_kernel",     Phase::ATTENTION },
    // SGLang Triton attention kernels
    { "context_attention_fwd",        Phase::ATTENTION },
    { "token_attention_fwd",          Phase::ATTENTION },
    { "extend_attention_fwd",         Phase::ATTENTION },
    // PyTorch SDPA (efficient_attention / flash_attn)
    { "scaled_dot_product_fused",     Phase::ATTENTION },
    { "efficient_attention_forward",  Phase::ATTENTION },
    { "AttentionForwardFromBlockedInput", Phase::ATTENTION },
    // NVIDIA FasterTransformer / TRT-LLM
    { "Attention",                    Phase::ATTENTION },   // broad: last resort

    // ── FFN / Linear kernels ─────────────────────────────────────────────
    // CUTLASS GEMM templates
    { "cutlass_gemm",                 Phase::FFN },
    { "cutlass3x",                    Phase::FFN },
    // cuBLAS
    { "cublas",                       Phase::FFN },         // cublasSgemm*, cublasBf16...
    // Ampere / Hopper architecture-specific names
    { "ampere_bf16",                  Phase::FFN },
    { "ampere_fp16",                  Phase::FFN },
    { "sm80_xmma",                    Phase::FFN },
    { "sm90_xmma",                    Phase::FFN },
    // Volta
    { "volta_sgemm",                  Phase::FFN },
    { "volta_hgemm",                  Phase::FFN },
    // PyTorch fused kernels for FFN
    { "fused_gelu_linear",            Phase::FFN },
    { "activation_kernel",            Phase::FFN },         // SiLU/GELU in MLP
    { "splitKreduce_kernel",          Phase::FFN },

    // sentinel
    { nullptr, Phase::UNKNOWN },
};

Phase classify_kernel(const char* name) noexcept {
    if (!name || *name == '\0') return Phase::UNKNOWN;
    for (const KernelPattern* p = kPatterns; p->substr; ++p) {
        if (std::strstr(name, p->substr)) return p->phase;
    }
    return Phase::OTHER;
}

// ---------------------------------------------------------------------------
// CUPTI callback (called synchronously on the CPU thread that called
// cudaLaunchKernel — not on the GPU side)
// ---------------------------------------------------------------------------

static void CUPTIAPI cupti_callback(void*             userdata,
                                    CUpti_CallbackDomain domain,
                                    CUpti_CallbackId  cbid,
                                    const void*       cbdata)
{
    // We only care about runtime API launches
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;

    // Act on ENTRY only (sets phase *before* the kernel is submitted to the
    // GPU queue — gives the pacing daemon maximum notice to pause)
    const auto* data = static_cast<const CUpti_CallbackData*>(cbdata);
    if (data->callbackSite != CUPTI_API_ENTER) return;

    // Accept both regular and graph-launch callbacks
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v3020 &&
        cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 &&
        cbid != CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000) return;

    Phase p = classify_kernel(data->symbolName);
    if (p == Phase::ATTENTION || p == Phase::FFN) {
        static_cast<AttentionPhaseMonitor*>(userdata)->_set_phase(p);
    }
}

// ---------------------------------------------------------------------------
// AttentionPhaseMonitor implementation
// ---------------------------------------------------------------------------

void AttentionPhaseMonitor::start() {
    if (started_) return;

    CUPTI_CHECK(cuptiSubscribe(&subscriber_,
                               reinterpret_cast<CUpti_CallbackFunc>(cupti_callback),
                               static_cast<void*>(this)));

    // Enable cudaLaunchKernel
    CUPTI_CHECK(cuptiEnableCallback(1, subscriber_,
                                    CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v3020));
    // Enable extended launch (used by newer PyTorch)
    CUPTI_CHECK(cuptiEnableCallback(1, subscriber_,
                                    CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060));
    // Enable CUDA Graph launch (FA3 uses CUDA graphs)
    CUPTI_CHECK(cuptiEnableCallback(1, subscriber_,
                                    CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000));
    started_ = true;
}

void AttentionPhaseMonitor::stop() noexcept {
    if (!started_ || !subscriber_) return;
    cuptiUnsubscribe(subscriber_);
    subscriber_ = nullptr;
    started_    = false;
}

void AttentionPhaseMonitor::_set_phase(Phase p) noexcept {
    Phase prev = phase_.exchange(p, std::memory_order_acq_rel);
    if (prev == p) return;  // no change

    if (p == Phase::ATTENTION) {
        attn_count_.fetch_add(1, std::memory_order_relaxed);
    } else if (p == Phase::FFN) {
        ffn_count_.fetch_add(1, std::memory_order_relaxed);
        // Wake pacing daemon blocked in wait_for_ffn()
        std::lock_guard<std::mutex> lk(cv_mtx_);
        cv_.notify_all();
    }
}

bool AttentionPhaseMonitor::wait_for_ffn(uint64_t timeout_us) noexcept {
    // Fast path: already in FFN
    if (is_ffn()) return true;

    std::unique_lock<std::mutex> lk(cv_mtx_);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(timeout_us);
    return cv_.wait_until(lk, deadline, [this] { return is_ffn(); });
}

} // namespace tempo
