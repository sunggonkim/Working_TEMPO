// Bounded two-node CUDA correctness smoke for Mooncake's official CXI
// TransferEngine.  The topology is explicit so this test never discovers or
// walks /sys.  This is a component preflight, not an end-to-end KV benchmark.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "transfer_engine.h"

using mooncake::TransferEngine;
using mooncake::TransferRequest;
using mooncake::TransferStatus;
using mooncake::TransferStatusEnum;

namespace {
constexpr size_t kBytes = 16ULL << 20;
constexpr int kTargetValue = 42;
constexpr int kInitialValue = 66;
constexpr auto kTransferTimeout = std::chrono::seconds(12);

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

std::string explicit_topology() {
    return R"({"cpu:0":[["cxi0"],[]],"cpu:1":[["cxi0"],[]],"cuda:0":[["cxi0"],[]]})";
}

bool configure(TransferEngine& engine, const std::string& local) {
    const auto colon = local.rfind(':');
    if (colon == std::string::npos) return false;
    const std::string host = local.substr(0, colon);
    const auto port = static_cast<uint64_t>(std::stoul(local.substr(colon + 1)));
    if (engine.init("P2PHANDSHAKE", local, host, port) != 0) {
        std::cerr << "engine.init failed\n";
        return false;
    }
    if (engine.getLocalTopology()->parse(explicit_topology()) != 0) {
        std::cerr << "explicit topology parse failed\n";
        return false;
    }
    if (engine.installTransport("cxi", nullptr) == nullptr) {
        std::cerr << "installTransport(cxi) failed\n";
        return false;
    }
    return true;
}

int target(const std::string& local) {
    TransferEngine engine(false);
    if (!configure(engine, local)) return 2;
    void* buffer = nullptr;
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice") ||
        !cuda_ok(cudaMalloc(&buffer, kBytes), "cudaMalloc") ||
        !cuda_ok(cudaMemset(buffer, kTargetValue, kBytes), "cudaMemset")) {
        return 3;
    }
    if (engine.registerLocalMemory(buffer, kBytes, "*", true) != 0) {
        std::cerr << "target registerLocalMemory failed\n";
        cudaFree(buffer);
        return 4;
    }
    std::cout << "TARGET_READY bytes=" << kBytes << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(18));
    engine.unregisterLocalMemory(buffer);
    cudaFree(buffer);
    return 0;
}

int initiator(const std::string& local, const std::string& remote) {
    TransferEngine engine(false);
    if (!configure(engine, local)) return 2;
    void* buffer = nullptr;
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice") ||
        !cuda_ok(cudaMalloc(&buffer, kBytes), "cudaMalloc") ||
        !cuda_ok(cudaMemset(buffer, kInitialValue, kBytes), "cudaMemset")) {
        return 3;
    }
    if (engine.registerLocalMemory(buffer, kBytes, "*", true) != 0) {
        std::cerr << "initiator registerLocalMemory failed\n";
        cudaFree(buffer);
        return 4;
    }

    mooncake::SegmentHandle segment = -1;
    for (int attempt = 0; attempt < 20 && segment < 0; ++attempt) {
        segment = engine.openSegment(remote);
        if (segment < 0) std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (segment < 0) {
        std::cerr << "openSegment failed\n";
        return 5;
    }
    auto description = engine.getMetadata()->getSegmentDescByID(segment);
    if (!description || description->buffers.empty() ||
        description->buffers.front().length < kBytes) {
        std::cerr << "remote segment has no 16MiB buffer\n";
        return 6;
    }

    const auto batch = engine.allocateBatchID(1);
    TransferRequest request;
    request.opcode = TransferRequest::READ;
    request.source = static_cast<uint8_t*>(buffer);
    request.target_id = segment;
    request.target_offset = description->buffers.front().addr;
    request.length = kBytes;
    const auto begin = std::chrono::steady_clock::now();
    const auto submitted = engine.submitTransfer(batch, {request});
    if (!submitted.ok()) {
        std::cerr << "submitTransfer failed: " << submitted.ToString() << "\n";
        return 7;
    }
    bool complete = false;
    while (std::chrono::steady_clock::now() - begin < kTransferTimeout) {
        TransferStatus status;
        engine.getTransferStatus(batch, 0, status);
        if (status.s == TransferStatusEnum::COMPLETED) {
            complete = true;
            break;
        }
        if (status.s == TransferStatusEnum::FAILED) break;
        std::this_thread::yield();
    }
    const auto end = std::chrono::steady_clock::now();
    engine.freeBatchID(batch);
    if (!complete) {
        std::cerr << "transfer failed or timed out\n";
        return 8;
    }

    std::vector<uint8_t> host(kBytes);
    if (!cuda_ok(cudaMemcpy(host.data(), buffer, kBytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy")) {
        return 9;
    }
    for (size_t index = 0; index < host.size(); ++index) {
        if (host[index] != kTargetValue) {
            std::cerr << "byte mismatch at " << index << "\n";
            return 10;
        }
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << "MOONCAKE_CXI_CUDA_OK bytes=" << kBytes
              << " elapsed_ms=" << elapsed_ms << std::endl;
    engine.closeSegment(segment);
    engine.unregisterLocalMemory(buffer);
    cudaFree(buffer);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: smoke target LOCAL | smoke initiator LOCAL REMOTE\n";
        return 64;
    }
    const std::string role(argv[1]);
    if (role == "target" && argc == 3) return target(argv[2]);
    if (role == "initiator" && argc == 4) return initiator(argv[2], argv[3]);
    return 64;
}
