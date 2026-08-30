#define main mooncake_cxi_cuda_pair_smoke_v1_main
#include "mooncake_cxi_cuda_pair_smoke.cpp"
#undef main

namespace {
int initiator_v2(const std::string& local, const std::string& remote) {
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

    mooncake::SegmentHandle segment = 0;
    std::shared_ptr<mooncake::TransferMetadata::SegmentDesc> description;
    for (int attempt = 0; attempt < 20; ++attempt) {
        segment = engine.openSegment(remote);
        description = engine.getMetadata()->getSegmentDescByID(segment);
        if (description && !description->buffers.empty() &&
            description->buffers.front().length >= kBytes) {
            break;
        }
        description.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (!description) {
        std::cerr << "remote segment has no 16MiB buffer\n";
        return 5;
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
        return 6;
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
        return 7;
    }

    std::vector<uint8_t> host(kBytes);
    if (!cuda_ok(cudaMemcpy(host.data(), buffer, kBytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy")) {
        return 8;
    }
    for (size_t index = 0; index < host.size(); ++index) {
        if (host[index] != kTargetValue) {
            std::cerr << "byte mismatch at " << index << "\n";
            return 9;
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
    if (argc != 3 && argc != 4) return 64;
    const std::string role(argv[1]);
    if (role == "target" && argc == 3) return target(argv[2]);
    if (role == "initiator" && argc == 4)
        return initiator_v2(argv[2], argv[3]);
    return 64;
}
