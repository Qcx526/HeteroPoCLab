// Small deterministic CUDA smoke test: no stress load or exploit behavior.
// Build with nvcc -std=c++14 -arch=sm_75 as documented by the smoke runner.
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>

namespace {
constexpr std::size_t kElements = 4096;
constexpr std::size_t kBytes = kElements * sizeof(float);
constexpr int kThreadsPerBlock = 128;

// Program exit codes: 0 pass, 1 CUDA operation failed, 2 unexpected/no GPU,
// 3 CPU/GPU mismatch or incomplete verification, 4 cleanup failed.
bool check_cuda(cudaError_t error, const char* operation, int failure_code,
                int& exit_code) {
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "CUDA_ERROR operation=%s code=%d name=%s message=%s\n",
                 operation, static_cast<int>(error), cudaGetErrorName(error),
                 cudaGetErrorString(error));
    if (exit_code == 0) {
        exit_code = failure_code;
    }
    return false;
}

__global__ void vector_add(const float* a, const float* b, float* result,
                           std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        result[index] = a[index] + b[index];
    }
}
}  // namespace

int main() {
    std::array<float, kElements> host_a{};
    std::array<float, kElements> host_b{};
    std::array<float, kElements> host_result{};
    // Small nonnegative integers and their sums are exactly representable in
    // float. This permits an exact per-element reference comparison.
    for (std::size_t i = 0; i < kElements; ++i) {
        host_a[i] = static_cast<float>(i % 251);
        host_b[i] = static_cast<float>((i * 7) % 127);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_result = nullptr;
    cudaDeviceProp properties{};
    bool properties_available = false;
    int device_count = 0;
    int exit_code = 0;
    std::size_t checked_elements = 0;
    std::size_t mismatch_count = 0;

    // A single cleanup path ensures each successful allocation is released,
    // including after an earlier operation has failed.
    do {
        if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount",
                        1, exit_code)) {
            break;
        }
        if (device_count < 1) {
            std::fprintf(stderr, "DEVICE_ERROR no CUDA device is available\n");
            exit_code = 2;
            break;
        }
        if (!check_cuda(cudaSetDevice(0), "cudaSetDevice(0)", 1, exit_code)) {
            break;
        }
        if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                        "cudaGetDeviceProperties(0)", 1, exit_code)) {
            break;
        }
        properties_available = true;
        if (properties.major != 7 || properties.minor != 5) {
            std::fprintf(stderr,
                         "DEVICE_ERROR this smoke test requires compute capability 7.5\n");
            exit_code = 2;
            break;
        }
        // Three 16 KiB allocations total 48 KiB of device memory.
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), kBytes),
                        "cudaMalloc(a)", 1, exit_code) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), kBytes),
                        "cudaMalloc(b)", 1, exit_code) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), kBytes),
                        "cudaMalloc(result)", 1, exit_code)) {
            break;
        }
        if (!check_cuda(cudaMemcpy(device_a, host_a.data(), kBytes,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy(a,H2D)", 1, exit_code) ||
            !check_cuda(cudaMemcpy(device_b, host_b.data(), kBytes,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy(b,H2D)", 1, exit_code)) {
            break;
        }

        const int blocks = static_cast<int>(
            (kElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
        vector_add<<<blocks, kThreadsPerBlock>>>(device_a, device_b,
                                                device_result, kElements);
        if (!check_cuda(cudaGetLastError(), "vector_add launch", 1, exit_code)) {
            break;
        }
        if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize",
                        1, exit_code)) {
            break;
        }
        if (!check_cuda(cudaMemcpy(host_result.data(), device_result, kBytes,
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(result,D2H)", 1, exit_code)) {
            break;
        }
        for (std::size_t i = 0; i < kElements; ++i) {
            const float expected = host_a[i] + host_b[i];
            ++checked_elements;
            if (host_result[i] != expected) {
                if (mismatch_count == 0) {
                    std::fprintf(stderr,
                                 "VERIFY_ERROR index=%zu expected=%.9g actual=%.9g\n",
                                 i, static_cast<double>(expected),
                                 static_cast<double>(host_result[i]));
                }
                ++mismatch_count;
            }
        }
        if (mismatch_count != 0) {
            exit_code = 3;
        }
    } while (false);

    if (device_result != nullptr) {
        check_cuda(cudaFree(device_result), "cudaFree(result)", 4, exit_code);
    }
    if (device_b != nullptr) {
        check_cuda(cudaFree(device_b), "cudaFree(b)", 4, exit_code);
    }
    if (device_a != nullptr) {
        check_cuda(cudaFree(device_a), "cudaFree(a)", 4, exit_code);
    }
    if (exit_code == 0 && checked_elements != kElements) {
        std::fprintf(stderr, "VERIFY_ERROR verification was incomplete\n");
        exit_code = 3;
    }

    std::printf("REQUESTED_DEVICE_INDEX=0\n");
    std::printf("DEVICE_NAME=%s\n",
                properties_available ? properties.name : "NOT_DETECTED");
    if (properties_available) {
        std::printf("DEVICE_CC=%d.%d\n", properties.major, properties.minor);
    } else {
        std::printf("DEVICE_CC=UNKNOWN\n");
    }
    std::printf("ELEMENTS=%zu\n", kElements);
    std::printf("CHECKED_ELEMENTS=%zu\n", checked_elements);
    std::printf("MISMATCH_COUNT=%zu\n", mismatch_count);
    std::printf("DEVICE_ALLOCATION_BYTES_MAX=%zu\n", 3 * kBytes);
    std::printf("RESULT=%s\n", exit_code == 0 ? "PASS" : "FAIL");
    std::printf("PROGRAM_EXIT_CODE=%d\n", exit_code);
    return exit_code;
}
