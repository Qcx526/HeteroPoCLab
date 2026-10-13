#line 1 "oob_write.cu"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef GPU_P1_FIXED
#define GPU_P1_FIXED 0
#endif

namespace {

constexpr int kThreadsPerBlock = 32;
constexpr int kMaximumCapacity = 32;

__global__ void bounded_write_kernel(int* output, int capacity, int logical_count, int extra) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int requested_end = logical_count + extra;
    if (index < requested_end) {
#if GPU_P1_FIXED
        if (index < capacity) {
            output[index] = index + 17;
        }
#else
        output[index] = index + 17;
#endif
    }
}

bool parse_bounded_int(const char* text, int minimum, int maximum, int* value) {
    if (text == nullptr || value == nullptr || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < minimum || parsed > maximum) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

int finish_app(int exit_code) {
    std::fprintf(stderr, "APP_EXIT_INTENT=%d\n", exit_code);
    return exit_code;
}

void print_cuda_status(const char* name, cudaError_t status) {
    std::printf("%s=%d:%s\n", name, static_cast<int>(status), cudaGetErrorName(status));
}

int finish_cuda_error(const char* name, cudaError_t status) {
    print_cuda_status(name, status);
    return finish_app(5);
}

}  // namespace

int main(int argc, char** argv) {
    int capacity = 0;
    int logical_count = 0;
    int extra = 0;
    if (argc != 4 ||
        !parse_bounded_int(argv[1], 1, kMaximumCapacity, &capacity) ||
        !parse_bounded_int(argv[2], 1, capacity, &logical_count) ||
        !parse_bounded_int(argv[3], 0, 2, &extra)) {
        std::fprintf(stderr, "INPUT_ERROR=expected capacity, logical_count, extra(0..2)\n");
        return finish_app(2);
    }

    std::printf("APP_VARIANT=%s\n", GPU_P1_FIXED ? "fixed" : "vulnerable");
    std::printf("APP_CAPACITY=%d\n", capacity);
    std::printf("APP_LOGICAL_COUNT=%d\n", logical_count);
    std::printf("APP_EXTRA_ELEMENTS=%d\n", extra);
    std::printf("APP_EXPECTED_TRIGGER=%s\n",
                (!GPU_P1_FIXED && logical_count == capacity && extra >= 1) ? "true" : "false");

    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        print_cuda_status("CUDA_GET_DEVICE", status);
        return finish_app(3);
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        print_cuda_status("CUDA_GET_DEVICE_PROPERTIES", status);
        return finish_app(3);
    }
    std::printf("APP_DEVICE=%s\n", properties.name);
    std::printf("APP_DEVICE_CC=%d.%d\n", properties.major, properties.minor);

    int* device_output = nullptr;
    status = cudaMalloc(&device_output, static_cast<size_t>(capacity) * sizeof(int));
    if (status != cudaSuccess) {
        print_cuda_status("CUDA_MALLOC", status);
        return finish_app(4);
    }
    status = cudaMemset(device_output, 0, static_cast<size_t>(capacity) * sizeof(int));
    if (status != cudaSuccess) {
        print_cuda_status("CUDA_MEMSET", status);
        (void)cudaFree(device_output);
        return finish_app(4);
    }

    const int requested_end = logical_count + extra;
    const int blocks = (requested_end + kThreadsPerBlock - 1) / kThreadsPerBlock;
    bounded_write_kernel<<<blocks, kThreadsPerBlock>>>(
        device_output, capacity, logical_count, extra);
    const cudaError_t launch_status = cudaPeekAtLastError();
    print_cuda_status("CUDA_KERNEL_LAUNCH", launch_status);
    if (launch_status != cudaSuccess) {
        (void)cudaFree(device_output);
        return finish_app(4);
    }

    const cudaError_t sync_status = cudaDeviceSynchronize();
    print_cuda_status("CUDA_DEVICE_SYNCHRONIZE", sync_status);
    if (sync_status != cudaSuccess) {
        std::printf("APP_CPU_COMPARE=SKIPPED_CUDA_API_ERROR\n");
        print_cuda_status("CUDA_FREE_AFTER_SYNC", cudaFree(device_output));
        return finish_cuda_error("APP_CUDA_SYNC_ERROR", sync_status);
    }

    std::vector<int> actual(static_cast<size_t>(capacity), 0);
    status = cudaMemcpy(actual.data(), device_output,
                        static_cast<size_t>(capacity) * sizeof(int), cudaMemcpyDeviceToHost);
    print_cuda_status("CUDA_MEMCPY_DEVICE_TO_HOST", status);
    print_cuda_status("CUDA_FREE", cudaFree(device_output));
    if (status != cudaSuccess) {
        return finish_app(4);
    }

    int mismatch_count = 0;
    const int expected_written = std::min(requested_end, capacity);
    for (int index = 0; index < capacity; ++index) {
        const int expected = index < expected_written ? index + 17 : 0;
        if (actual[static_cast<size_t>(index)] != expected) {
            ++mismatch_count;
        }
    }
    std::printf("APP_CPU_COMPARE_ELEMENTS=%d\n", capacity);
    std::printf("APP_CPU_COMPARE_MISMATCHES=%d\n", mismatch_count);
    return finish_app(mismatch_count == 0 ? 0 : 6);
}
