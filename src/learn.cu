#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "cnmem.h"
#include <cassert>
#include "learn.h"
#include <cstdlib>

using namespace std;

void InitGpu() {
    int deviceid = 0;
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.device = deviceid;
    device.size = 1000000000;
    cudaSetDevice(deviceid);
    cnmemStatus_t status = cnmemInit(1, &device, CNMEM_FLAGS_CANNOT_GROW);
    std::cerr << "status:" << cnmemGetErrorString(status) << std::endl;
    assert(status == CNMEM_STATUS_SUCCESS);
}


__global__ void max(float **arr) {
    int idx = threadIdx.x;
    float max = -1;
    for (int i = 0; i < 100; ++i) {
        if (arr[idx][i] > max) {
            max = arr[idx][i];
        }
    }
}

__global__ void print_arr(float *arr) {
    printf("idx:%d\n", threadIdx.x);
    printf("%f\n", arr[threadIdx.x]);
}

void TestMemCpy() {
    void *gpu_mem;
    cnmemStatus_t status = cnmemMalloc(&gpu_mem, 100 * sizeof(float), NULL);
    assert(status == CNMEM_STATUS_SUCCESS);
    float *gpu_arr = static_cast<float *>(gpu_mem);
    float *cpu_mem = (float *)malloc(100 * sizeof(float));
    for (int i = 0; i < 100; ++i) {
        cpu_mem[i] = i;
    }
    checkCudaErrors(cudaMemcpy(gpu_arr, cpu_mem, 100 * sizeof(float), cudaMemcpyHostToDevice));
    print_arr<<<1, 100>>>(gpu_arr);
    cudaDeviceSynchronize();
}

void TestGetMax() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    void *gpu_mem;
    cnmemStatus_t status = cnmemMalloc(&gpu_mem, 100 * sizeof(float*), NULL);
    assert(status == CNMEM_STATUS_SUCCESS);
    float **gpu_arr = static_cast<float **>(gpu_mem);
    float **cpu_mem = (float **)malloc(100 * sizeof(float*));
    for (int i = 0; i< 100; ++i) {
        void *gpu_mem2;
        cnmemStatus_t status = cnmemMalloc(&gpu_mem2, 100 * sizeof(float), NULL);
        assert(status == CNMEM_STATUS_SUCCESS);
        float *gpu_arr2 = static_cast<float *>(gpu_mem2);
        float *cpu_mem2 = (float*)malloc(100 * sizeof(float));
        for (int j = 0; j < 100; ++j) {
            cpu_mem2[j] = i * j;
        }
        cudaEventRecord(start);
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        checkCudaErrors(cudaMemcpyAsync(gpu_arr2, cpu_mem2, 100 * sizeof(float), cudaMemcpyHostToDevice, stream));
        cudaEventRecord(stop);
        free(cpu_mem2);
        cpu_mem[i] = gpu_arr2;
    }
    checkCudaErrors(cudaMemcpy(gpu_arr, cpu_mem, 100 * sizeof(float*), cudaMemcpyHostToDevice));
    std::cout << "begin max" << std::endl;
    max<<<1, 100>>>(gpu_arr);
    cudaDeviceSynchronize();
    std::cout << "end max" << std::endl;
}

__global__ void set_zero(float *mem) {
    int v = threadIdx.x;
    mem[v] = v;
}

void TestFastCpy() {
    void *src_mem;
    void *dest_mem;
    cnmemStatus_t status = cnmemMalloc(&src_mem, 100 * sizeof(float), NULL);
    assert(status == CNMEM_STATUS_SUCCESS);
    set_zero<<<1, 100>>>((float*)src_mem);
    status = cnmemMalloc(&dest_mem, 100 * sizeof(float), NULL);
    assert(status == CNMEM_STATUS_SUCCESS);
    cudaMemcpyAsync((float*)dest_mem, (float*)src_mem, 100 * sizeof(float), cudaMemcpyDeviceToDevice, 0);
    status = cnmemFree(src_mem, 0);
    assert(status == CNMEM_STATUS_SUCCESS);
    status = cnmemFree(dest_mem, 0);
    assert(status == CNMEM_STATUS_SUCCESS);
}
