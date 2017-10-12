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

void TestMemcpy() {
    int size = 10;
    int memsize = size * sizeof(int);
    int *cpu_mem = (int *)malloc(memsize);
    assert(cpu_mem != NULL);
    void *gpu_mem;
    cnmemStatus_t status = cnmemMalloc(&gpu_mem, memsize, NULL);
    std::cerr << "status:" << cnmemGetErrorString(status) << std::endl;

    assert(status == CNMEM_STATUS_SUCCESS);
    int *gpu_arr = static_cast<int *>(gpu_mem);
    checkCudaErrors(cudaMemcpy(gpu_arr, cpu_mem, memsize, cudaMemcpyHostToDevice));

    free(cpu_mem);
}
