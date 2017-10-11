#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "cnmem.h"
#include <cassert>
#include "learn.h"
#include <cstdlib>

void InitGpu() {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.device = 0;
    device.size = 10000000000;
    cudaSetDevice(0);
    assert(cnmemInit(1, &device, CNMEM_FLAGS_CANNOT_GROW));
}

void TestMemcpy() {
    int size = 100;
    int memsize = size * sizeof(int);
    int *cpu_mem = (int *)malloc(memsize);
    assert(cpu_mem != NULL);
    void *gpu_mem;
    cnmemStatus_t status = cnmemMalloc(&gpu_mem, memsize, NULL);
    assert(status == CNMEM_STATUS_SUCCESS);
    int *gpu_arr = static_cast<int *>(gpu_mem);
    checkCudaErrors(cudaMemcpy(gpu_arr, cpu_mem, memsize, cudaMemcpyHostToDevice));

    free(cpu_mem);
}
