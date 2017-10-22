#ifndef CUDA_LEARN_LEARN_H
#define CUDA_LEARN_LEARN_H
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "cnmem.h"
#include <cassert>

void InitGpu();
void TestGetMax();
void TestMemCpy();

void TestFastCpy(cudaStream_t stream);

#endif
