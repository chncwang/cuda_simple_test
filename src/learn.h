#ifndef CUDA_LEARN_LEARN_H
#define CUDA_LEARN_LEARN_H
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <cassert>

template<typename T>
void CopyGlobalArray(T *dest, T *src, int length);

void PrintGPUVector(float *vec, int dim);
void PrintGPUVector(void **vec, int dim);

void PrintCPUVector(float *vec, int dim);

void InitGPUVector(float *vec, int dim);

void InitCPUVector(float *vec, int dim);

float *NewGPUVector(int dim);

float *NewCPUVector(int dim);

void N3LDGCopyArray(float *src, float *dest, int len);

void N3LDGTanh(float *src, float *dest, int len);

void N3LDGTanh(float **src, float **dest, int len, int count);
float **ToGpuVectorArray(float** vec, int len);
void Benchmark();

#endif
