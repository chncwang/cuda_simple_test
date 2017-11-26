#include "learn.h"
#include <cstdlib>
#include <random>
#include <cublas_v2.h>

using namespace std;

__global__ void CopyElement(char *dest, char *src) {
    int index = threadIdx.x;
    dest[index] = src[index];
}

template<typename T>
void CopyGlobalArray(T *dest, T *src, int length) {
    char *dest_int = (char*)(dest);
    char *src_int = (char*)(src);
    CopyElement<<<1, length * sizeof(T)>>>(dest_int, src_int);
}

__global__ void GlobalAssignVector(float *vec, int index, float v) {
    vec[index] = v;
}

__global__ void GlobalPrintVector(float *vec, int dim) {
    printf("GlobalPrintVector:");
    for (int i = 0; i < dim; ++i) {
        printf("%f, ", vec[i]);
    }
    printf("\n");
}

void PrintGPUVector(float *vec, int dim) {
    GlobalPrintVector<<<1, 1>>>(vec, dim);
    cudaDeviceSynchronize();
}

__global__ void N3LDGKernelPrintVector(void **vec, int dim) {
    printf("N3LDGKernelPrintVector: dim %d, vec %p\n", dim, vec);
    for (int i = 0; i < dim; ++i) {
        printf("%p, ", vec[i]);
    }
    printf("\n");
}

void PrintGPUVector(void **vec, int dim) {
    N3LDGKernelPrintVector<<<1, 1>>>(vec, dim);
    cudaDeviceSynchronize();
}

void PrintCPUVector(float *vec, int dim) {
    for (int i = 0; i < dim; ++i ) {
        cout << vec[dim] << ", ";
    }
    cout << endl;
}

void InitGPUVector(float *vec, int dim) {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<> dist(-10, 10);
    for (int i = 0; i < dim; ++i) {
        GlobalAssignVector<<<1, 1>>>(vec, i, dist(mt));
    }
}

void InitCPUVector(float *vec, int dim) {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<> dist(-10, 10);
    for (int i = 0; i < dim; ++i) {
        vec[i] = dist(mt);
    }
}

float *NewGPUVector(int dim) {
    float *v;
    assert(cudaMalloc((void**)&v, sizeof(float) * dim) == cudaSuccess);
    InitGPUVector(v, dim);
    GlobalPrintVector<<<1, 1>>>(v, dim);
    cudaDeviceSynchronize();
    return v;
}

float *NewCPUVector(int dim) {
    float *v = (float*)malloc(sizeof(float) * dim);
    InitCPUVector(v, dim);
    return v;
}

constexpr int THREAD_COUNT_PER_BLOCK = 1024;
constexpr int THREAD_COUNT_PER_WRAP = 32;
constexpr int MAX_BLOCK_COUNT = 56;

int BlockCount(int size) {
    int n = (size + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK;
    return n > MAX_BLOCK_COUNT ? MAX_BLOCK_COUNT : n;
}

__global__ void Copy(float *src, float *dest, int len) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = index; i < len; i += step) {
        dest[index] = src[index];
    }
}

void N3LDGCopyArray(float *src, float *dest, int len) {
    Copy<<<BlockCount(len) ,THREAD_COUNT_PER_BLOCK>>>(src, dest, len);
}

__global__ void N3LDGKernelTanh(float *src, float *dest, int len) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = index; i < len; i += step) {
        dest[index] = tanh(src[index]);
    }
}

void N3LDGTanh(float *src, float *dest, int len) {
    N3LDGKernelTanh<<<BlockCount(len) ,THREAD_COUNT_PER_BLOCK>>>(src, dest, len);
}

__global__ void N3LDGKernelTanh(float **src, float **dest, int len, int count) {
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < len; ++j) {
            dest[i][j] = 12;
        }
    }
//    int thread_count_per_arr = (len / THREAD_COUNT_PER_WRAP + 1) *
//        THREAD_COUNT_PER_WRAP;
//    int index = blockDim.x * blockIdx.x + threadIdx.x;
//    //printf("thread count per arr:%d\n", thread_count_per_arr);
//    //printf("index:%d\n", index);
//    int step = blockDim.x * gridDim.x / thread_count_per_arr;
//    //printf("step:%d\n", step);
//    int count_index = index / thread_count_per_arr;
//    //printf("count_index:%d\n", count_index);
//    for (int i = count_index; i < count; i += step) {
//        int arr_index = index % thread_count_per_arr;
//        //printf("arr index:%d\n", arr_index);
//        if (arr_index < len) {
//            dest[i][arr_index] = tanh(src[i][arr_index]);
//        }
//    }
}

void N3LDGTanh(float **src, float **dest, int len, int count) {
    assert(len <= MAX_BLOCK_COUNT * THREAD_COUNT_PER_BLOCK);
    int block_count = BlockCount(len / THREAD_COUNT_PER_WRAP * THREAD_COUNT_PER_WRAP * count);
    //N3LDGKernelTanh<<<block_count, THREAD_COUNT_PER_BLOCK>>>(src, dest, len, count);
    N3LDGKernelTanh<<<1, 1>>>(src, dest, len, count);
}

float **ToGpuVectorArray(float** vec, int len) {
    void *result;
    int size = len * sizeof(float*);
    assert(cudaSuccess == cudaMalloc(&result, size));
    assert(cudaMemcpy(result, vec, size, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    return (float**)result;
}
