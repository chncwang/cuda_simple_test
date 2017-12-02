#include "learn.h"
#include <cstdlib>
#include <random>
#include <cublas_v2.h>
#include <utility>

using namespace std;

constexpr int THREAD_COUNT_PER_BLOCK = 1024;
constexpr int THREAD_COUNT_PER_WRAP = 32;
constexpr int MAX_BLOCK_COUNT = 56;


void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status) << std::endl;
        abort();
    }
}

int BlockCount(int size) {
    int n = (size + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK;
    return n > MAX_BLOCK_COUNT ? MAX_BLOCK_COUNT : n;
}

#define  KernelPrintLine(format, ...)
//{\
//    printf("block:x=%d,y=%d thread:x=%d,y=%d "#format"\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,\
//            __VA_ARGS__);\
//}

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

__global__ void GlobalAssignVector(float *vec, int dim) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x) {
        vec[i] = (i * 1664525 + 1013904223) % 100000000 / 5000000.0 - 10;
    }
}

void InitGPUVector(float *vec, int dim) {
    GlobalAssignVector<<<BlockCount(dim), THREAD_COUNT_PER_BLOCK>>>(vec, dim);
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
    CallCuda(cudaMalloc((void**)&v, sizeof(float) * dim));
    InitGPUVector(v, dim);
    cudaDeviceSynchronize();
    return v;
}

float *NewCPUVector(int dim) {
    float *v = (float*)malloc(sizeof(float) * dim);
    InitCPUVector(v, dim);
    return v;
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

__global__ void SingleThreadKernelTanh(float **src, float **dest, int len, int count) {
    for (int i = 0; i< count; ++i)
        for (int j = 0; j<len;++j)
            dest[i][j] = tanh(src[i][j]);
}


void N3LDGSingleThreadTanh(float **src, float **dest, int len, int count) {
    SingleThreadKernelTanh<<<1, 1>>>(src, dest, len, count);
}

__global__ void N3LDGKernelTanh(float **src, float **dest, int len, int count) {
    int thread_count_per_arr = ((len - 1) / THREAD_COUNT_PER_WRAP + 1) * THREAD_COUNT_PER_WRAP;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x / thread_count_per_arr;
    assert(step > 0);
    int count_index = index / thread_count_per_arr;

    for (int i = count_index; i < count; i += step) {
        int arr_index = index % thread_count_per_arr;
        if (arr_index < len) {
            dest[i][arr_index] = tanh(src[i][arr_index]);
        }
    }
}

__global__ void N3LDGKernelTanhWithoutLimit(float **src, float **dest, int len, int count) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int total_count = len * count;
    int step = blockDim.x * gridDim.x;
    for (int i = index; i < total_count; i += step) {
        int ci = i / len;
        int li = i % len;
        dest[ci][li] = tanh(src[ci][li]);
    }
}

void N3LDGTanhWithoutLimit(float **src, float **dest, int len, int count) {
    N3LDGKernelTanhWithoutLimit<<<BlockCount(len * count), THREAD_COUNT_PER_BLOCK>>>(src, dest, len, count);
}

void N3LDGTanh(float **src, float **dest, int len, int count) {
    if (len <= MAX_BLOCK_COUNT * THREAD_COUNT_PER_BLOCK) {
        int block_count = BlockCount(((len - 1) / THREAD_COUNT_PER_WRAP + 1) *
                THREAD_COUNT_PER_WRAP * count);
        N3LDGKernelTanh<<<block_count, THREAD_COUNT_PER_BLOCK>>>(src, dest, len, count);
    } else {
        N3LDGTanhWithoutLimit(src, dest, len, count);
    }
}


__global__ void N3LDGKernelTanhByBlock(float **src, float **dest, int len, int count) {
    __shared__ volatile float *src_arr;
    __shared__ volatile float *dest_arr;
    for (int i = blockIdx.x; i < count; i += gridDim.x) {
        __syncthreads();
        if (threadIdx.x == 0) {
            src_arr = src[i];
            dest_arr = dest[i];
        }
        __syncthreads();
        for (int j=threadIdx.x; j< len; j += blockDim.x) {
            dest_arr[j] = tanh(src_arr[j]);
        }
    }
}

void N3LDGTanhByBlock(float **src, float **dest, int len, int count) {
    N3LDGKernelTanhByBlock<<<min(MAX_BLOCK_COUNT, count), THREAD_COUNT_PER_BLOCK>>>(src, dest, len, count);
}

float** NewGPUVectors(int count, int dim) {
    float**  result = (float**)malloc(count * sizeof(float*));
    for (int i = 0; i<count; ++i) {
        float * vec = NewGPUVector(dim);
        result[i] = vec;
    }
    return result;
}

__global__ void KernelAssertEqual(float **a, float **b, int len, int count) {
    for (int i = 0; i < count; ++i) {
        for (int j=0; j < len; ++j) {
            float s = (a[i][j] - b[i][j]) / a[i][j];
            if(s < -0.001 || s > 0.001) {
                printf("i:%d, j:%d, a[i][j]:%f, b[i][j]:%f, s:%f\n", i, j, a[i][j], b[i][j], s);
            }
        }
    }
}

__global__ void KernelAssertEqual(float *a, float *b, int len) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < len; i += gridDim.x * blockDim.x) {
        int s = (a[i] - b[i]) / a[i];
        float m = 0.001;
        if(s < -m || s > m) {
            printf("i:%d, a[i]:%f, b[i]:%f\n", i, a[i], b[i]);
        }
    }
}

void N3LDGAssertEqual(float *a, float*b, int len) {
    KernelAssertEqual<<<min(MAX_BLOCK_COUNT, BlockCount(len)), THREAD_COUNT_PER_BLOCK>>>(a, b, len);
}

__global__ void N3LDGKernelMultiply(float *matrix, int row, int col, float* vec, float *result) {
    assert(blockDim.x <= col);
    __shared__ volatile float temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile float shared_vec[THREAD_COUNT_PER_BLOCK];
    KernelPrintLine("gridDim.x=%d,gridDim.y=%d,blockDim.x=%d", gridDim.x, gridDim.y, blockDim.x);
    int matrix_index = blockIdx.y * blockDim.y * blockDim.x +
        blockIdx.x * min(blockDim.y * blockDim.x * gridDim.y, col) + threadIdx.x;

    for (int i = blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x; i < row;
            i += gridDim.x * gridDim.y * blockDim.x) {
        KernelPrintLine("set result as 0, i:%d", i);
        result[i] = 0;
    }

    int vec_index = threadIdx.x + THREAD_COUNT_PER_BLOCK * blockIdx.y;
    if (vec_index >= col) {
        return;
    }
    shared_vec[threadIdx.x] = vec[vec_index];
    __syncthreads();

    if (matrix_index >= row * col) {
        return;
    }
    KernelPrintLine("matrix_index:%d", matrix_index);
    temp[threadIdx.x] = matrix[matrix_index] * shared_vec[threadIdx.x];
    __syncthreads();

    int len = min(blockDim.x, col - blockIdx.y * blockDim.x);
    KernelPrintLine("len:%d", len);
    int last_j = len;
    for (int j = ((len - 1) >> 1) + 1;; j=((j - 1) >>1) + 1) {
        if (threadIdx.x < j) {
            KernelPrintLine("j:%d, last_j;%d", j, last_j);
            temp[threadIdx.x] += threadIdx.x + j < last_j ? temp[threadIdx.x +j] : 0;
        }
        __syncthreads();
        if (j == 1) break;
        last_j = j;
    }

    if (threadIdx.x == 0) {
        KernelPrintLine("gridDim.x:%d ", gridDim.x);
        int result_index = blockIdx.x;
        KernelPrintLine("result_index:%d", result_index);
        atomicAdd(result + result_index, temp[blockDim.x * threadIdx.y]);
    }
}

void N3LDGMultiply(float *matrix, int row, int col, float* vec, float *result) {
    int block_need_y = (col + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK;
    dim3 block_dim(row, block_need_y);

    N3LDGKernelMultiply<<<block_dim, THREAD_COUNT_PER_BLOCK>>>(matrix, row, col, vec, result);
}

void N3LDGKernelTest() {
    CallCuda(cudaSetDevice(1));
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int dima = 1; dima < 100000; dima =dima*2 + 1) {
        for (int dimb = 1024; dimb < min(100000, 2000000000 / dima); dimb = dimb*2+1) {
            pair<int, int> dim(dima, dimb);
            float *W = NewGPUVector(dim.first * dim.second);
            float *x = NewGPUVector(dim.second);
            float *y = NewGPUVector(dim.first);
            float *v = NewGPUVector(dim.first);

            float alpha = 1.0;
            float beta = 0.0;
            cout << "first:" << dim.first << " second:" << dim.second << endl;
            N3LDGMultiply(W, dim.first, dim.second, x, y);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, x, 1, W, dim.second, &beta, v, 1);
            //PrintGPUVector(W, dim.first * dim.second);
            //PrintGPUVector(x, dim.second);
            //PrintGPUVector(y, min(dim.first, 100));
            //PrintGPUVector(v, min(dim.first, 100));
            N3LDGAssertEqual(y, v, dim.first);

            CallCuda(cudaFree(W));
            CallCuda(cudaFree(x));
            CallCuda(cudaFree(y));
            CallCuda(cudaFree(v));
            CallCuda(cudaGetLastError());
        }
    }
}

void Benchmark() {
    CallCuda(cudaSetDevice(1));
    vector<pair<int, int>> dims = {
        pair<int, int>(2, 2000),
        pair<int, int>(5, 2000),
        pair<int, int>(10, 2000),
        pair<int, int>(20, 2000),
        pair<int, int>(50, 2000),
        pair<int, int>(100, 2000),
        pair<int, int>(200, 2000),
        pair<int, int>(500, 2000),
        pair<int, int>(1000, 2000),
        pair<int, int>(2000, 2000),
        pair<int, int>(5000, 2000),
        pair<int, int>(10000, 2000),
        pair<int, int>(20000, 2000),
        pair<int, int>(50000, 2000),
        pair<int, int>(2, 5000),
        pair<int, int>(5, 5000),
        pair<int, int>(10, 5000),
        pair<int, int>(20, 5000),
        pair<int, int>(50, 5000),
        pair<int, int>(100, 5000),
        pair<int, int>(200, 5000),
        pair<int, int>(500, 5000),
        pair<int, int>(1000, 5000),
        pair<int, int>(2000, 5000),
        pair<int, int>(5000, 5000),
        pair<int, int>(10000, 5000),
        pair<int, int>(20000, 5000),
        pair<int, int>(50000, 5000),
        pair<int, int>(2, 10000),
        pair<int, int>(5, 10000),
        pair<int, int>(10, 10000),
        pair<int, int>(20, 10000),
        pair<int, int>(50, 10000),
        pair<int, int>(100, 10000),
        pair<int, int>(200, 10000),
        pair<int, int>(500, 10000),
        pair<int, int>(1000, 10000),
        pair<int, int>(2000, 10000),
        pair<int, int>(5000, 10000),
        pair<int, int>(10000, 10000),
        pair<int, int>(20000, 10000),
        pair<int, int>(2, 100000),
        pair<int, int>(5, 100000),
        pair<int, int>(10, 100000),
        pair<int, int>(20, 100000),
        pair<int, int>(50, 100000),
        pair<int, int>(100, 100000),
        pair<int, int>(200, 100000),
        pair<int, int>(500, 100000),
        pair<int, int>(1000, 100000),
        pair<int, int>(2000, 100000),
    };
    float alpha = 1.0;
    float beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    for (auto &dim : dims) {
        cout << "begin cal" << endl;
        float *W = NewGPUVector(dim.first * dim.second);
        float *x = NewGPUVector(dim.second);
        float *y = NewGPUVector(dim.first);
        float *v = NewGPUVector(dim.first);

        float sum = 0;
        int iter = 1000;
        for (int i = 0; i < iter; ++i) {
            //cout << "first:" << dim.first << " second:" << dim.second << endl;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            //N3LDGMultiply(W, dim.first, dim.second, x, y);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, x, 1, W, dim.second, &beta, v, 1);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float mill;
            cudaEventElapsedTime(&mill, start, stop);
            sum += mill;
            cudaDeviceSynchronize();
        }
        cout << "dim:" << dim.first << "," << dim.second <<  " time:" << sum * 1000 / iter  << endl;

        CallCuda(cudaFree(W));
        CallCuda(cudaFree(x));
        CallCuda(cudaFree(y));
        CallCuda(cudaFree(v));
    }
    CallCuda(cudaGetLastError());
}
