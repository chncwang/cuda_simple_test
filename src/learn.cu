#include "learn.h"
#include <cstdlib>
#include <random>
#include <cublas_v2.h>
#include <utility>

using namespace std;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status) << std::endl;
        abort();
    }
}

#define  KernelPrintLine(format, ...)
//{\
//    printf("block:%d thread:x%d,y%d "#format"\n", blockIdx.x, threadIdx.x, threadIdx.y,\
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
            int s = a[i][j] - b[i][j];
            if(s < -0.001 || s > 0.001) {
                printf("i:%d, j:%d, a[i][j]:%f, b[i][j]:%f\n", i, j, a[i][j], b[i][j]);
                assert(false);
            }
        }
    }
}

__global__ void KernelAssertEqual(float *a, float *b, int len) {
    for (int i = threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        int s = a[i] - b[i];
        float m = 0.00001;
        if(s < -m || s > m) {
            printf("i:%d, a[i]:%f, b[i]:%f\n", i, a[i], b[i]);
            assert(false);
        }
    }
}

void N3LDGAssertEqual(float *a, float*b, int len) {
    KernelAssertEqual<<<min(MAX_BLOCK_COUNT, BlockCount(len)), THREAD_COUNT_PER_BLOCK>>>(a, b, len);
}

__global__ void N3LDGKernelMultiply(float *matrix, int row, int col, float* vec, float *result) {
    assert(blockDim.x == col);
    __shared__ volatile float temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile float shared_vec[THREAD_COUNT_PER_BLOCK];
    int temp_index = blockDim.x * threadIdx.y + threadIdx.x;
    int matrix_index_begin = blockIdx.x * blockDim.y * blockDim.x + temp_index;
    if (threadIdx.y == 0) {
        shared_vec[threadIdx.x] = vec[threadIdx.x];
    }
    __syncthreads();
    for (int h = 0;; ++h) {
        int matrix_index = matrix_index_begin + h * gridDim.x * blockDim.x * blockDim.y;
        if (matrix_index >= row * col) {
            break;
        }
        KernelPrintLine("matrix_index:%d temp_index:%d", matrix_index, temp_index);
        temp[temp_index] = matrix[matrix_index] * shared_vec[threadIdx.x];
        __syncthreads();

        int last_i = blockDim.x;
        for (int i = ((blockDim.x - 1) >> 1) + 1;; i=((i - 1) >>1) + 1) {
            if (threadIdx.x < i) {
                KernelPrintLine("temp_index:%d, i:%d, last_i;%d", temp_index, i, last_i);
                temp[temp_index] += threadIdx.x + i < last_i ? temp[temp_index +i] : 0;
            }
            __syncthreads();
            if (i == 1) break;
            last_i = i;
        }

        if (threadIdx.x == 0) {
            KernelPrintLine("h:%d gridDim.x:%d blockDim.y:%d", h, gridDim.x, blockDim.y);
            int result_index = blockDim.y * blockIdx.x + threadIdx.y +
                h * gridDim.x * blockDim.y;
            KernelPrintLine("result_index:%d temp_index:%d", result_index,
                    blockDim.x * threadIdx.y);
            result[result_index] = temp[blockDim.x * threadIdx.y];
        }
    }
}

void N3LDGMultiply(float *matrix, int row, int col, float* vec, float *result) {
    assert(col <= THREAD_COUNT_PER_BLOCK); // TODO

    int row_count_per_block = THREAD_COUNT_PER_BLOCK / col;
    int block_need = (row - 1) / row_count_per_block + 1;
    int block_count = min(block_need, MAX_BLOCK_COUNT);
    dim3 thread_dim(col, row_count_per_block);
    //printf("block_count:%d thread_dim:x%d,y%d\n", block_count, col, row_count_per_block);

    N3LDGKernelMultiply<<<block_count, thread_dim>>>(matrix, row, col, vec, result);
}

void N3LDGKernelTest() {
    CallCuda(cudaSetDevice(1));
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int dima = 50000; dima <= 50000; dima = dima * 4) {
        for (int dimb = 1000; dimb <= 1000; dimb = dimb * 2) {
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
      pair<int, int>(50, 50),
      pair<int, int>(100, 100),
      pair<int, int>(200, 200),
      pair<int, int>(500, 500),
      pair<int, int>(1000, 1000),
      pair<int, int>(2000, 1000),
      pair<int, int>(5000, 1000),
      pair<int, int>(10000, 1000),
      pair<int, int>(20000, 1000),
      pair<int, int>(50000, 1000),
    };

    cublasHandle_t handle;
    cublasCreate(&handle);
    for (auto &dim : dims) {
        cout << "begin cal" << endl;
        float *W = NewGPUVector(dim.first * dim.second);
        float *x = NewGPUVector(dim.second);
        float *y = NewGPUVector(dim.first);
        float *v = NewGPUVector(dim.first);

        float sum = 0;
        int iter = 10000;
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
