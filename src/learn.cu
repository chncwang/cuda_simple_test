#include "learn.h"
#include <cstdlib>
#include <random>
#include <cublas_v2.h>

using namespace std;

cudaStream_t streams[1000];

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
    int thread_count_per_arr = (len / THREAD_COUNT_PER_WRAP + 1) *
        THREAD_COUNT_PER_WRAP;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x / thread_count_per_arr;
    int count_index = index / thread_count_per_arr;
    for (int i = count_index; i < count; i += step) {
        int arr_index = index % thread_count_per_arr;
        if (arr_index < len) {
            dest[i][arr_index] = tanh(src[i][arr_index]);
        }
    }
}

void N3LDGTanh(float **src, float **dest, int len, int count) {
    assert(len <= MAX_BLOCK_COUNT * THREAD_COUNT_PER_BLOCK);
    int block_count = BlockCount(((len - 1) / THREAD_COUNT_PER_WRAP + 1) *
            THREAD_COUNT_PER_WRAP * count);
    N3LDGKernelTanh<<<block_count, THREAD_COUNT_PER_BLOCK>>>(src, dest, len, count);
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
    N3LDGKernelTanh<<<BlockCount(len * count), THREAD_COUNT_PER_BLOCK>>>(src, dest, len, count);
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

void N3LDGKernelTanhByCopy(float **src, float **dest, int len, int count) {
    float *src_big, *dest_big;
    int size = len * count * sizeof(float);
    int per_size = len * sizeof(float);
    assert(cudaMalloc((void**)&src_big, size) == cudaSuccess);
    assert(cudaMalloc((void**)&dest_big, size) == cudaSuccess);
    for (int i = 0; i<count; ++i) {
        Copy<<<BlockCount(per_size), THREAD_COUNT_PER_BLOCK, 0, streams[i]>>>(src[i], src_big + i * per_size, per_size);
    }
    cudaFree(src_big);
    cudaFree(dest_big);
}

float **ToGpuVectorArray(float** vec, int len) {
    void *result;
    int size = len * sizeof(float*);
    assert(cudaSuccess == cudaMalloc(&result, size));
    assert(cudaMemcpy(result, vec, size, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    return (float**)result;
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

void Benchmark() {
    assert(cudaSetDevice(1) == cudaSuccess);
    std::vector<int> dims = {100, 1000};
    std::vector<int> counts = {10, 100, 1000};

    for (int i = 0; i<1000; ++i) {
        assert(cudaStreamCreate(streams + i) == cudaSuccess);
    }

    float alpha = 1.0;
    for (auto dim : dims) {
        for (auto count : counts) {
            float** gpu_vec_a = NewGPUVectors(count, dim);
            float** gpu_vec_b = NewGPUVectors(count, dim);
            cout << "begin cal" << endl;
            float sum = 0;
            int iter = 1;
            for (int i = 0; i < iter; ++i) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                for (int j = 0; j < count; ++j) {
                    N3LDGKernelTanhByCopy(gpu_vec_a, gpu_vec_b, dim, count);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float mill;
                cudaEventElapsedTime(&mill, start, stop);
                sum += mill;
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            cout << "dim:" << dim << " count:" <<count << " time:" << sum * 1000 / iter  << endl;
        }
    }
}
