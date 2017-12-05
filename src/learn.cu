#include "learn.h"
#include <cstdlib>
#include <random>
#include <cublas_v2.h>
#include <utility>
#include "cuPrintf.cu"
#include <tuple>

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

//#define KERNEL_LOG

#ifdef KERNEL_LOG
#define  KernelPrintLine(format, ...)\
{\
    cuPrintf("block:x=%d,y=%d thread:x=%d,y=%d "#format"\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,\
            __VA_ARGS__);\
}
#else
#define KernelPrintLine(format, ...)
#endif

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
        vec[i] = ((i + gridDim.x + blockDim.x + blockIdx.z) * 1664525 + 1013904223) % 100000000 / 5000000.0 - 10;
    }
}

void InitGPUVector(float *vec, int dim) {
    GlobalAssignVector<<<BlockCount(dim), THREAD_COUNT_PER_BLOCK>>>(vec, dim);
}

__global__ void GlobalPrintVector(float *vec, int dim) {
    cuPrintf("GlobalPrintVector:");
    for (int i = 0; i < dim; ++i) {
        cuPrintf("%f\n", vec[i]);
    }
}

void PrintGPUVector(float *vec, int dim) {
    GlobalPrintVector<<<1, 1>>>(vec, dim);
    cudaDeviceSynchronize();
}

__global__ void N3LDGKernelPrintVector(void **vec, int dim) {
    cuPrintf("N3LDGKernelPrintVector: dim %d, vec %p\n", dim, vec);
    for (int i = 0; i < dim; ++i) {
        cuPrintf("%p, ", vec[i]);
    }
    cuPrintf("\n");
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
    cuPrintfRestrict(CUPRINTF_UNRESTRICTED, CUPRINTF_UNRESTRICTED);
    int i = blockIdx.x;
    for (int j = threadIdx.x; j < len; j += THREAD_COUNT_PER_BLOCK) {
        float s = a[i][j] - b[i][j];
        float ratio = s / a[i][j];
        if((s < -0.001 || s > 0.001) && (ratio < -0.001 || ratio > 0.001)) {
            printf("i:%d, j:%d, a[i][j]:%f, b[i][j]:%f, s:%f\n", i, j, a[i][j], b[i][j], s);
            printf("len:%d count:%d\n", len, count);
        }
    }
}

void N3LDGAssertEqual(float **a, float **b, int len, int count) {
    KernelAssertEqual<<<count, THREAD_COUNT_PER_BLOCK>>>(a, b, len, count);
}

__global__ void KernelAssertEqual(float *a, float *b, int len) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < len; i += gridDim.x * blockDim.x) {
        int s = (a[i] - b[i]) / a[i];
        float m = 0.001;
        if(s < -m || s > m) {
            cuPrintf("i:%d, a[i]:%f, b[i]:%f\n", i, a[i], b[i]);
        }
    }
}

void N3LDGAssertEqual(float *a, float*b, int len) {
    KernelAssertEqual<<<min(MAX_BLOCK_COUNT, BlockCount(len)), THREAD_COUNT_PER_BLOCK>>>(a, b, len);
}

__global__ void N3LDGKernelMultiplySmallVector(float *matrix, int row, int col, float* vec, float *result) {
    assert(blockDim.x == col);
    __shared__ volatile float temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile float shared_vec[THREAD_COUNT_PER_BLOCK];
    int temp_index = blockDim.x * threadIdx.y + threadIdx.x;
    int matrix_index_begin = blockIdx.x * blockDim.y * blockDim.x + temp_index;
    //KernelPrintLine("matrix_index_begin:%d", matrix_index_begin);
    if (threadIdx.y == 0) {
        shared_vec[threadIdx.x] = vec[threadIdx.x];
    }
    __syncthreads();
    for (int i = 0;; ++i) {
        int matrix_index = matrix_index_begin + i * gridDim.x * blockDim.x * blockDim.y;
        if (matrix_index >= row * col) {
            break;
        }
        //KernelPrintLine("matrix_index:%d temp_index:%d", matrix_index, temp_index);
        temp[temp_index] = matrix[matrix_index] * shared_vec[threadIdx.x];
        __syncthreads();

        int last_j = blockDim.x;
        for (int j = ((blockDim.x - 1) >> 1) + 1;; j=((j - 1) >>1) + 1) {
            if (threadIdx.x < j) {
                //KernelPrintLine("temp_index:%d, j:%d, last_j;%d", temp_index, j, last_j);
                temp[temp_index] += threadIdx.x + j < last_j ? temp[temp_index +j] : 0;
            }
            __syncthreads();
            if (j == 1) break;
            last_j = j;
        }

        if (threadIdx.x == 0) {
            //KernelPrintLine("i:%d gridDim.x:%d blockDim.y:%d", i, gridDim.x, blockDim.y);
            int result_index = blockDim.y * blockIdx.x + threadIdx.y +
                i * gridDim.x * blockDim.y;
            //KernelPrintLine("result_index:%d temp_index:%d", result_index,
             //       blockDim.x * threadIdx.y);
            result[result_index] = temp[blockDim.x * threadIdx.y];
        }
    }
}


__global__ void N3LDGKernelMultiplyLargeVector(float *matrix, int row, int col, float* vec, float *result) {
    assert(blockDim.x <= col);
    __shared__ volatile float temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile float shared_vec[THREAD_COUNT_PER_BLOCK];
    KernelPrintLine("gridDim.x=%d,gridDim.y=%d,blockDim.x=%d", gridDim.x, gridDim.y, blockDim.x);
    int matrix_index = blockIdx.y * blockDim.x +
        blockIdx.x * min(blockDim.x * gridDim.y, col) + threadIdx.x;

    for (int i = blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x + threadIdx.x; i < row;
            i += gridDim.x * gridDim.y * blockDim.x) {
        KernelPrintLine("set result as 0, i:%d", i);
        result[i] = 0;
    }

    int vec_index = threadIdx.x + THREAD_COUNT_PER_BLOCK * blockIdx.y;
    if (vec_index < col) {
        shared_vec[threadIdx.x] = vec[vec_index];
    }
    __syncthreads();

    KernelPrintLine("matrix_index:%d", matrix_index);
    if (matrix_index < row * col) {
        temp[threadIdx.x] = matrix[matrix_index] * shared_vec[threadIdx.x];
    }
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
        atomicAdd(result + result_index, temp[0]);
    }
}

void N3LDGMultiply(float *matrix, int row, int col, float* vec, float *result) {
    if (col <= THREAD_COUNT_PER_BLOCK) {
        int row_count_per_block = THREAD_COUNT_PER_BLOCK / col;
        int block_need = (row - 1) / row_count_per_block + 1;
        int block_count = min(block_need, MAX_BLOCK_COUNT);
        dim3 thread_dim(col, row_count_per_block);
        N3LDGKernelMultiplySmallVector<<<block_count, thread_dim>>>(matrix, row, col, vec, result);
    } else {
        int block_need_y = (col + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK;
        dim3 block_dim(row, block_need_y);
        N3LDGKernelMultiplyLargeVector<<<block_dim, THREAD_COUNT_PER_BLOCK>>>(matrix, row, col, vec, result);
    }
}

__global__ void N3LDGKernelMultiplySmallVectorBatch(float *matrix, int row, int col, float** vectors, int count, float **results) {
    assert(blockDim.x == col);
    __shared__ volatile float temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile float shared_vec[THREAD_COUNT_PER_BLOCK];
    int temp_index = blockDim.x * threadIdx.y + threadIdx.x;
    int matrix_index_begin = blockIdx.x * blockDim.y * blockDim.x + temp_index;
    KernelPrintLine("matrix_index_begin:%d", matrix_index_begin);

    int matrix_index_upper_bound = row * col;

    for (int h = blockIdx.y; h < count; h +=gridDim.y) {
        KernelPrintLine("h:%d gridDim.y:%d", h, gridDim.y);
        __syncthreads();
        if (threadIdx.y == 0) {
            shared_vec[threadIdx.x] = vectors[h][threadIdx.x];
        }
        __syncthreads();
        for (int i = 0;; ++i) {
            int matrix_index = matrix_index_begin + i * gridDim.x * blockDim.x * blockDim.y;
            KernelPrintLine("matrix_index:%d temp_index:%d", matrix_index, temp_index);
            __syncthreads();
            if (matrix_index < matrix_index_upper_bound) {
                temp[temp_index] = matrix[matrix_index] * shared_vec[threadIdx.x];
            }
            __syncthreads();

            int last_j = blockDim.x;
            for (int j = ((blockDim.x - 1) >> 1) + 1;; j=((j - 1) >>1) + 1) {
                if (threadIdx.x < j && matrix_index < matrix_index_upper_bound) {
                    KernelPrintLine("temp_index:%d, j:%d, last_j;%d", temp_index, j, last_j);
                    temp[temp_index] += threadIdx.x + j < last_j ? temp[temp_index +j] : 0;
                }
                __syncthreads();
                if (j == 1) break;
                last_j = j;
            }

            if (threadIdx.x == 0 && matrix_index < matrix_index_upper_bound) {
                KernelPrintLine("i:%d gridDim.x:%d blockDim.y:%d", i, gridDim.x, blockDim.y);
                int result_index = blockDim.y * blockIdx.x + threadIdx.y +
                    i * gridDim.x * blockDim.y;
                KernelPrintLine("h:%d result_index:%d temp_index:%d", h, result_index,
                        blockDim.x * threadIdx.y);
                results[h][result_index] = temp[blockDim.x * threadIdx.y];
            }

            if (__syncthreads_and(matrix_index >= matrix_index_upper_bound)) {
                break;
            }
        }
    }
}

__global__ void N3LDGKernelMultiplyLargeVectorBatch(float *matrix, int row, int col, float** vectors,
        int count,
        float **results) {
    assert(blockDim.x <= col);
    __shared__ volatile float temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile float shared_vec[THREAD_COUNT_PER_BLOCK];
    KernelPrintLine("gridDim.x=%d,gridDim.y=%d,blockDim.x=%d", gridDim.x, gridDim.y, blockDim.x);
    int matrix_index = blockIdx.y * blockDim.x +
        blockIdx.x * min(blockDim.x * gridDim.y, col) + threadIdx.x;

    float * result = results[blockIdx.z];
    for (int i = blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x + threadIdx.x; i < row;
            i += gridDim.x * gridDim.y * blockDim.x) {
        KernelPrintLine("set result as 0, i:%d", i);
        result[i] = 0;
    }

    int vec_index = threadIdx.x + THREAD_COUNT_PER_BLOCK * blockIdx.y;
    if (vec_index < col) {
        shared_vec[threadIdx.x] = vectors[blockIdx.z][vec_index];
    }
    __syncthreads();

    KernelPrintLine("matrix_index:%d", matrix_index);
    if (matrix_index < col * row) {
        temp[threadIdx.x] = matrix[matrix_index] * shared_vec[threadIdx.x];
    }
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
        atomicAdd(result + result_index, temp[0]);
    }
}

void N3LDGMultiplyVectorBatch(float *matrix, int row, int col, float** vectors, int count, float **results) {
    //printf("col:%d, THREAD_COUNT_PER_BLOCK:%d", col, THREAD_COUNT_PER_BLOCK);
    if (col <= THREAD_COUNT_PER_BLOCK) {
        int row_count_per_block = THREAD_COUNT_PER_BLOCK / col;
        dim3 thread_dim(col, row_count_per_block);
        //printf("col:%d row_count_per_block:%d\n", col, row_count_per_block);
        int block_need_x = (row - 1) / row_count_per_block + 1;
        int block_count_x = min(block_need_x, MAX_BLOCK_COUNT);
        int block_count_y = (MAX_BLOCK_COUNT + block_count_x - 1) / block_count_x;
        //printf("block_count_x:%d, block_count_y:%d\n", block_count_x, block_count_y);
        dim3 block_dim(block_count_x, block_count_y);
        N3LDGKernelMultiplySmallVectorBatch<<<block_dim, thread_dim>>>(matrix, row, col, vectors,
                count, results);
    } else {
        int block_need_y = (col + THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK;
        int block_count_per_vector = row * block_need_y;
        int block_count_z = count;
        //printf("row:%d, block_need_y:%d, block_count_z:%d\n", row, block_need_y, block_count_z);
        dim3 block_dim(row, block_need_y, block_count_z);
        N3LDGKernelMultiplyLargeVectorBatch<<<block_dim, THREAD_COUNT_PER_BLOCK>>>(matrix,
                row, col, vectors, count,results);
    }
}

void N3LDGKernelTestSingleProduct() {
    CallCuda(cudaSetDevice(1));
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int dima = 1; dima < 100000; dima = 2 * dima + 1) {
        for (int dimb = 1; dimb < min(THREAD_COUNT_PER_BLOCK, 2000000000 / dima); dimb = dimb * 2 + 1) {
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

void N3LDGKernelTest() {
    CallCuda(cudaSetDevice(1));
    CallCuda(cudaPrintfInit());
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int count = 1; count <= 4000; count = count * 4 + 1) {
        for (int dima = 1; dima <= 200000; dima = dima * 2 + 1) {
            for (int dimb = 1; dimb <= min(200000, 1000000000 / dima / count);
                    dimb = dimb * 2 + 1) {
                pair<int, int> dim(dima, dimb);
                float *W = NewGPUVector(dim.first * dim.second);
                float **xs = NewGPUVectors(count, dim.second);
                float **ys = NewGPUVectors(count, dim.first);
                float **vs = NewGPUVectors(count, dim.first);
                float **Ws = (float**)malloc(sizeof(float*) * count);
                for (int i = 0; i < count; ++i) {
                    Ws[i] = W;
                }
                float **gpu_xs, **gpu_ys, **gpu_vs, **gpu_Ws;
                CallCuda(cudaMalloc((void**)&gpu_xs, sizeof(float*) * count));
                CallCuda(cudaMalloc((void**)&gpu_ys, sizeof(float*) * count));
                CallCuda(cudaMalloc((void**)&gpu_vs, sizeof(float*) * count));
                CallCuda(cudaMalloc((void**)&gpu_Ws, sizeof(float*) * count));
                CallCuda(cudaMemcpy(gpu_xs, xs, sizeof(float*) * count, cudaMemcpyHostToDevice));
                CallCuda(cudaMemcpy(gpu_ys, ys, sizeof(float*) * count, cudaMemcpyHostToDevice));
                CallCuda(cudaMemcpy(gpu_vs, vs, sizeof(float*) * count, cudaMemcpyHostToDevice));
                CallCuda(cudaMemcpy(gpu_Ws, Ws, sizeof(float*) * count, cudaMemcpyHostToDevice));

                float alpha = 1.0;
                float beta = 0.0;
                cout << "count:" << count <<"first:" << dim.first << " second:" << dim.second << endl;
                N3LDGMultiplyVectorBatch(W, dim.first, dim.second, gpu_xs, count, gpu_ys);
                //N3LDGAssertEqual(ys[0], ys[1], dim.second);
                //cudaDeviceSynchronize();
                //printf("ys[0], ys[1] assert finished\n");
                //CallCuda(cudaGetLastError());
                cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, (const float**)gpu_xs, 1 ,(const float **)gpu_Ws, dim.second, &beta, gpu_vs, 1, count);
                N3LDGAssertEqual(gpu_ys, gpu_vs, dim.first, count);
                //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, xs[1],
                //        1, W, dim.second, &beta, vs[1], 1);
                //PrintGPUVector(vs[1], 10);
                //N3LDGMultiply(W, dim.first, dim.second, xs[1], ys[1]);
                //PrintGPUVector(ys[1], 10);

                CallCuda(cudaFree(gpu_Ws));
                CallCuda(cudaFree(gpu_xs));
                CallCuda(cudaFree(gpu_ys));
                CallCuda(cudaFree(gpu_vs));
                CallCuda(cudaFree(W));
                for (int i = 0; i < count; ++i) {
                    CallCuda(cudaFree(xs[i]));
                    CallCuda(cudaFree(ys[i]));
                    CallCuda(cudaFree(vs[i]));
                }
                cudaPrintfDisplay(stdout, true);
            }
        }
    }
    cudaPrintfEnd();
}

void BatchMultiplyBenchmark() {
    CallCuda(cudaSetDevice(1));
    CallCuda(cudaPrintfInit());
    cublasHandle_t handle;
    cublasCreate(&handle);
    vector<tuple<int, int, int>> dims = {
        tuple<int,int,int>(10, 100, 100),
        tuple<int,int,int>(100, 100, 100),
        tuple<int,int,int>(1000, 100, 100),
        tuple<int,int,int>(10, 200, 200),
        tuple<int,int,int>(100, 200, 200),
        tuple<int,int,int>(1000, 200, 200),
        tuple<int,int,int>(10, 500, 500),
        tuple<int,int,int>(100, 500, 500),
        tuple<int,int,int>(1000, 500, 500),
        tuple<int,int,int>(10, 1000, 1000),
        tuple<int,int,int>(100, 1000, 1000),
        tuple<int,int,int>(10, 2000, 2000),
    };

    for (auto &d : dims) {
        int count = get<0>(d);
        pair<int, int> dim(get<1>(d), get<2>(d));
        float *W = NewGPUVector(dim.first * dim.second);
        float **xs = NewGPUVectors(count, dim.second);
        float **ys = NewGPUVectors(count, dim.first);
        float **vs = NewGPUVectors(count, dim.first);
        float **Ws = (float**)malloc(sizeof(float*) * count);
        for (int i = 0; i < count; ++i) {
            Ws[i] = W;
        }
        float **gpu_xs, **gpu_ys, **gpu_vs, **gpu_Ws;
        CallCuda(cudaMalloc((void**)&gpu_xs, sizeof(float*) * count));
        CallCuda(cudaMalloc((void**)&gpu_ys, sizeof(float*) * count));
        CallCuda(cudaMalloc((void**)&gpu_vs, sizeof(float*) * count));
        CallCuda(cudaMalloc((void**)&gpu_Ws, sizeof(float*) * count));
        CallCuda(cudaMemcpy(gpu_vs, vs, sizeof(float*) * count, cudaMemcpyHostToDevice));
        CallCuda(cudaMemcpy(gpu_Ws, Ws, sizeof(float*) * count, cudaMemcpyHostToDevice));
            CallCuda(cudaMemcpy(gpu_xs, xs, sizeof(float*) * count, cudaMemcpyHostToDevice));
            CallCuda(cudaMemcpy(gpu_ys, ys, sizeof(float*) * count, cudaMemcpyHostToDevice));

        float sum = 0;
        int iter = 1000;
        for (int i = 0; i < iter; ++i) {
            float alpha = 1.0;
            float beta = 0.0;
            //cout << "first:" << dim.first << " second:" << dim.second << endl;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            N3LDGMultiplyVectorBatch(W, dim.first, dim.second, gpu_xs, count, gpu_ys);
            //cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, (const float**)gpu_xs, 1 ,(const float **)gpu_Ws, dim.second, &beta, gpu_vs, 1, count);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float mill;
            cudaEventElapsedTime(&mill, start, stop);
            sum += mill;
            cudaDeviceSynchronize();
        }
        cout << "count:" << count<< "dim:" << dim.first << "," << dim.second <<  " time:" << sum * 1000 / iter  << endl;

        //N3LDGAssertEqual(ys[0], ys[1], dim.second);
        //cudaDeviceSynchronize();
        //printf("ys[0], ys[1] assert finished\n");
        //CallCuda(cudaGetLastError());
        //cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, (const float**)gpu_xs, 1 ,(const float **)gpu_Ws, dim.second, &beta, gpu_vs, 1, count);
        //N3LDGAssertEqual(gpu_ys, gpu_vs, dim.first, count);
        //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, xs[1],
        //        1, W, dim.second, &beta, vs[1], 1);
        //PrintGPUVector(vs[1], 10);
        //N3LDGMultiply(W, dim.first, dim.second, xs[1], ys[1]);
        //PrintGPUVector(ys[1], 10);

        CallCuda(cudaFree(gpu_Ws));
        CallCuda(cudaFree(gpu_xs));
        CallCuda(cudaFree(gpu_ys));
        CallCuda(cudaFree(gpu_vs));
        CallCuda(cudaFree(W));
        for (int i = 0; i < count; ++i) {
            CallCuda(cudaFree(xs[i]));
            CallCuda(cudaFree(ys[i]));
            CallCuda(cudaFree(vs[i]));
        }
        cudaPrintfDisplay(stdout, true);
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
        pair<int, int>(2000, 2000),
        pair<int, int>(5000, 5000),
        pair<int, int>(10000, 10000),
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
            N3LDGMultiply(W, dim.first, dim.second, x, y);
            //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim.first, dim.second, &alpha, x, 1, W, dim.second, &beta, v, 1);
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
