#include <vector>
#include "learn.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_profiler_api.h"
#include <iostream>

using namespace std;

int main() {
    InitGpu();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<cudaStream_t> streams;
    streams.resize(100);
    for (int i = 0; i<100; ++i) {
        cudaStreamCreateWithFlags(&streams.at(i), cudaStreamNonBlocking);
    }

    cudaEventRecord(start);
    for (int n = 0; n < 10000; ++n) {
        for (int i = 0; i<100; ++i) {
            TestFastCpy(streams.at(i));
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float mill = 0;
    cudaEventElapsedTime(&mill, start, stop);
    cout << mill << endl;
    for (int i = 0; i<100; ++i) {
        cudaStreamDestroy(streams.at(i));
    }
    return 0;
}
