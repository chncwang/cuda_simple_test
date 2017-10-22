#include "learn.h"
#include "cuda_profiler_api.h"

int main() {
    InitGpu();
    for (int i = 0; i<100000; ++i) {
        TestFastCpy();
    }
    return 0;
}
