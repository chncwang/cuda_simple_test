set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=compute_61;-code=sm_61;-std=c++11;-DVERBOSE;-DEIGEN_USE_GPU;-DHAVE_CUDA;")
if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
        # gcc 4.9 or later versions raise SEGV due to the optimization problem.
        # Use -O1 instead for now.
        list(APPEND CUDA_NVCC_FLAGS "-O1")
    else()
        list(APPEND CUDA_NVCC_FLAGS "-O2")
    endif()
else()
    list(APPEND CUDA_NVCC_FLAGS "-O2")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND CUDA_NVCC_FLAGS "-G")
    list(APPEND CUDA_NVCC_FLAGS "-g")
endif()

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_executable(main main.cc)
cuda_add_library(learn learn.cu)
cuda_add_cublas_to_target(learn)
target_link_libraries(learn ${LIBS})
SET(LIBS ${LIBS} learn)
target_link_libraries(main ${LIBS})
