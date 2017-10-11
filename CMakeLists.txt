project(N3LDG-classifier-gpu)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

if(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w -msse3 -funroll-loops  -std=c++11 -O1 -pg" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -w -std=c++11 -Ofast -march=native" )
endif()

if(USE_FLOAT)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FLOAT")
endif()

add_definitions(-DUSE_GPU)
include_directories(/usr/local/cuda-8.0/samples/common/inc)
find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
message("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
#list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
#list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
set(LIBS ${LIBS} ${CUDA_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_HAS_CUDA_FP16 -DEIGEN_USE_GPU")

include_directories(cnmem/include)

add_subdirectory(cnmem)
set(LIBS ${LIBS} cnmem)
add_subdirectory(src)
