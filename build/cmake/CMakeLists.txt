cmake_minimum_required(VERSION 3.5)

project(cuda-min-cut)

#set (CMAKE_CXX_STANDARD 11)

OPTION (USE_OpenMP "Use OpenMP" ON)

find_package(CUDA REQUIRED)

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()


#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;) 

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

cuda_add_executable(
    main
    ${SRC_DIR}/Main.cu
    ${SRC_DIR}/Data.cu
    ${SRC_DIR}/Data_kernel.cu
)

set_target_properties(
    main
    PROPERTIES

    CXX_STANDARD 11
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
)

set(COMMON_OPTIONS "-Wall -Wextra -Wno-reorder")


target_compile_options(
    main
    PRIVATE
    #-g -G
    -Wall -Wextra -Wno-reorder
)

#target_compile_options(
#    main
#    PRIVATE
#    "$<$<CONFIG:RELASE>: ${COMMON_OPTIONS} -O3 -fopenmp>"
#)


#target_link_libraries(
#    main
#    PRIVATE
#    -L/usr/lib/nvidia-396 -lOpenGL -lEGL
#)

