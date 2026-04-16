cmake_minimum_required(VERSION 3.16)
project(cuda_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python for ptx2cpp.py
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# cuda_sim_add_kernel(target ptx_file)
#
# Translate a .ptx file to C++ and add it to an existing target.
# Also generates a .h header with launch declarations.
#
# Usage:
#   add_executable(my_app src/main.cpp src/utils.cpp)
#   cuda_sim_add_kernel(my_app kernels/physics.ptx)
#   cuda_sim_add_kernel(my_app kernels/sort.ptx)
# ---------------------------------------------------------------------------
function(cuda_sim_add_kernel TARGET PTX_FILE)
    get_filename_component(PTX_NAME ${PTX_FILE} NAME_WE)
    set(GENERATED_CPP "${CMAKE_CURRENT_BINARY_DIR}/${PTX_NAME}_cpu.cpp")
    set(GENERATED_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PTX_NAME}_cpu.h")
    set(PTX2CPP "${CMAKE_SOURCE_DIR}/tools/ptx2cpp.py")

    # Custom command: ptx → cpp + header
    add_custom_command(
        OUTPUT ${GENERATED_CPP} ${GENERATED_HDR}
        COMMAND ${Python3_EXECUTABLE} ${PTX2CPP}
            ${CMAKE_CURRENT_SOURCE_DIR}/${PTX_FILE}
            -o ${GENERATED_CPP}
            -H ${GENERATED_HDR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${PTX_FILE} ${PTX2CPP}
        COMMENT "ptx2cpp: ${PTX_FILE} → ${PTX_NAME}_cpu.cpp + .h"
    )

    # Add generated source to target
    target_sources(${TARGET} PRIVATE ${GENERATED_CPP})

    # Add include paths for generated header + cuda_sim headers
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/include/compat
        ${CMAKE_SOURCE_DIR}/include)
endfunction()

# ---------------------------------------------------------------------------
# cuda_sim_add_executable(target ptx_file source1 [source2 ...])
#
# Convenience: create executable + add kernel in one call.
# For simple single-kernel examples.
# ---------------------------------------------------------------------------
function(cuda_sim_add_executable TARGET PTX_FILE)
    add_executable(${TARGET} ${ARGN})
    cuda_sim_add_kernel(${TARGET} ${PTX_FILE})
endfunction()

# ---------------------------------------------------------------------------
# cuda_sim_preprocess(input_file output_file)
#
# Convert <<<>>> launch syntax to _launch() calls.
# Use this on host .cpp files that use CUDA kernel launch syntax.
#
# Usage:
#   cuda_sim_preprocess(src/main_cuda.cpp ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)
#   add_executable(my_app ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)
#   cuda_sim_add_kernel(my_app kernel.ptx)
# ---------------------------------------------------------------------------
function(cuda_sim_preprocess INPUT_FILE OUTPUT_FILE)
    set(PREPROCESS_PY "${CMAKE_SOURCE_DIR}/tools/cuda_preprocess.py")
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${Python3_EXECUTABLE} ${PREPROCESS_PY}
            ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}
            -o ${OUTPUT_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE} ${PREPROCESS_PY}
        COMMENT "cuda_preprocess: ${INPUT_FILE} (<<<>>> → _launch)"
    )
endfunction()

# Examples
add_subdirectory(examples/vector_add)
add_subdirectory(examples/histogram)
add_subdirectory(examples/reduction)
add_subdirectory(examples/matrix_mul)
add_subdirectory(examples/saxpy)
add_subdirectory(examples/warp_reduce)

# Tests
enable_testing()
add_test(NAME test_vector_add COMMAND vector_add_example)
add_test(NAME test_histogram COMMAND histogram_example)
add_test(NAME test_reduction COMMAND reduction_example)
add_test(NAME test_matrix_mul COMMAND matrix_mul_example)
add_test(NAME test_saxpy COMMAND saxpy_example)
add_test(NAME test_warp_reduce COMMAND warp_reduce_example)
add_test(NAME test_ptx_parser
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_ptx_parser.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
