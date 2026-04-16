cmake_minimum_required(VERSION 3.16)
project(cuda_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python for ptx2cpp.py
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Helper function: translate PTX to C++ and build
# Usage: cuda_sim_add_executable(target ptx_file main.cpp [extra.cpp ...])
function(cuda_sim_add_executable TARGET PTX_FILE)
    get_filename_component(PTX_NAME ${PTX_FILE} NAME_WE)
    set(GENERATED_CPP "${CMAKE_CURRENT_BINARY_DIR}/${PTX_NAME}_cpu.cpp")
    set(PTX2CPP "${CMAKE_SOURCE_DIR}/tools/ptx2cpp.py")

    # Custom command: ptx → cpp
    add_custom_command(
        OUTPUT ${GENERATED_CPP}
        COMMAND ${Python3_EXECUTABLE} ${PTX2CPP} ${CMAKE_CURRENT_SOURCE_DIR}/${PTX_FILE} -o ${GENERATED_CPP}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${PTX_FILE} ${PTX2CPP}
        COMMENT "Translating ${PTX_FILE} → ${PTX_NAME}_cpu.cpp"
    )

    # Build executable from host sources + generated kernel
    add_executable(${TARGET} ${ARGN} ${GENERATED_CPP})
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/compat
        ${CMAKE_SOURCE_DIR}/include)
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
