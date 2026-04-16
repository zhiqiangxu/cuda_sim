# CudaSimConfig.cmake
#
# Include this in your project to get cuda_sim functions.
#
# Usage in user's CMakeLists.txt:
#   list(APPEND CMAKE_MODULE_PATH "/path/to/cuda_sim/cmake")
#   include(CudaSimConfig)
#
# Or simply add_subdirectory(cuda_sim) if it's a subdirectory.
#
# Provides:
#   cuda_sim_add_kernel(target ptx_file)
#   cuda_sim_add_executable(target ptx_file sources...)
#   cuda_sim_preprocess(input_file output_file)
#   CUDA_SIM_INCLUDE_DIR  — path to include/compat and include/

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CUDA_SIM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(CUDA_SIM_INCLUDE_COMPAT "${CUDA_SIM_ROOT}/include/compat")
set(CUDA_SIM_INCLUDE "${CUDA_SIM_ROOT}/include")
set(CUDA_SIM_PTX2CPP "${CUDA_SIM_ROOT}/tools/ptx2cpp.py")
set(CUDA_SIM_PREPROCESS "${CUDA_SIM_ROOT}/tools/cuda_preprocess.py")

function(cuda_sim_add_kernel TARGET PTX_FILE)
    get_filename_component(PTX_NAME ${PTX_FILE} NAME_WE)
    set(GENERATED_CPP "${CMAKE_CURRENT_BINARY_DIR}/${PTX_NAME}_cpu.cpp")
    set(GENERATED_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PTX_NAME}_cpu.h")

    add_custom_command(
        OUTPUT ${GENERATED_CPP} ${GENERATED_HDR}
        COMMAND ${Python3_EXECUTABLE} ${CUDA_SIM_PTX2CPP}
            ${CMAKE_CURRENT_SOURCE_DIR}/${PTX_FILE}
            -o ${GENERATED_CPP}
            -H ${GENERATED_HDR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${PTX_FILE} ${CUDA_SIM_PTX2CPP}
        COMMENT "ptx2cpp: ${PTX_FILE} → ${PTX_NAME}_cpu.cpp + .h"
    )

    target_sources(${TARGET} PRIVATE ${GENERATED_CPP})
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CUDA_SIM_INCLUDE_COMPAT}
        ${CUDA_SIM_INCLUDE})
endfunction()

function(cuda_sim_add_executable TARGET PTX_FILE)
    add_executable(${TARGET} ${ARGN})
    cuda_sim_add_kernel(${TARGET} ${PTX_FILE})
endfunction()

function(cuda_sim_preprocess INPUT_FILE OUTPUT_FILE)
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${Python3_EXECUTABLE} ${CUDA_SIM_PREPROCESS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}
            -o ${OUTPUT_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE} ${CUDA_SIM_PREPROCESS}
        COMMENT "cuda_preprocess: ${INPUT_FILE} (<<<>>> → _launch)"
    )
endfunction()
