# CudaSimCompile.cmake — Compile .cu files using cuda_sim (CPU simulation)
#
# Provides cuda_sim_add_library() as a drop-in replacement for cuda_add_library().
# .cu files are split into device code (→ nvcc -ptx → ptx2cpp.py → g++) and
# host code (→ g++ with cuda_sim compat headers).
#
# Usage:
#   set(CUDA_SIM_ROOT /path/to/cuda_sim)
#   include(${CUDA_SIM_ROOT}/cmake/CudaSimCompile.cmake)
#   cuda_sim_add_library(mylib STATIC kernel.cu host.cpp)
#
# Requirements:
#   - nvcc available (for .cu → PTX compilation)
#   - python3 available (for ptx2cpp.py)
#   - CUDA_SIM_ROOT set to cuda_sim directory

if(NOT CUDA_SIM_ROOT)
    message(FATAL_ERROR "CUDA_SIM_ROOT not set")
endif()

# Find tools
find_program(CUDA_SIM_NVCC nvcc)
find_program(CUDA_SIM_PYTHON3 python3)

if(NOT CUDA_SIM_PYTHON3)
    message(FATAL_ERROR "python3 not found (required for ptx2cpp.py)")
endif()

set(CUDA_SIM_PTX2CPP "${CUDA_SIM_ROOT}/tools/ptx2cpp.py")
set(CUDA_SIM_PREPROCESS "${CUDA_SIM_ROOT}/tools/cuda_preprocess.py")

# cuda_sim_compile_cu(OUTPUT_VAR source.cu)
# Compiles a .cu file through the cuda_sim pipeline:
#   1. nvcc -ptx → .ptx
#   2. ptx2cpp.py → _cpu.cpp
#   3. Return generated .cpp path in OUTPUT_VAR
function(cuda_sim_compile_cu OUTPUT_VAR CU_FILE)
    get_filename_component(CU_NAME "${CU_FILE}" NAME_WE)
    get_filename_component(CU_DIR "${CU_FILE}" DIRECTORY)
    if(NOT IS_ABSOLUTE "${CU_FILE}")
        set(CU_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${CU_FILE}")
    endif()

    set(PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CU_NAME}.ptx")
    set(CPU_CPP "${CMAKE_CURRENT_BINARY_DIR}/${CU_NAME}_cpu.cpp")
    set(CPU_HDR "${CMAKE_CURRENT_BINARY_DIR}/${CU_NAME}_cpu.h")

    if(CUDA_SIM_NVCC)
        # Step 1: nvcc -ptx
        set(NVCC_FLAGS --ptx -O2)
        if(CUDA_SIM_COMPUTE)
            list(APPEND NVCC_FLAGS "-arch=compute_${CUDA_SIM_COMPUTE}")
        else()
            list(APPEND NVCC_FLAGS "-arch=compute_75")
        endif()

        add_custom_command(
            OUTPUT "${PTX_FILE}"
            COMMAND ${CUDA_SIM_NVCC} ${NVCC_FLAGS}
                -I"${CU_DIR}"
                "${CU_FILE}" -o "${PTX_FILE}"
            DEPENDS "${CU_FILE}"
            COMMENT "cuda_sim: nvcc -ptx ${CU_NAME}.cu"
            VERBATIM
        )

        # Step 2: ptx2cpp.py
        add_custom_command(
            OUTPUT "${CPU_CPP}" "${CPU_HDR}"
            COMMAND ${CUDA_SIM_PYTHON3} "${CUDA_SIM_PTX2CPP}"
                "${PTX_FILE}" -o "${CPU_CPP}" -H "${CPU_HDR}"
            DEPENDS "${PTX_FILE}" "${CUDA_SIM_PTX2CPP}"
            COMMENT "cuda_sim: ptx2cpp.py ${CU_NAME}.ptx"
            VERBATIM
        )

        set(${OUTPUT_VAR} "${CPU_CPP}" PARENT_SCOPE)
    else()
        message(WARNING "nvcc not found — .cu file ${CU_NAME}.cu will not be compiled")
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()

# cuda_sim_add_library(name STATIC|SHARED source1.cu source2.cpp ...)
# Drop-in replacement for cuda_add_library.
# .cu files go through nvcc → ptx2cpp.py pipeline.
# .cpp files are compiled directly with cuda_sim compat headers.
function(cuda_sim_add_library LIB_NAME LIB_TYPE)
    set(ALL_SOURCES "")
    set(CU_TARGETS "")

    foreach(SRC ${ARGN})
        get_filename_component(EXT "${SRC}" EXT)
        if("${EXT}" STREQUAL ".cu")
            cuda_sim_compile_cu(COMPILED_CPP "${SRC}")
            if(COMPILED_CPP)
                list(APPEND ALL_SOURCES "${COMPILED_CPP}")
            endif()
        else()
            list(APPEND ALL_SOURCES "${SRC}")
        endif()
    endforeach()

    add_library(${LIB_NAME} ${LIB_TYPE} ${ALL_SOURCES})

    # Add cuda_sim include paths
    target_include_directories(${LIB_NAME} PUBLIC
        "${CUDA_SIM_ROOT}/include/compat"
        "${CUDA_SIM_ROOT}/include"
    )

    # Add current binary dir for generated headers
    target_include_directories(${LIB_NAME} PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # dlopen/dlsym for JIT
    if(UNIX)
        target_link_libraries(${LIB_NAME} dl)
    endif()
endfunction()
