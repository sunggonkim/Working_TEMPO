cmake_minimum_required(VERSION 3.21)
project(tempo VERSION 0.1.0 LANGUAGES CXX CUDA)

# ============================================================================
# Standard settings
# ============================================================================

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# ============================================================================
# Dependencies
# ============================================================================

# --- CUDA ---
find_package(CUDAToolkit REQUIRED)

# --- CUPTI (ships with the CUDA toolkit) ---
# On Perlmutter: $CUDA_HOME/extras/CUPTI/lib64/libcupti.so
find_library(CUPTI_LIB
    NAMES cupti
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
        ENV CUPTI_LIB_DIR
    REQUIRED)
get_filename_component(CUPTI_LIB_DIR ${CUPTI_LIB} DIRECTORY)
find_path(CUPTI_INCLUDE_DIR
    NAMES cupti.h
    HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
        ${CUDAToolkit_INCLUDE_DIRS}
    REQUIRED)
message(STATUS "CUPTI: ${CUPTI_LIB}")

# --- liburing (io_uring userspace library) ---
# On Perlmutter: available as system package; or build from source.
find_library(URING_LIB NAMES uring REQUIRED)
find_path(URING_INCLUDE_DIR NAMES liburing.h REQUIRED)
message(STATUS "liburing: ${URING_LIB}")

# --- pybind11 (for Python bindings) ---
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================================
# libtempo_core — static library with C++ pacing algorithms
# ============================================================================

add_library(tempo_core STATIC
    src/attention_monitor/monitor.cu
    src/spike_absorber/absorber.cpp
    src/pacing_daemon/pacing_daemon.cpp
)

target_include_directories(tempo_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUPTI_INCLUDE_DIR}
    ${URING_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tempo_core PUBLIC
    CUDA::cudart
    ${CUPTI_LIB}
    ${URING_LIB}
    pthread
)

set_target_properties(tempo_core PROPERTIES
    CUDA_ARCHITECTURES "80;90"   # A100 = sm_80, H100 = sm_90
)

# ============================================================================
# libtempo.so — shared library exposed via LD_PRELOAD / ctypes
#
# Exports a clean C API:
#   tempo_create_engine(...)  → opaque handle
#   tempo_absorb(handle, ...)
#   tempo_destroy_engine(handle)
# ============================================================================

add_library(tempo SHARED
    src/c_api/tempo_c_api.cpp
)

target_include_directories(tempo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUPTI_INCLUDE_DIR}
    ${URING_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tempo PRIVATE
    tempo_core
    CUDA::cudart
    ${CUPTI_LIB}
    ${URING_LIB}
    pthread
)

set_target_properties(tempo PROPERTIES
    OUTPUT_NAME "tempo"
    SOVERSION   1
    VERSION     ${PROJECT_VERSION}
    CUDA_ARCHITECTURES "80;90"
)

# ============================================================================
# tempo_cpp — pybind11 extension module (imported as `import tempo_cpp`)
# ============================================================================

pybind11_add_module(tempo_cpp
    src/python_bindings/bindings.cpp
)

target_include_directories(tempo_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUPTI_INCLUDE_DIR}
    ${URING_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tempo_cpp PRIVATE
    tempo_core
    CUDA::cudart
    ${CUPTI_LIB}
    ${URING_LIB}
)

set_target_properties(tempo_cpp PROPERTIES
    CUDA_ARCHITECTURES "80;90"
)

# ============================================================================
# Tests
# ============================================================================

option(TEMPO_BUILD_TESTS "Build unit tests" ON)
if(TEMPO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ============================================================================
# Install targets
# ============================================================================

install(TARGETS tempo tempo_cpp tempo_core
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
)
install(DIRECTORY src/
    DESTINATION include/tempo
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "TEMPO ${PROJECT_VERSION} build configuration:")
message(STATUS "  Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA archs     : 80 (A100) + 90 (H100)")
message(STATUS "  CUPTI          : ${CUPTI_LIB}")
message(STATUS "  liburing       : ${URING_LIB}")
message(STATUS "  Build dir      : ${CMAKE_BINARY_DIR}")
message(STATUS "  Install prefix : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Quick build:")
message(STATUS "  mkdir build && cd build && cmake .. && make -j\$(nproc)")
message(STATUS "")
