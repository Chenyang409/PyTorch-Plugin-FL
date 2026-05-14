cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda or maca")

if(ACCELERATOR STREQUAL "maca")
  project(TORCH_FLAGOS CXX C)
else()
  project(TORCH_FLAGOS CUDA CXX C)
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_MESSAGE NEVER)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)

if(ACCELERATOR STREQUAL "maca")
  # Muxi MACA SDK: use cu-bridge headers and runtime_cu library
  if(NOT MACA_PATH)
    set(MACA_PATH "$ENV{MACA_PATH}")
  endif()
  if(NOT MACA_PATH)
    set(MACA_PATH "$ENV{MACA_HOME}")
  endif()
  if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca")
  endif()
  message(STATUS "Muxi MACA SDK path: ${MACA_PATH}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MACA_PATH}/tools/cu-bridge/include;${MACA_PATH}/include;${MACA_PATH}/include/mcr"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${MACA_PATH}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE runtime_cu)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely.
  # cuda.cmake guard: if(TARGET torch::cudart) return() endif()
  add_library(torch::cudart INTERFACE IMPORTED)

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
else()
  # Workaround for CUDA 12.x where nvToolsExt is header-only (no separate library)
  # PyTorch's CMake expects CUDA::nvToolsExt target which doesn't exist in CUDA 12.x
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET CUDA::nvToolsExt)
    # Create a dummy IMPORTED target for nvToolsExt (NVTX3 is header-only in CUDA 12.x)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    if(TARGET CUDA::nvtx3)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        INTERFACE_LINK_LIBRARIES CUDA::nvtx3)
    endif()
  endif()

  set(CUDA_RUNTIME_LIB CUDA::cudart)
endif()

# PyTorch's Caffe2 cuda.cmake still follows legacy FindCUDA expectations
# (CUDA_INCLUDE_DIRS / CUDA_TOOLKIT_ROOT_DIR). Conda split layouts only
# populate CUDAToolkit_*; without this, find_package(Torch) fails with
# "Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS)".
if(ACCELERATOR STREQUAL "cuda" AND CUDAToolkit_FOUND)
  set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}" CACHE STRING
      "CUDA include dirs (compat for PyTorch cuda.cmake)" FORCE)
  if(CUDAToolkit_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT_DIR}" CACHE PATH
        "CUDA toolkit root (compat for PyTorch cuda.cmake)" FORCE)
  else()
    list(GET CUDAToolkit_INCLUDE_DIRS 0 _torch_cuda_inc0)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${_torch_cuda_inc0}" DIRECTORY)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH
        "CUDA toolkit root (compat for PyTorch cuda.cmake)" FORCE)
  endif()
endif()

find_package(Torch REQUIRED)

# --- FlagGems (C++ operators): directory that contains FlagGemsConfig.cmake
#     (pip: .../site-packages/flag_gems/lib/cmake/FlagGems;
#      in-tree: .../FlagGems/build/<preset>/FlagGemsConfig.cmake).
if(FlagGems_DIR)
  set(FLAGGEMS_DIR "${FlagGems_DIR}")
elseif(FLAGGEMS_DIR)
  # already set (cache / -DFLAGGEMS_DIR)
elseif(DEFINED ENV{FLAGGEMS_DIR} AND NOT "$ENV{FLAGGEMS_DIR}" STREQUAL "")
  set(FLAGGEMS_DIR "$ENV{FLAGGEMS_DIR}")
endif()

if(NOT FLAGGEMS_DIR)
  file(
    GLOB _fg_cfg_candidates
    LIST_DIRECTORIES false
    "${CMAKE_SOURCE_DIR}/FlagGems/build/*/FlagGemsConfig.cmake"
    "${CMAKE_SOURCE_DIR}/FlagGems/build/FlagGemsConfig.cmake"
  )
  if(_fg_cfg_candidates)
    list(SORT _fg_cfg_candidates COMPARE NATURAL ORDER DESCENDING)
    list(GET _fg_cfg_candidates 0 _fg_cfg_file)
    get_filename_component(FLAGGEMS_DIR "${_fg_cfg_file}" DIRECTORY)
    message(STATUS "FlagGems_DIR (in-tree build): ${FLAGGEMS_DIR}")
  endif()
endif()

if(NOT FLAGGEMS_DIR OR NOT EXISTS "${FLAGGEMS_DIR}/FlagGemsConfig.cmake")
  message(
    FATAL_ERROR
      "FlagGems C++ package not found (need FlagGemsConfig.cmake).\n"
      "  Option A — use the bundled submodule (from repo root):\n"
      "    cmake -S FlagGems -B FlagGems/build/cpython-313 \\\n"
      "      -DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DPython_EXECUTABLE=/path/to/your/python\n"
      "    cmake --build FlagGems/build/cpython-313 -j\n"
      "  Option B — install the flag_gems wheel and re-run pip, or set:\n"
      "    export FLAGGEMS_DIR=/path/to/flag_gems/lib/cmake/FlagGems\n"
      "  Option C — pass -DFlagGems_DIR=... to cmake."
  )
endif()

set(FlagGems_DIR "${FLAGGEMS_DIR}" CACHE PATH "Directory containing FlagGemsConfig.cmake" FORCE)

# FlagGemsConfig.cmake calls find_dependency(Torch MODULE); FindTorch.cmake must be on MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_DIR}")
find_package(FlagGems REQUIRED PATHS "${FLAGGEMS_DIR}" NO_DEFAULT_PATH)

# RPATH for liboperators: build tree uses ${FLAGGEMS_DIR}/lib; pip layout uses ${FLAGGEMS_DIR}/../..
set(_flaggems_libdir "")
if(EXISTS "${FLAGGEMS_DIR}/lib/liboperators.so")
  get_filename_component(_flaggems_libdir "${FLAGGEMS_DIR}/lib" REALPATH)
else()
  file(GLOB _fg_ops "${FLAGGEMS_DIR}/lib/liboperators.so*")
  if(_fg_ops)
    list(GET _fg_ops 0 _fg_op0)
    get_filename_component(_flaggems_libdir "${_fg_op0}" DIRECTORY)
  elseif(EXISTS "${FLAGGEMS_DIR}/../lib/liboperators.so")
    get_filename_component(_flaggems_libdir "${FLAGGEMS_DIR}/../lib" REALPATH)
  else()
    file(GLOB _fg_ops2 "${FLAGGEMS_DIR}/../lib/liboperators.so*")
    if(_fg_ops2)
      list(GET _fg_ops2 0 _fg_op1)
      get_filename_component(_flaggems_libdir "${_fg_op1}" DIRECTORY)
    endif()
  endif()
endif()
if(NOT _flaggems_libdir)
  get_filename_component(_flaggems_libdir "${FLAGGEMS_DIR}/../.." REALPATH)
endif()
if(EXISTS "${_flaggems_libdir}")
  list(APPEND CMAKE_INSTALL_RPATH "${_flaggems_libdir}")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/accelerator)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_fl/csrc)
