cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(unsharp_mask LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 62 75 86)
  # See https://developer.nvidia.com/cuda-gpus for GPU compute capabilities (CC)
  # Compilation speed will increase if you only list the CC of your own card
endif()

set_source_files_properties( unsharp_mask.cpp PROPERTIES LANGUAGE CUDA )

add_executable(unsharp_mask unsharp_mask.cpp unsharp_mask.hpp blur.hpp add_weighted.hpp ppm.hpp)

set_property(TARGET unsharp_mask PROPERTY CXX_STANDARD 17)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT unsharp_mask)