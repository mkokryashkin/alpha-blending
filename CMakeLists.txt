cmake_minimum_required(VERSION 3.15)
project(alpha-blending)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-march=native -O3")

add_executable(blender main.cpp)