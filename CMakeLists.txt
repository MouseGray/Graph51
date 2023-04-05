cmake_minimum_required(VERSION 3.5)

project(Graph LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Graph src/main.cpp
    src/algorithm.h
    src/bitmap.h
    src/tree.h
    src/utils.h
    src/vpoint.h
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

