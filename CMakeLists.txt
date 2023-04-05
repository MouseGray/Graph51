cmake_minimum_required(VERSION 3.5)

project(Graph LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Graph main.cpp
    algorithm.h
    bitmap.h
    tree.h
    vpoint.h
    utils.h
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

