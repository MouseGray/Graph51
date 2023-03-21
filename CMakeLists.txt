cmake_minimum_required(VERSION 3.5)

project(Graph3 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(Graph3 main.cpp)

target_link_libraries(Graph3 OpenCL::OpenCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

