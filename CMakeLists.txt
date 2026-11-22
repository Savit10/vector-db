cmake_minimum_required(VERSION 3.16)
project(vectordb CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

add_library(vdb src/flat_index.cpp)
target_include_directories(vdb PUBLIC include)

enable_testing()
add_executable(test_flat_index tests/test_flat_index.cpp)
target_link_libraries(test_flat_index PRIVATE vdb)
add_test(NAME flat_index COMMAND test_flat_index)

add_executable(bench_flat bench/bench_flat.cpp)
target_link_libraries(bench_flat PRIVATE vdb)
