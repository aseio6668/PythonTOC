cmake_minimum_required(VERSION 3.12)
project(glintcopy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)

# Add executable
add_executable(glintcopy main.cpp)

# Link libraries
# TODO: Add specific library linking based on dependencies

# Compiler options
target_compile_options(glintcopy PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)