cmake_minimum_required(VERSION 3.12)
project(calculator_class)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages

# Add executable
add_executable(calculator_class main.cpp)

# Link libraries
# TODO: Add specific library linking based on dependencies

# Compiler options
target_compile_options(calculator_class PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)