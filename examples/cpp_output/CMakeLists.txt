cmake_minimum_required(VERSION 3.12)
project(cpp_output)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES calculator_class.cpp data_structures.cpp simple_functions.cpp)

# Add executable
add_executable(cpp_output ${SOURCES})

# Compiler options
target_compile_options(cpp_output PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
