# glintcopy

Generated C++ project from Python source: glintcopy.py

## Building

### With CMake:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### With vcpkg (if vcpkg.json exists):
```bash
vcpkg install
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
cmake --build .
```

### With Conan (if conanfile.txt exists):
```bash
mkdir build
cd build
conan install ..
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

## Dependencies

See the dependency report and CMakeLists.txt for required libraries.
