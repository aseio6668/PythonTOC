# Module Dependency Analysis Report

Found 4 dependencies:

## scipy
- **Installed**: No
- **Builtin**: No
- **Pure Python**: Yes

### Recommended C++ Alternative: Eigen + GSL
Combination of Eigen and GNU Scientific Library

```cmake
find_package(Eigen3 REQUIRED)
```
```cmake
find_package(GSL REQUIRED)
```
**vcpkg**: `vcpkg install eigen3 gsl`

## PIL
- **Installed**: No
- **Builtin**: No
- **Pure Python**: Yes

### Recommended C++ Alternative: OpenCV
Computer vision library with image processing

```cmake
find_package(OpenCV REQUIRED)
```
**vcpkg**: `vcpkg install opencv`
**Documentation**: https://opencv.org/

## numpy
- **Installed**: Yes
- **Builtin**: No
- **Pure Python**: Yes

### Recommended C++ Alternative: Eigen
C++ template library for linear algebra

```cmake
find_package(Eigen3 REQUIRED)
```
**vcpkg**: `vcpkg install eigen3`
**Documentation**: https://eigen.tuxfamily.org/

## click
- **Installed**: Yes
- **Builtin**: No
- **Pure Python**: Yes

### Recommended C++ Alternative: CLI11
Command line parser for C++11

**vcpkg**: `vcpkg install cli11`
**Documentation**: https://github.com/CLIUtils/CLI11

## Build Configuration

Add the following to your CMakeLists.txt:

```cmake
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenCV REQUIRED)
```

## Package Manager Commands

### vcpkg
```bash
vcpkg install cli11 eigen3 gsl opencv
```

### Conan
```bash
conan install eigen/3.4.0
conan install gsl/2.7
conan install opencv/4.5.5
conan install eigen/3.4.0
conan install cli11/2.2.0
```