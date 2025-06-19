# Module Dependency Analysis Report

Found 2 dependencies:

⚠️  **MACHINE LEARNING FRAMEWORKS DETECTED**

This project uses ML frameworks that require special handling. See the ML Migration Guide section below.

- **torch**: HIGH complexity conversion

---

## torch
- **Installed**: No
- **Builtin**: No
- **Pure Python**: Yes

### Recommended C++ Alternative: LibTorch
PyTorch C++ API for deep learning

```cmake
find_package(Torch REQUIRED)
```
**Documentation**: https://pytorch.org/cppdocs/

## numpy
- **Installed**: No
- **Builtin**: No
- **Pure Python**: Yes

### Recommended C++ Alternative: Eigen
C++ template library for linear algebra

```cmake
find_package(Eigen3 REQUIRED)
```
**vcpkg**: `vcpkg install eigen3`
**Documentation**: https://eigen.tuxfamily.org/

## Build Configuration

Add the following to your CMakeLists.txt:

```cmake
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
```

## Package Manager Commands

### vcpkg
```bash
vcpkg install eigen3
```

### Conan
```bash
conan install eigen/3.4.0
```

---

# Machine Learning Framework Migration Guide

This project uses machine learning frameworks that require special attention during C++ conversion.

## Summary of ML Dependencies

### torch → LibTorch
**Complexity**: HIGH
**Description**: PyTorch C++ API for deep learning

**Important Notes:**
- LibTorch provides the complete PyTorch C++ API
- Supports CUDA for GPU acceleration
- Requires manual installation from pytorch.org
- Compatible with PyTorch models and checkpoints

**Alternative Approaches:**
- **ONNX Runtime**: Cross-platform ML inference
- **TensorFlow C++**: TensorFlow C++ API

## Migration Strategies

### Strategy 1: Direct Framework Translation
- Use the C++ API of the same framework (LibTorch, TensorFlow C++)
- Maintains full compatibility with Python models
- Requires framework-specific setup and dependencies

### Strategy 2: Model Export + Inference Engine
- Export Python models to ONNX format
- Use ONNX Runtime for C++ inference
- Lighter weight, good for inference-only scenarios

### Strategy 3: Algorithm Reimplementation
- Reimplement algorithms using general-purpose C++ libraries
- More work but better control and potential optimization
- Good for simple ML algorithms

## Recommended Approach

For deep learning frameworks (PyTorch/TensorFlow):
1. **For inference**: Export models to ONNX and use ONNX Runtime
2. **For training**: Use LibTorch C++ API or TensorFlow C++
3. **For simple networks**: Consider reimplementation with Eigen

## Implementation Steps

1. **Analyze your models**: Determine if you need training or just inference
2. **Choose strategy**: Based on complexity and requirements
3. **Set up environment**: Install chosen C++ ML framework
4. **Port incrementally**: Start with data loading, then model inference
5. **Optimize**: Profile and optimize performance-critical sections

## Useful Resources

- [LibTorch Tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
- [TensorFlow C++ Guide](https://www.tensorflow.org/guide/extend/cc)
- [ONNX Runtime C++](https://onnxruntime.ai/docs/api/c/)
- [mlpack Documentation](https://mlpack.org/doc.html)


# Machine Learning Framework Conversion Guide

This project uses machine learning frameworks that require special attention for C++ conversion.

## TORCH Conversion

**Strategy**: LibTorch C++ API

### Conversion Steps:
- 1. Convert PyTorch model to TorchScript (.pt file)
- 2. Load model in C++ using torch::jit::load()
- 3. Replace Python training code with C++ LibTorch equivalents
- 4. Use torch::Tensor for data handling

### Code Example:
```python
# Python: model = torch.nn.Linear(10, 1)
```
```cpp
// C++: torch::nn::Linear linear(10, 1);
```

### Build Instructions:
- 1. Download LibTorch from pytorch.org
- 2. Extract to /path/to/libtorch
- 3. Add to CMakeLists.txt: set(CMAKE_PREFIX_PATH /path/to/libtorch)
- 4. Use: find_package(Torch REQUIRED)
- 5. Link: target_link_libraries(your_target ${TORCH_LIBRARIES})

## General Recommendations

1. **Model Serialization**: Use ONNX format for cross-framework compatibility
2. **Inference vs Training**: Consider if you need training in C++ or just inference
3. **Performance**: C++ implementations can be 2-10x faster than Python
4. **Memory Management**: Use smart pointers for model management
5. **Threading**: Take advantage of C++ threading for parallel processing

## Alternative Approaches

1. **Python Extension**: Keep ML code in Python, create C++ extensions for performance-critical parts
2. **REST API**: Keep Python ML service, call from C++ via HTTP
3. **Embedded Python**: Embed Python interpreter in C++ application
4. **Model Serving**: Use TensorFlow Serving, ONNX Runtime, or TorchServe