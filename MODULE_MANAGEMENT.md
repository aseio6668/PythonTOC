# Python Module to C++ Dependency Management

This document explains how the Python to C++ translator handles module dependencies and provides automated C++ project generation.

## 🎯 **How It Works**

### 1. **Dependency Analysis**
The translator analyzes Python files to extract:
- Import statements (`import numpy`)
- From-imports (`from PIL import Image`)
- Module usage patterns

### 2. **C++ Equivalent Mapping**
For each Python module, the system provides:
- **Direct C++ libraries** (numpy → Eigen)
- **Alternative solutions** (PIL → OpenCV or SOIL2)
- **Build system integration** (CMake, vcpkg, Conan)

### 3. **Project Generation**
Creates complete C++ projects with:
- **CMakeLists.txt** with proper find_package commands
- **vcpkg.json** for dependency management
- **conanfile.txt** for Conan package management
- **README.md** with build instructions

## 📋 **Supported Python Modules**

### **Scientific Computing**
| Python Module | C++ Alternative | Package Manager |
|---------------|-----------------|-----------------|
| `numpy` | Eigen3 | `vcpkg install eigen3` |
| `scipy` | Eigen3 + GSL | `vcpkg install eigen3 gsl` |
| `pandas` | Custom/Arrow | Manual implementation |

### **Image Processing**
| Python Module | C++ Alternative | Package Manager |
|---------------|-----------------|-----------------|
| `PIL/Pillow` | OpenCV | `vcpkg install opencv` |
| `opencv-python` | OpenCV | `vcpkg install opencv` |

### **CLI and Utilities**
| Python Module | C++ Alternative | Package Manager |
|---------------|-----------------|-----------------|
| `click` | CLI11 | `vcpkg install cli11` |
| `argparse` | CLI11/argparse | `vcpkg install cli11` |

### **Web and Networking**
| Python Module | C++ Alternative | Package Manager |
|---------------|-----------------|-----------------|
| `requests` | libcurl | `vcpkg install curl` |
| `urllib` | libcurl | `vcpkg install curl` |

### **Data Formats**
| Python Module | C++ Alternative | Package Manager |
|---------------|-----------------|-----------------|
| `json` | nlohmann/json | `vcpkg install nlohmann-json` |

## 🚀 **Usage Examples**

### **Basic Dependency Analysis**
```bash
# Analyze dependencies and generate report
python translate.py your_file.py --deps-report --dependencies

# This creates: your_file.dependencies.md
```

### **Create Complete C++ Project**
```bash
# Generate complete C++ project with dependencies
python translate.py glintcopy.py --create-project --dependencies

# Creates: cpp_modules/glintcopy/ with all build files
```

### **Advanced Usage**
```bash
# Combine multiple features
python translate.py complex_app.py \
    --create-project \
    --dependencies \
    --namespace myapp \
    --verbose

# Batch process with dependency management
python translate.py python_project/ \
    --batch \
    --dependencies \
    --cmake \
    --verbose
```

## 📁 **Generated Project Structure**

When using `--create-project --dependencies`, you get:

```
cpp_modules/your_project/
├── CMakeLists.txt      # CMake build configuration
├── vcpkg.json          # vcpkg dependency specification
├── conanfile.txt       # Conan dependency specification
├── main.cpp            # Translated C++ code
└── README.md           # Build instructions
```

## 🔧 **Building Generated Projects**

### **With vcpkg (Recommended)**
```bash
cd cpp_modules/your_project
vcpkg install  # Installs dependencies from vcpkg.json
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
cmake --build .
```

### **With Conan**
```bash
cd cpp_modules/your_project
mkdir build
cd build
conan install ..
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

### **Manual CMake** (if dependencies are already installed)
```bash
cd cpp_modules/your_project
mkdir build
cd build
cmake ..
cmake --build .
```

## 🎛️ **Configuration Options**

### **CLI Flags**
- `--dependencies`: Enable dependency analysis
- `--deps-report`: Generate detailed dependency report
- `--create-project`: Create complete C++ project
- `--download-deps`: Download pure Python dependencies (future feature)

### **Configuration File** (`config.json`)
```json
{
  "dependency_management": {
    "preferred_package_manager": "vcpkg",
    "alternative_libraries": {
      "PIL": "SOIL2",
      "matplotlib": "custom_plotting"
    },
    "output_directory": "cpp_projects"
  }
}
```

## 🔍 **Real Example: glintcopy.py**

**Original Python dependencies:**
- `click` (CLI parsing)
- `numpy` (array operations)
- `PIL` (image processing)
- `scipy` (image filters)

**Generated C++ project:**
- **CLI11** for command-line parsing
- **Eigen3** for numerical operations
- **OpenCV** for image processing
- **GSL** for additional scientific functions

**Build commands:**
```bash
vcpkg install cli11 eigen3 opencv gsl
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build .
```

## 🛠️ **Manual Translation Required**

Some Python features still need manual attention:

### **Immediate TODOs in Generated Code**
- `/* TODO: GeneratorExp */` - Generator expressions
- `/* TODO: Subscript */` - Array subscripting
- `/* TODO: AugAssign */` - Augmented assignment (`+=`, `*=`)
- `/* TODO: JoinedStr */` - F-string formatting

### **Library-Specific Mappings**
```cpp
// Python: img = Image.open("file.jpg")
// C++: cv::Mat img = cv::imread("file.jpg");

// Python: arr = np.array([1, 2, 3])
// C++: Eigen::VectorXd arr(3); arr << 1, 2, 3;

// Python: @click.command()
// C++: CLI::App app{"Description"};
```

## 📈 **Future Enhancements**

### **Planned Features**
- [ ] Automatic downloading of pure Python modules
- [ ] Direct conversion of simple Python libraries
- [ ] Better template/generic type handling
- [ ] CMake target linking automation
- [ ] Docker container generation
- [ ] Cross-platform build scripts

### **Advanced Dependency Management**
- [ ] Version constraint handling
- [ ] Dependency conflict resolution
- [ ] Private package repository support
- [ ] Custom library mappings

## 🤝 **Contributing New Module Mappings**

To add support for a new Python module:

1. **Add to `cpp_equivalents` dictionary** in `dependency_manager.py`
2. **Specify C++ alternative**, includes, and package manager info
3. **Test with real Python code** using that module
4. **Submit pull request** with documentation

Example:
```python
'your_module': {
    'name': 'YourCppLib',
    'description': 'C++ equivalent description',
    'cmake_find': 'find_package(YourCppLib REQUIRED)',
    'includes': ['#include <yourcpplib/header.h>'],
    'vcpkg': 'yourcpplib',
    'conan': 'yourcpplib/1.0.0',
    'url': 'https://yourcpplib.org/'
}
```

---

This system bridges the gap between Python's rich ecosystem and C++'s performance, making it easier to port Python applications to C++ with proper dependency management.
