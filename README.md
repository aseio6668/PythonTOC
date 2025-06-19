# Python to C++ Translator

A comprehensive tool for converting Python source code to C++ with intelligent type inference, memory management, and library mapping.

## Features

- **AST-based parsing**: Uses Python's Abstract Syntax Tree for accurate code analysis
- **Type inference**: Intelligent type detection for Python variables and functions
- **Memory management**: Automatic conversion to appropriate C++ memory management patterns
- **Library mapping**: Maps Python standard library functions to C++ equivalents
- **Object-oriented support**: Converts Python classes to C++ classes
- **Error handling**: Translates Python exceptions to C++ exception handling
- **🆕 Dynamic Module Analysis**: Real-time analysis and translation of unknown Python modules
- **🆕 Dependency Management**: Intelligent mapping of Python packages to C++ equivalents
- **🆕 ML Framework Support**: Specialized guidance for machine learning framework migration
- **🆕 Build System Integration**: Automatic CMake, vcpkg, and Conan configuration generation

## Project Structure

```
PythonToC/
├── src/
│   ├── __init__.py
│   ├── main.py              # Main CLI interface
│   ├── parser/              # Python code parsing
│   │   ├── __init__.py
│   │   ├── ast_parser.py    # AST parsing logic
│   │   └── type_inferrer.py # Type inference engine
│   ├── translator/          # Core translation logic
│   │   ├── __init__.py
│   │   ├── translator.py    # Main translator class
│   │   ├── cpp_generator.py # C++ code generation
│   │   └── library_mapper.py # Python to C++ library mapping
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── helpers.py       # Helper functions
│       └── templates.py     # C++ code templates
├── tests/                   # Test files
├── examples/                # Example Python files to translate
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the translator: `python src/main.py <input_file.py> [output_file.cpp]`

## Usage

### Command Line Interface
```bash
python src/main.py input.py output.cpp
```

### Python API
```python
from src.translator.translator import PythonToCppTranslator

translator = PythonToCppTranslator()
cpp_code = translator.translate_file("input.py")
```

## Examples

### Input Python Code:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        self.result = a + b
        return self.result
```

### Generated C++ Code:
```cpp
#include <iostream>
#include <memory>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
private:
    int result;
    
public:
    Calculator() : result(0) {}
    
    int add(int a, int b) {
        result = a + b;
        return result;
    }
};
```

## Supported Python Features

- [x] Functions and methods
- [x] Classes and inheritance
- [x] Control flow (if/else, loops)
- [x] Basic data types (int, float, string, bool)
- [x] Lists and dictionaries (mapped to vectors and maps)
- [x] Exception handling
- [ ] Decorators
- [ ] Generators
- [ ] Lambda functions
- [ ] Complex data structures

## New Features 🆕

### **Advanced Module Dependency Management**
- **Automatic dependency analysis** from Python imports
- **C++ library mapping** for popular Python packages
- **Complete project generation** with build configurations
- **Package manager integration** (vcpkg, Conan, CMake)

### **Supported Python Modules**
- **Scientific**: numpy → Eigen3, scipy → Eigen3+GSL
- **Image Processing**: PIL → OpenCV, opencv-python → OpenCV
- **CLI Tools**: click → CLI11, argparse → CLI11
- **Networking**: requests → libcurl
- **Data**: json → nlohmann/json

### **Enhanced CLI**
```bash
# Analyze dependencies and create complete C++ project
python translate.py your_app.py --create-project --dependencies

# Generate detailed dependency report
python translate.py your_app.py --deps-report --dependencies

# Batch process with dependency management
python translate.py python_project/ --batch --dependencies --cmake
```

### Advanced Features

#### Dynamic Module Analysis
Analyze unknown Python modules and generate C++ translations automatically:

```bash
# Analyze dependencies with dynamic analysis
python translate.py my_script.py --deps-report

# Auto-translate unknown modules  
python translate.py my_script.py --create-project --download-deps

# Generate comprehensive migration guide
python translate.py ml_script.py --deps-report  # Includes ML framework guidance
```

#### Dependency Management
Intelligent mapping and build system generation:

```bash
# Generate CMake configuration
python translate.py my_script.py --cmake

# Create complete C++ project
python translate.py my_script.py --create-project

# Batch translate directory
python translate.py src/ --batch --dependencies
```

For detailed information about the dynamic analysis system, see [`DYNAMIC_ANALYSIS.md`](DYNAMIC_ANALYSIS.md).

## Practical Example: Image Processing App

Let's convert a Python image processing application to C++:

### **Original Python Code** (`glintcopy.py`)
```python
import click
import numpy as np
from PIL import Image
from scipy.ndimage import sobel

@click.command()
@click.argument("image_path")
@click.option("--width", default=100)
def image_to_ascii(image_path, width):
    img = Image.open(image_path).convert("L")
    # ... image processing logic
```

### **Translation with Dependency Management**
```bash
python translate.py glintcopy.py --create-project --dependencies --verbose
```

### **Generated C++ Project Structure**
```
cpp_modules/glintcopy/
├── CMakeLists.txt     # OpenCV + Eigen3 + CLI11 + GSL
├── vcpkg.json         # Package dependencies
├── conanfile.txt      # Alternative package manager
├── main.cpp           # Translated C++ code
└── README.md          # Build instructions
```

### **Build and Run**
```bash
cd cpp_modules/glintcopy
vcpkg install  # Installs: opencv eigen3 cli11 gsl
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build .
./glintcopy image.jpg --width 80
```

### **Dependency Mapping Results**
- `click` → **CLI11** (Modern C++ command line parser)
- `numpy` → **Eigen3** (Linear algebra library)
- `PIL` → **OpenCV** (Computer vision library)
- `scipy` → **GSL** (GNU Scientific Library)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License
