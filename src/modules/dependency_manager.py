"""
Module dependency manager for Python to C++ translation
Handles analysis, downloading, and conversion of Python modules
"""

import os
import sys
import ast
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.request import urlopen
from urllib.parse import urljoin
import zipfile
# Dynamic analysis imports
import importlib
import inspect
import re
from enum import Enum


@dataclass
class ModuleInfo:
    """Information about a Python module"""
    name: str
    version: Optional[str] = None
    is_builtin: bool = False
    is_installed: bool = False
    is_pure_python: bool = True
    source_url: Optional[str] = None
    cpp_equivalent: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ConversionResult:
    """Result of module conversion"""
    success: bool
    module_name: str
    cpp_files: List[Path] = None
    header_files: List[Path] = None
    cmake_config: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.cpp_files is None:
            self.cpp_files = []
        if self.header_files is None:
            self.header_files = []


class ModuleDependencyManager:
    """Manages Python module dependencies for C++ conversion"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("cpp_modules")
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache for module information
        self.module_cache: Dict[str, ModuleInfo] = {}
        
        # Dynamic analyzer for unknown modules
        self.dynamic_analyzer = None  # Will be initialized with translator reference
        
        # Known C++ equivalents for popular Python libraries
        self.cpp_equivalents = {
            'numpy': {
                'name': 'Eigen',
                'description': 'C++ template library for linear algebra',
                'cmake_find': 'find_package(Eigen3 REQUIRED)',
                'includes': ['#include <Eigen/Dense>'],
                'vcpkg': 'eigen3',
                'conan': 'eigen/3.4.0',
                'url': 'https://eigen.tuxfamily.org/'
            },
            'PIL': {
                'name': 'OpenCV',
                'description': 'Computer vision library with image processing',
                'cmake_find': 'find_package(OpenCV REQUIRED)',
                'includes': ['#include <opencv2/opencv.hpp>'],
                'vcpkg': 'opencv',
                'conan': 'opencv/4.5.5',
                'url': 'https://opencv.org/',
                'alternative': {
                    'name': 'SOIL2',
                    'description': 'Simple OpenGL Image Library',
                    'includes': ['#include <SOIL2/SOIL2.h>']
                }
            },
            'matplotlib': {
                'name': 'matplotlib-cpp',
                'description': 'C++ plotting library inspired by matplotlib',
                'includes': ['#include <matplotlibcpp.h>'],
                'vcpkg': 'matplotlib-cpp',
                'url': 'https://github.com/lava/matplotlib-cpp'
            },
            'requests': {
                'name': 'libcurl',
                'description': 'HTTP client library',
                'cmake_find': 'find_package(CURL REQUIRED)',
                'includes': ['#include <curl/curl.h>'],
                'vcpkg': 'curl',
                'conan': 'libcurl/7.80.0'
            },
            'click': {
                'name': 'CLI11',
                'description': 'Command line parser for C++11',
                'includes': ['#include <CLI/CLI.hpp>'],
                'vcpkg': 'cli11',
                'conan': 'cli11/2.2.0',
                'url': 'https://github.com/CLIUtils/CLI11',
                'alternative': {
                    'name': 'argparse',
                    'description': 'Argument parser for C++17',
                    'includes': ['#include <argparse/argparse.hpp>']
                }
            },
            'scipy': {
                'name': 'Eigen + GSL',
                'description': 'Combination of Eigen and GNU Scientific Library',
                'cmake_find': ['find_package(Eigen3 REQUIRED)', 'find_package(GSL REQUIRED)'],
                'includes': ['#include <Eigen/Dense>', '#include <gsl/gsl.h>'],
                'vcpkg': ['eigen3', 'gsl'],
                'conan': ['eigen/3.4.0', 'gsl/2.7']
            },
            # Data Science and Analytics
            'pandas': {
                'name': 'Apache Arrow + Custom DataFrame',
                'description': 'Use Apache Arrow for columnar data processing',
                'cmake_find': 'find_package(Arrow REQUIRED)',
                'includes': ['#include <arrow/api.h>', '// Custom DataFrame implementation needed'],
                'vcpkg': 'arrow',
                'conan': 'arrow/8.0.0',
                'url': 'https://arrow.apache.org/docs/cpp/',
                'alternative': {
                    'name': 'Custom DataFrame',
                    'description': 'Implement custom DataFrame class using std containers'
                }
            },
            'plotly': {
                'name': 'matplotlib-cpp',
                'description': 'C++ plotting library',
                'includes': ['#include <matplotlibcpp.h>'],
                'vcpkg': 'matplotlib-cpp'
            },
            'seaborn': {
                'name': 'matplotlib-cpp',
                'description': 'Use matplotlib-cpp for statistical plotting',
                'includes': ['#include <matplotlibcpp.h>'],
                'vcpkg': 'matplotlib-cpp'
            },
            'json': {
                'name': 'nlohmann/json',
                'description': 'JSON for Modern C++',
                'includes': ['#include <nlohmann/json.hpp>'],
                'vcpkg': 'nlohmann-json',
                'conan': 'nlohmann_json/3.11.2'
            },
            
            # Machine Learning and Deep Learning Frameworks
            'torch': {
                'name': 'LibTorch',
                'description': 'PyTorch C++ API for deep learning',
                'cmake_find': 'find_package(Torch REQUIRED)',
                'includes': ['#include <torch/torch.h>'],
                'url': 'https://pytorch.org/cppdocs/',
                'installation': 'Download LibTorch from pytorch.org',
                'complexity': 'high',
                'notes': [
                    'LibTorch provides the complete PyTorch C++ API',
                    'Supports CUDA for GPU acceleration',
                    'Requires manual installation from pytorch.org',
                    'Compatible with PyTorch models and checkpoints'
                ],
                'alternatives': [
                    {
                        'name': 'ONNX Runtime',
                        'description': 'Cross-platform ML inference',
                        'includes': ['#include <onnxruntime_cxx_api.h>'],
                        'vcpkg': 'onnxruntime'
                    },
                    {
                        'name': 'TensorFlow C++',
                        'description': 'TensorFlow C++ API',
                        'url': 'https://www.tensorflow.org/install/lang_c'
                    }
                ]
            },
            'tensorflow': {
                'name': 'TensorFlow C++',
                'description': 'TensorFlow C++ API for machine learning',
                'cmake_find': 'find_package(TensorFlow REQUIRED)',
                'includes': ['#include <tensorflow/cc/client/client_session.h>',
                           '#include <tensorflow/cc/ops/standard_ops.h>'],
                'url': 'https://www.tensorflow.org/install/lang_c',
                'complexity': 'high',
                'notes': [
                    'TensorFlow C++ API for production ML models',
                    'Supports GPU acceleration with CUDA',
                    'Can load SavedModel and frozen graphs',
                    'Complex build process - consider using Docker'
                ]
            },
            'sklearn': {
                'name': 'Multiple Options',
                'description': 'No direct equivalent - use specialized libraries',
                'complexity': 'medium',
                'alternatives': [
                    {
                        'name': 'Eigen + Custom',
                        'description': 'Implement algorithms using Eigen',
                        'includes': ['#include <Eigen/Dense>'],
                        'vcpkg': 'eigen3'
                    },
                    {
                        'name': 'mlpack',
                        'description': 'C++ machine learning library',
                        'includes': ['#include <mlpack/mlpack.hpp>'],
                        'vcpkg': 'mlpack',
                        'url': 'https://mlpack.org/'
                    },
                    {
                        'name': 'Shark-ML',
                        'description': 'Machine learning library',
                        'url': 'http://shark-ml.org/'
                    }
                ]
            },
            'keras': {
                'name': 'TensorFlow C++ / LibTorch',
                'description': 'Use underlying framework C++ API',
                'complexity': 'high',
                'notes': [
                    'Keras models can be converted to TensorFlow SavedModel',
                    'Or exported to ONNX format for inference',
                    'Consider using TensorFlow C++ or ONNX Runtime'
                ]
            },
            'transformers': {
                'name': 'ONNX Runtime + Hugging Face Models',
                'description': 'Use ONNX Runtime for transformer inference',
                'includes': ['#include <onnxruntime_cxx_api.h>'],
                'vcpkg': 'onnxruntime',
                'complexity': 'high',
                'notes': [
                    'Export Hugging Face models to ONNX format',
                    'Use ONNX Runtime for efficient inference',
                    'Supports BERT, GPT, and other transformer models'
                ]
            },
            'xgboost': {
                'name': 'XGBoost C++',
                'description': 'XGBoost C++ API for gradient boosting',
                'cmake_find': 'find_package(xgboost REQUIRED)',
                'includes': ['#include <xgboost/c_api.h>'],
                'vcpkg': 'xgboost',
                'url': 'https://xgboost.readthedocs.io/en/latest/',
                'complexity': 'medium'
            },
            'lightgbm': {
                'name': 'LightGBM C++',
                'description': 'LightGBM C++ API for gradient boosting',
                'includes': ['#include <LightGBM/c_api.h>'],
                'url': 'https://lightgbm.readthedocs.io/',
                'complexity': 'medium'
            }
        }
        
        # Builtin modules that don't need conversion
        self.builtin_modules = {
            'os', 'sys', 'math', 'random', 'time', 'datetime', 'collections',
            'itertools', 'functools', 'operator', 'copy', 'io', 'pathlib',
            'typing', 're', 'string', 'threading', 'multiprocessing'
        }
    
    def analyze_dependencies(self, file_path: Path) -> List[ModuleInfo]:
        """Analyze Python file to extract module dependencies"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            imported_modules = set()
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name.split('.')[0])  # Get top-level module
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module.split('.')[0])
            
            # Create ModuleInfo for each dependency
            for module_name in imported_modules:
                if module_name not in self.module_cache:
                    self.module_cache[module_name] = self._analyze_module(module_name)
                dependencies.append(self.module_cache[module_name])
        
        except Exception as e:
            print(f"Error analyzing dependencies: {e}")
        
        return dependencies
    
    def _analyze_module(self, module_name: str) -> ModuleInfo:
        """Analyze a single module to determine its characteristics"""
        module_info = ModuleInfo(name=module_name)
        
        # Check if it's a builtin module
        if module_name in self.builtin_modules:
            module_info.is_builtin = True
            module_info.cpp_equivalent = f"C++ standard library equivalent"
            return module_info
        
        # Check if it's installed
        try:
            __import__(module_name)
            module_info.is_installed = True
        except ImportError:
            module_info.is_installed = False
        
        # Check if we have a known C++ equivalent
        if module_name in self.cpp_equivalents:
            equiv = self.cpp_equivalents[module_name]
            module_info.cpp_equivalent = equiv['name']
        
        # Try to determine if it's pure Python
        if module_info.is_installed:
            module_info.is_pure_python = self._is_pure_python_module(module_name)
        
        return module_info
    
    def _is_pure_python_module(self, module_name: str) -> bool:
        """Check if a module is pure Python (no C extensions)"""
        try:
            module = __import__(module_name)
            if hasattr(module, '__file__') and module.__file__:
                return module.__file__.endswith('.py')
            return False
        except:
            return False
    
    def suggest_cpp_alternatives(self, dependencies: List[ModuleInfo]) -> Dict[str, Dict]:
        """Suggest C++ alternatives for Python dependencies"""
        suggestions = {}
        
        for dep in dependencies:
            if dep.name in self.cpp_equivalents:
                suggestions[dep.name] = self.cpp_equivalents[dep.name]
            elif dep.is_builtin:
                suggestions[dep.name] = {
                    'name': 'C++ Standard Library',
                    'description': f'Use C++ standard library equivalent for {dep.name}',
                    'includes': ['// See library mapper for specific mappings']
                }
            else:
                suggestions[dep.name] = {
                    'name': 'Manual Implementation Required',
                    'description': f'No direct C++ equivalent found for {dep.name}',
                    'includes': [f'// TODO: Implement {dep.name} functionality']
                }
        
        return suggestions
    
    def download_module_source(self, module_name: str, version: str = None) -> Optional[Path]:
        """Download source code for a Python module from PyPI"""
        try:
            # Use pip to download source
            download_dir = self.output_dir / "downloads" / module_name
            download_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "pip", "download",
                "--no-deps", "--no-binary", ":all:",
                "--dest", str(download_dir)
            ]
            
            if version:
                cmd.append(f"{module_name}=={version}")
            else:
                cmd.append(module_name)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find downloaded file
                for file in download_dir.iterdir():
                    if file.suffix in ['.tar.gz', '.zip']:
                        return self._extract_source(file, download_dir)
            else:
                print(f"Failed to download {module_name}: {result.stderr}")
        
        except Exception as e:
            print(f"Error downloading {module_name}: {e}")
        
        return None
    
    def _extract_source(self, archive_path: Path, extract_dir: Path) -> Optional[Path]:
        """Extract source archive and return path to extracted directory"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                # Handle .tar.gz
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            # Find extracted directory
            for item in extract_dir.iterdir():
                if item.is_dir() and item.name != '__pycache__':
                    return item
        
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
        
        return None
    
    def convert_pure_python_module(self, module_path: Path, translator) -> ConversionResult:
        """Convert a pure Python module to C++"""
        result = ConversionResult(success=False, module_name=module_path.name)
        
        try:
            # Find Python files in the module
            python_files = list(module_path.glob("**/*.py"))
            
            if not python_files:
                result.error_message = "No Python files found in module"
                return result
            
            # Convert each Python file
            converted_files = []
            for py_file in python_files:
                try:
                    cpp_code = translator.translate_file(str(py_file))
                    
                    # Create output file path
                    relative_path = py_file.relative_to(module_path)
                    cpp_file = self.output_dir / module_path.name / relative_path.with_suffix('.cpp')
                    cpp_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write C++ code
                    with open(cpp_file, 'w', encoding='utf-8') as f:
                        f.write(cpp_code)
                    
                    converted_files.append(cpp_file)
                    result.cpp_files.append(cpp_file)
                
                except Exception as e:
                    print(f"Error converting {py_file}: {e}")
            
            if converted_files:
                result.success = True
                result.cmake_config = self._generate_module_cmake(module_path.name, converted_files)
            else:
                result.error_message = "No files were successfully converted"
        
        except Exception as e:
            result.error_message = f"Conversion error: {e}"
        
        return result
    
    def _generate_module_cmake(self, module_name: str, cpp_files: List[Path]) -> str:
        """Generate CMake configuration for a converted module"""
        source_files = [f.name for f in cpp_files]
        
        cmake_content = f"""# CMake configuration for {module_name} module
# Generated by Python to C++ translator

add_library({module_name} STATIC
    {' '.join(source_files)}
)

target_include_directories({module_name} PUBLIC
    ${{CMAKE_CURRENT_SOURCE_DIR}}
)

target_compile_features({module_name} PRIVATE cxx_std_17)
"""
        
        return cmake_content
    
    def generate_dependency_report(self, dependencies: List[ModuleInfo], suggestions: Dict[str, Dict]) -> str:
        """Generate a detailed dependency report"""
        report_lines = [
            "# Module Dependency Analysis Report",
            "",
            f"Found {len(dependencies)} dependencies:",
            ""
        ]
        
        # Check for ML frameworks and add warning
        ml_complexity = self.analyze_ml_complexity(dependencies)
        if ml_complexity:
            report_lines.extend([
                "⚠️  **MACHINE LEARNING FRAMEWORKS DETECTED**",
                "",
                "This project uses ML frameworks that require special handling. See the ML Migration Guide section below.",
                ""
            ])
            
            for framework, complexity in ml_complexity.items():
                report_lines.append(f"- **{framework}**: {complexity.upper()} complexity conversion")
            
            report_lines.extend(["", "---", ""])
        
        for dep in dependencies:
            report_lines.extend([
                f"## {dep.name}",
                f"- **Installed**: {'Yes' if dep.is_installed else 'No'}",
                f"- **Builtin**: {'Yes' if dep.is_builtin else 'No'}",
                f"- **Pure Python**: {'Yes' if dep.is_pure_python else 'No'}",
                ""
            ])
            
            if dep.name in suggestions:
                suggestion = suggestions[dep.name]
                report_lines.extend([
                    f"### Recommended C++ Alternative: {suggestion['name']}",
                    f"{suggestion['description']}",
                    ""
                ])
                
                if 'cmake_find' in suggestion:
                    cmake_commands = suggestion['cmake_find']
                    if isinstance(cmake_commands, list):
                        for cmd in cmake_commands:
                            report_lines.append(f"```cmake\n{cmd}\n```")
                    else:
                        report_lines.append(f"```cmake\n{cmake_commands}\n```")
                
                if 'vcpkg' in suggestion:
                    vcpkg_packages = suggestion['vcpkg']
                    if isinstance(vcpkg_packages, list):
                        report_lines.append(f"**vcpkg**: `vcpkg install {' '.join(vcpkg_packages)}`")
                    else:
                        report_lines.append(f"**vcpkg**: `vcpkg install {vcpkg_packages}`")
                
                if 'url' in suggestion:
                    report_lines.append(f"**Documentation**: {suggestion['url']}")
                
                report_lines.append("")
        
        # Add build instructions
        report_lines.extend([
            "## Build Configuration",
            "",
            "Add the following to your CMakeLists.txt:",
            "",
            "```cmake"
        ])
        
        # Collect all cmake_find commands
        cmake_commands = set()
        for dep_name, suggestion in suggestions.items():
            if 'cmake_find' in suggestion:
                commands = suggestion['cmake_find']
                if isinstance(commands, list):
                    cmake_commands.update(commands)
                else:
                    cmake_commands.add(commands)
        
        for cmd in sorted(cmake_commands):
            report_lines.append(cmd)
        
        report_lines.extend([
            "```",
            "",
            "## Package Manager Commands",
            "",
            "### vcpkg",
            "```bash"
        ])
        
        # Collect vcpkg packages
        vcpkg_packages = set()
        for suggestion in suggestions.values():
            if 'vcpkg' in suggestion:
                packages = suggestion['vcpkg']
                if isinstance(packages, list):
                    vcpkg_packages.update(packages)
                else:
                    vcpkg_packages.add(packages)
        
        if vcpkg_packages:
            report_lines.append(f"vcpkg install {' '.join(sorted(vcpkg_packages))}")
        
        report_lines.extend([
            "```",
            "",
            "### Conan",
            "```bash"
        ])
          # Collect conan packages
        for suggestion in suggestions.values():
            if 'conan' in suggestion:
                packages = suggestion['conan']
                if isinstance(packages, list):
                    for pkg in packages:
                        report_lines.append(f"conan install {pkg}")
                else:
                    report_lines.append(f"conan install {packages}")
        
        report_lines.append("```")
        
        # Add ML migration guide if ML frameworks are detected
        ml_guide = self.generate_ml_migration_guide(dependencies)
        if ml_guide:
            report_lines.extend([
                "",
                "---",
                "",
                ml_guide
            ])
        
        return '\n'.join(report_lines)
    
    def create_module_project(self, dependencies: List[ModuleInfo], project_name: str) -> Path:
        """Create a complete C++ project with module dependencies"""
        project_dir = self.output_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Generate main CMakeLists.txt
        suggestions = self.suggest_cpp_alternatives(dependencies)
        main_cmake = self._generate_main_cmake(project_name, suggestions)
        
        with open(project_dir / "CMakeLists.txt", 'w') as f:
            f.write(main_cmake)
        
        # Create conanfile.txt
        conanfile = self._generate_conanfile(suggestions)
        if conanfile:
            with open(project_dir / "conanfile.txt", 'w') as f:
                f.write(conanfile)
        
        # Create vcpkg.json
        vcpkg_json = self._generate_vcpkg_json(project_name, suggestions)
        if vcpkg_json:
            with open(project_dir / "vcpkg.json", 'w') as f:
                f.write(vcpkg_json)
        
        return project_dir
    
    def _generate_main_cmake(self, project_name: str, suggestions: Dict[str, Dict]) -> str:
        """Generate main CMakeLists.txt for the project"""
        cmake_lines = [
            f"cmake_minimum_required(VERSION 3.12)",
            f"project({project_name})",
            "",
            "set(CMAKE_CXX_STANDARD 17)",
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
            "",
            "# Find packages"
        ]
        
        # Add find_package commands
        for suggestion in suggestions.values():
            if 'cmake_find' in suggestion:
                commands = suggestion['cmake_find']
                if isinstance(commands, list):
                    cmake_lines.extend(commands)
                else:
                    cmake_lines.append(commands)
        
        cmake_lines.extend([
            "",
            "# Add executable",
            f"add_executable({project_name} main.cpp)",
            "",
            "# Link libraries",
            "# TODO: Add specific library linking based on dependencies",
            "",
            "# Compiler options",
            f"target_compile_options({project_name} PRIVATE",
            "    $<$<CXX_COMPILER_ID:MSVC>:/W4>",
            "    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>",
            ")"
        ])
        
        return '\n'.join(cmake_lines)
    
    def _generate_conanfile(self, suggestions: Dict[str, Dict]) -> Optional[str]:
        """Generate conanfile.txt"""
        requires = []
        
        for suggestion in suggestions.values():
            if 'conan' in suggestion:
                packages = suggestion['conan']
                if isinstance(packages, list):
                    requires.extend(packages)
                else:
                    requires.append(packages)
        
        if not requires:
            return None
        
        return f"""[requires]
{chr(10).join(requires)}

[generators]
CMakeDeps
CMakeToolchain

[options]

[imports]
"""
    
    def _generate_vcpkg_json(self, project_name: str, suggestions: Dict[str, Dict]) -> Optional[str]:
        """Generate vcpkg.json"""
        dependencies = []
        
        for suggestion in suggestions.values():
            if 'vcpkg' in suggestion:
                packages = suggestion['vcpkg']
                if isinstance(packages, list):
                    dependencies.extend(packages)
                else:
                    dependencies.append(packages)
        
        if not dependencies:
            return None
        
        vcpkg_config = {
            "name": project_name.lower().replace('_', '-'),
            "version": "1.0.0",
            "dependencies": sorted(list(set(dependencies)))
        }
        
        return json.dumps(vcpkg_config, indent=2)
    
    def get_ml_framework_guidance(self, dependencies: List[ModuleInfo]) -> Dict[str, Dict]:
        """Provide specific guidance for ML framework conversion"""
        ml_frameworks = ['torch', 'tensorflow', 'keras', 'sklearn', 'xgboost', 'lightgbm']
        guidance = {}
        
        for dep in dependencies:
            if dep.name in ml_frameworks:
                if dep.name == 'torch':
                    guidance[dep.name] = {
                        'conversion_strategy': 'LibTorch C++ API',
                        'model_conversion': 'Use TorchScript for model serialization',
                        'steps': [
                            '1. Convert PyTorch model to TorchScript (.pt file)',
                            '2. Load model in C++ using torch::jit::load()',
                            '3. Replace Python training code with C++ LibTorch equivalents',
                            '4. Use torch::Tensor for data handling'
                        ],
                        'example_conversion': {
                            'python': 'model = torch.nn.Linear(10, 1)',
                            'cpp': 'torch::nn::Linear linear(10, 1);'
                        },
                        'build_instructions': [
                            '1. Download LibTorch from pytorch.org',
                            '2. Extract to /path/to/libtorch',
                            '3. Add to CMakeLists.txt: set(CMAKE_PREFIX_PATH /path/to/libtorch)',
                            '4. Use: find_package(Torch REQUIRED)',
                            '5. Link: target_link_libraries(your_target ${TORCH_LIBRARIES})'
                        ]
                    }
                elif dep.name == 'sklearn':
                    guidance[dep.name] = {
                        'conversion_strategy': 'mlpack or manual implementation',
                        'model_conversion': 'Export trained model parameters, implement in C++',
                        'steps': [
                            '1. Export model parameters from sklearn (joblib.dump)',
                            '2. Load parameters in C++ (JSON or binary format)',
                            '3. Implement algorithm using mlpack or Eigen',
                            '4. For simple models, manual implementation is feasible'
                        ],
                        'supported_algorithms': [
                            'Linear Regression → mlpack::regression::LinearRegression',
                            'Logistic Regression → mlpack::regression::LogisticRegression',
                            'Random Forest → mlpack::tree::RandomForest',
                            'SVM → mlpack::svm::SVM',
                            'K-Means → mlpack::kmeans::KMeans'
                        ]
                    }
        
        return guidance
    
    def generate_ml_conversion_guide(self, dependencies: List[ModuleInfo]) -> str:
        """Generate detailed ML framework conversion guide"""
        ml_guidance = self.get_ml_framework_guidance(dependencies)
        
        if not ml_guidance:
            return ""
        
        guide_lines = [
            "# Machine Learning Framework Conversion Guide",
            "",
            "This project uses machine learning frameworks that require special attention for C++ conversion.",
            ""
        ]
        
        for framework, guidance in ml_guidance.items():
            guide_lines.extend([
                f"## {framework.upper()} Conversion",
                "",
                f"**Strategy**: {guidance['conversion_strategy']}",
                "",
                "### Conversion Steps:",
            ])
            
            for step in guidance['steps']:
                guide_lines.append(f"- {step}")
            
            guide_lines.append("")
            
            if 'example_conversion' in guidance:
                example = guidance['example_conversion']
                guide_lines.extend([
                    "### Code Example:",
                    "```python",
                    f"# Python: {example['python']}",
                    "```",
                    "```cpp",
                    f"// C++: {example['cpp']}",
                    "```",
                    ""
                ])
            
            if 'build_instructions' in guidance:
                guide_lines.extend([
                    "### Build Instructions:",
                ])
                for instruction in guidance['build_instructions']:
                    guide_lines.append(f"- {instruction}")
                guide_lines.append("")
            
            if 'supported_algorithms' in guidance:
                guide_lines.extend([
                    "### Supported Algorithms:",
                ])
                for algorithm in guidance['supported_algorithms']:
                    guide_lines.append(f"- {algorithm}")
                guide_lines.append("")
        
        # Add general recommendations
        guide_lines.extend([
            "## General Recommendations",
            "",
            "1. **Model Serialization**: Use ONNX format for cross-framework compatibility",
            "2. **Inference vs Training**: Consider if you need training in C++ or just inference",
            "3. **Performance**: C++ implementations can be 2-10x faster than Python",
            "4. **Memory Management**: Use smart pointers for model management",
            "5. **Threading**: Take advantage of C++ threading for parallel processing",
            "",
            "## Alternative Approaches",
            "",
            "1. **Python Extension**: Keep ML code in Python, create C++ extensions for performance-critical parts",
            "2. **REST API**: Keep Python ML service, call from C++ via HTTP",
            "3. **Embedded Python**: Embed Python interpreter in C++ application",
            "4. **Model Serving**: Use TensorFlow Serving, ONNX Runtime, or TorchServe"
        ])
        
        return '\n'.join(guide_lines)
    
    def analyze_ml_complexity(self, dependencies: List[ModuleInfo]) -> Dict[str, str]:
        """Analyze the complexity of ML framework conversions"""
        complexity_analysis = {}
        
        ml_frameworks = {'torch', 'tensorflow', 'sklearn', 'keras', 'transformers', 'xgboost', 'lightgbm'}
        
        for dep in dependencies:
            if dep.name in ml_frameworks and dep.name in self.cpp_equivalents:
                equiv = self.cpp_equivalents[dep.name]
                complexity = equiv.get('complexity', 'medium')
                complexity_analysis[dep.name] = complexity
        
        return complexity_analysis
    
    def generate_ml_migration_guide(self, dependencies: List[ModuleInfo]) -> Optional[str]:
        """Generate specialized migration guide for ML frameworks"""
        ml_deps = [dep for dep in dependencies if dep.name in 
                  {'torch', 'tensorflow', 'sklearn', 'keras', 'transformers', 'xgboost', 'lightgbm'}]
        
        if not ml_deps:
            return None
        
        guide_lines = [
            "# Machine Learning Framework Migration Guide",
            "",
            "This project uses machine learning frameworks that require special attention during C++ conversion.",
            "",
            "## Summary of ML Dependencies",
            ""
        ]
        
        for dep in ml_deps:
            if dep.name in self.cpp_equivalents:
                equiv = self.cpp_equivalents[dep.name]
                complexity = equiv.get('complexity', 'medium')
                
                guide_lines.extend([
                    f"### {dep.name} → {equiv['name']}",
                    f"**Complexity**: {complexity.upper()}",
                    f"**Description**: {equiv['description']}",
                    ""
                ])
                
                if 'notes' in equiv:
                    guide_lines.append("**Important Notes:**")
                    for note in equiv['notes']:
                        guide_lines.append(f"- {note}")
                    guide_lines.append("")
                
                if 'alternatives' in equiv:
                    guide_lines.append("**Alternative Approaches:**")
                    for alt in equiv['alternatives']:
                        guide_lines.append(f"- **{alt['name']}**: {alt['description']}")
                    guide_lines.append("")
        
        # Add migration strategies
        guide_lines.extend([
            "## Migration Strategies",
            "",
            "### Strategy 1: Direct Framework Translation",
            "- Use the C++ API of the same framework (LibTorch, TensorFlow C++)",
            "- Maintains full compatibility with Python models",
            "- Requires framework-specific setup and dependencies",
            "",
            "### Strategy 2: Model Export + Inference Engine",
            "- Export Python models to ONNX format",
            "- Use ONNX Runtime for C++ inference",
            "- Lighter weight, good for inference-only scenarios",
            "",
            "### Strategy 3: Algorithm Reimplementation",
            "- Reimplement algorithms using general-purpose C++ libraries",
            "- More work but better control and potential optimization",
            "- Good for simple ML algorithms",
            "",
            "## Recommended Approach",
            ""
        ])
        
        # Provide specific recommendations based on detected frameworks
        if any(dep.name in ['torch', 'tensorflow'] for dep in ml_deps):
            guide_lines.extend([
                "For deep learning frameworks (PyTorch/TensorFlow):",
                "1. **For inference**: Export models to ONNX and use ONNX Runtime",
                "2. **For training**: Use LibTorch C++ API or TensorFlow C++",
                "3. **For simple networks**: Consider reimplementation with Eigen",
                ""
            ])
        
        if any(dep.name in ['sklearn'] for dep in ml_deps):
            guide_lines.extend([
                "For scikit-learn:",
                "1. **Simple algorithms**: Reimplement using Eigen",
                "2. **Complex pipelines**: Use mlpack or Shark-ML",
                "3. **Tree models**: Consider XGBoost or LightGBM C++",
                ""
            ])
        
        guide_lines.extend([
            "## Implementation Steps",
            "",
            "1. **Analyze your models**: Determine if you need training or just inference",
            "2. **Choose strategy**: Based on complexity and requirements",
            "3. **Set up environment**: Install chosen C++ ML framework",
            "4. **Port incrementally**: Start with data loading, then model inference",
            "5. **Optimize**: Profile and optimize performance-critical sections",
            "",
            "## Useful Resources",
            "",
            "- [LibTorch Tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html)",
            "- [TensorFlow C++ Guide](https://www.tensorflow.org/guide/extend/cc)",
            "- [ONNX Runtime C++](https://onnxruntime.ai/docs/api/c/)",
            "- [mlpack Documentation](https://mlpack.org/doc.html)",
            ""
        ])
        
        return '\n'.join(guide_lines)
    
    def initialize_dynamic_analyzer(self, translator_instance=None):
        """Initialize the dynamic analyzer with a translator reference"""
        self.dynamic_analyzer = DynamicModuleAnalyzer(translator_instance)
    
    def analyze_unknown_module(self, module_name: str, module_path: str = None) -> 'ModuleAnalysis':
        """Analyze an unknown module using dynamic analysis"""
        if not self.dynamic_analyzer:
            self.initialize_dynamic_analyzer()
        
        if module_path:
            return self.dynamic_analyzer.analyze_module(module_path)
        else:
            # Try to find module in PyPI and download
            downloaded_path = self.download_module_source(module_name)
            if downloaded_path:
                # Find main module file
                main_file = self._find_main_module_file(downloaded_path, module_name)
                if main_file:
                    return self.dynamic_analyzer.analyze_module(main_file)        # Return a basic analysis if we can't find the module
        # Use globals() to access classes defined later in the file
        ModuleAnalysisClass = globals().get('ModuleAnalysis')
        ModuleComplexityClass = globals().get('ModuleComplexity')
        
        if ModuleAnalysisClass and ModuleComplexityClass:
            return ModuleAnalysisClass(
                name=module_name,
                complexity=ModuleComplexityClass.COMPLEX,
                translatable_functions=[],
                translatable_classes=[],
                dependencies=[],
                c_extensions=[],
                estimated_effort="high",
                translation_notes=["Module not found or not analyzable"],
                suggested_approach="find_cpp_alternative"
            )
        else:
            # Fallback - return a simple dict
            return {
                'name': module_name,
                'complexity': 'complex',
                'translatable_functions': [],
                'translatable_classes': [],
                'dependencies': [],
                'c_extensions': [],
                'estimated_effort': 'high',
                'translation_notes': ["Module not found or not analyzable"],
                'suggested_approach': 'find_cpp_alternative'
            }

    # ...existing code...
    def dynamic_analysis(self, module_name: str) -> 'DynamicModuleInfo':
        """Perform dynamic analysis of a Python module"""
        info = DynamicModuleInfo(name=module_name)
        
        try:
            # Import module
            module = importlib.import_module(module_name)
            
            # Check if module has a __file__ attribute
            if hasattr(module, '__file__') and module.__file__:
                info.is_analyzable = True
                
                # Analyze functions and classes
                functions = [f for f in dir(module) if callable(getattr(module, f))]
                classes = [c for c in dir(module) if isinstance(getattr(module, c), type)]
                
                # Heuristic: if module has no functions or classes, it's not analyzable
                if not functions and not classes:
                    info.is_analyzable = False
                    info.analysis_notes.append("Module has no functions or classes")
                
                # Generate C++ stub code as a placeholder
                cpp_stub = self.generate_cpp_stub(module)
                info.generated_cpp = cpp_stub
                
                # Basic quality assessment
                if len(cpp_stub) < 50:
                    info.auto_translation_quality = "poor"
                    info.analysis_notes.append("Generated C++ code is too short")
                else:
                    info.auto_translation_quality = "good"
            
            else:
                info.analysis_notes.append("Module does not have a __file__ attribute")
        
        except Exception as e:
            info.analysis_notes.append(f"Error during dynamic analysis: {e}")
        
        return info
    
    def generate_cpp_stub(self, module) -> str:
        """Generate a C++ stub for a Python module (for dynamic analysis)"""
        cpp_code = []
        
        try:
            # Write module docstring
            if module.__doc__:
                cpp_code.append(f"/* {module.__doc__} */")
            
            cpp_code.append(f"#include \"{module.__name__}.h\"")
            cpp_code.append(f"using namespace {module.__name__};")
            cpp_code.append("")
            
            # Analyze functions
            functions = [f for f in dir(module) if callable(getattr(module, f))]
            for func_name in functions:
                func = getattr(module, func_name)
                
                # Skip built-in functions
                if inspect.isbuiltin(func):
                    continue
                
                # Generate C++ function signature
                signature = self.generate_cpp_signature(func)
                cpp_code.append(signature)
                cpp_code.append("{")
                cpp_code.append(f"    // TODO: Implement {func_name}")
                cpp_code.append("}")
                cpp_code.append("")
            
            # Analyze classes
            classes = [c for c in dir(module) if isinstance(getattr(module, c), type)]
            for class_name in classes:
                cls = getattr(module, class_name)
                
                # Generate C++ class definition
                cpp_code.append(f"class {class_name} {{")
                cpp_code.append("public:")
                
                # Analyze class methods
                methods = [m for m in dir(cls) if callable(getattr(cls, m)) and not m.startswith('__')]
                for method_name in methods:
                    method = getattr(cls, method_name)
                    
                    # Generate C++ method signature
                    signature = self.generate_cpp_signature(method, class_name)
                    cpp_code.append(f"    {signature};")
                
                cpp_code.append("};")
                cpp_code.append("")
        
        except Exception as e:
            cpp_code.append(f"// Error generating C++ stub: {e}")
        
        return "\n".join(cpp_code)
    
    def generate_cpp_signature(self, func, class_name: str = "") -> str:
        """Generate C++ function or method signature from Python function"""
        signature = ""
        
        try:
            params = inspect.signature(func).parameters
            param_list = []
            
            for param in params.values():
                # Default to double type for simplicity
                param_type = "double"
                if param.annotation != inspect.Parameter.empty:
                    # Map Python types to C++ types (basic mapping)
                    if param.annotation == int:
                        param_type = "int"
                    elif param.annotation == float:
                        param_type = "double"
                    elif param.annotation == str:
                        param_type = "std::string"
                    elif param.annotation == bool:
                        param_type = "bool"
                  # Add parameter to list
                param_list.append(f"{param_type} {param.name}")
            
            if class_name:
                # Method signature
                signature = f"void {class_name}::{func.__name__}(" + ", ".join(param_list) + ")"
            else:
                # Free function signature
                signature = f"void {func.__name__}(" + ", ".join(param_list) + ")"
        
        except Exception as e:
            signature = "// Error generating signature"
        
        return signature


@dataclass 
class DynamicModuleInfo:
    """Information about a dynamically analyzed module"""
    name: str
    is_analyzable: bool = False
    auto_translation_quality: str = "unknown"
    analysis_notes: List[str] = None
    generated_cpp: str = ""
    
    def __post_init__(self):
        if self.analysis_notes is None:
            self.analysis_notes = []


class ModuleComplexity(Enum):
    """Module complexity levels for translation"""
    SIMPLE = "simple"      # Pure Python, basic functions/classes
    MODERATE = "moderate"  # Some advanced features, decorators, etc.
    COMPLEX = "complex"    # Heavy use of metaclasses, dynamic features
    NATIVE = "native"      # Contains C extensions, cannot translate


@dataclass
class ModuleAnalysis:
    """Analysis result for a Python module"""
    name: str
    complexity: ModuleComplexity
    translatable_functions: List[str]
    translatable_classes: List[str]
    dependencies: List[str]
    c_extensions: List[str]
    estimated_effort: str  # "low", "medium", "high"
    translation_notes: List[str]
    suggested_approach: str


class DynamicModuleAnalyzer:
    """Analyzes Python modules dynamically to determine translation feasibility"""
    
    def __init__(self, translator_instance=None):
        self.translator = translator_instance
        self.analysis_cache = {}
        
    def analyze_module(self, module_path: Union[str, Path]) -> ModuleAnalysis:
        """Analyze a Python module for translation feasibility"""
        module_path = Path(module_path)
        
        if str(module_path) in self.analysis_cache:
            return self.analysis_cache[str(module_path)]
        
        try:
            # Parse the module's AST
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            analysis = self._analyze_ast(tree, module_path.stem)
            
            # Cache the result
            self.analysis_cache[str(module_path)] = analysis
            return analysis
            
        except Exception as e:
            # Return a basic analysis if parsing fails
            return ModuleAnalysis(
                name=module_path.stem,
                complexity=ModuleComplexity.COMPLEX,
                translatable_functions=[],
                translatable_classes=[],
                dependencies=[],
                c_extensions=[],
                estimated_effort="high",
                translation_notes=[f"Failed to parse: {e}"],
                suggested_approach="manual_review"
            )
    
    def _analyze_ast(self, tree: ast.AST, module_name: str) -> ModuleAnalysis:
        """Analyze AST to determine module characteristics"""
        analysis = ModuleAnalysis(
            name=module_name,
            complexity=ModuleComplexity.SIMPLE,
            translatable_functions=[],
            translatable_classes=[],
            dependencies=[],
            c_extensions=[],
            estimated_effort="low",
            translation_notes=[],
            suggested_approach="automatic"
        )
        
        complexity_score = 0
        
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._analyze_imports(node, analysis)
            
            # Check for function definitions
            elif isinstance(node, ast.FunctionDef):
                func_complexity = self._analyze_function(node)
                analysis.translatable_functions.append(node.name)
                complexity_score += func_complexity
            
            # Check for class definitions
            elif isinstance(node, ast.ClassDef):
                class_complexity = self._analyze_class(node)
                analysis.translatable_classes.append(node.name)
                complexity_score += class_complexity
            
            # Check for complex features
            elif isinstance(node, ast.Lambda):
                complexity_score += 2
                analysis.translation_notes.append("Contains lambda expressions")
            
            elif isinstance(node, ast.ListComp):
                complexity_score += 1
                analysis.translation_notes.append("Contains list comprehensions")
            
            elif isinstance(node, ast.DictComp):
                complexity_score += 1
                analysis.translation_notes.append("Contains dict comprehensions")
            
            elif isinstance(node, ast.GeneratorExp):
                complexity_score += 3
                analysis.translation_notes.append("Contains generator expressions")
            
            elif isinstance(node, ast.Yield):
                complexity_score += 4
                analysis.translation_notes.append("Contains generators/yield")
            
            elif isinstance(node, ast.AsyncFunctionDef):
                complexity_score += 5
                analysis.translation_notes.append("Contains async functions")
            
            elif isinstance(node, ast.With):
                complexity_score += 2
                analysis.translation_notes.append("Contains context managers")
        
        # Determine complexity level
        if complexity_score <= 5:
            analysis.complexity = ModuleComplexity.SIMPLE
            analysis.estimated_effort = "low"
        elif complexity_score <= 15:
            analysis.complexity = ModuleComplexity.MODERATE
            analysis.estimated_effort = "medium"
        else:
            analysis.complexity = ModuleComplexity.COMPLEX
            analysis.estimated_effort = "high"
        
        # Determine suggested approach
        if analysis.c_extensions:
            analysis.complexity = ModuleComplexity.NATIVE
            analysis.suggested_approach = "find_cpp_alternative"
        elif complexity_score <= 10:
            analysis.suggested_approach = "automatic"
        elif complexity_score <= 20:
            analysis.suggested_approach = "semi_automatic"
        else:
            analysis.suggested_approach = "manual_review"
        
        return analysis
    
    def _analyze_imports(self, node: Union[ast.Import, ast.ImportFrom], analysis: ModuleAnalysis):
        """Analyze import statements"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis.dependencies.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                analysis.dependencies.append(node.module)
                
                # Check for known C extension modules
                c_extension_modules = {
                    'numpy', 'scipy', 'pandas', 'cv2', 'PIL', 'torch', 
                    'tensorflow', 'matplotlib', 'lxml', 'psycopg2'
                }
                if node.module in c_extension_modules:
                    analysis.c_extensions.append(node.module)
    
    def _analyze_function(self, node: ast.FunctionDef) -> int:
        """Analyze function complexity and return complexity score"""
        complexity = 0
        
        # Check for decorators
        if node.decorator_list:
            complexity += len(node.decorator_list) * 2
        
        # Check for complex arguments
        if node.args.vararg or node.args.kwarg:
            complexity += 2
        
        # Check for default arguments
        complexity += len(node.args.defaults)
        
        # Check for annotations
        if node.returns or any(arg.annotation for arg in node.args.args):
            complexity -= 1  # Type hints make translation easier
        
        return max(0, complexity)
    
    def _analyze_class(self, node: ast.ClassDef) -> int:
        """Analyze class complexity and return complexity score"""
        complexity = 0
        
        # Multiple inheritance
        if len(node.bases) > 1:
            complexity += 3
        
        # Metaclasses
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                complexity += 5
        
        # Decorators
        complexity += len(node.decorator_list) * 2
        
        # Special methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name.startswith('__') and item.name.endswith('__'):
                    complexity += 1
        
        return complexity
    
    def translate_module(self, module_path: Union[str, Path], output_dir: Optional[Path] = None) -> ConversionResult:
        """Translate a Python module to C++ using analysis-guided approach"""
        analysis = self.analyze_module(module_path)
        
        if analysis.complexity == ModuleComplexity.NATIVE:
            return ConversionResult(
                success=False,
                module_name=analysis.name,
                error_message="Module contains C extensions - cannot translate automatically"
            )
        
        try:
            # Use the main translator if available
            if self.translator:
                cpp_code = self.translator.translate_file(str(module_path))
            else:
                # Fallback: basic translation
                with open(module_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                cpp_code = self._basic_translation(tree, analysis)
            
            # Write output if directory specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                cpp_file = output_dir / f"{analysis.name}.cpp"
                header_file = output_dir / f"{analysis.name}.h"
                
                with open(cpp_file, 'w', encoding='utf-8') as f:
                    f.write(cpp_code)
                
                # Generate header file
                header_code = self._generate_header(analysis)
                with open(header_file, 'w', encoding='utf-8') as f:
                    f.write(header_code)
                
                return ConversionResult(
                    success=True,
                    module_name=analysis.name,
                    cpp_files=[cpp_file],
                    header_files=[header_file]
                )
            else:
                return ConversionResult(
                    success=True,
                    module_name=analysis.name,
                    cpp_files=[],
                    header_files=[]
                )
                
        except Exception as e:
            return ConversionResult(
                success=False,
                module_name=analysis.name,
                error_message=f"Translation failed: {e}"
            )
    
    def _basic_translation(self, tree: ast.AST, analysis: ModuleAnalysis) -> str:
        """Basic C++ translation when main translator is not available"""
        lines = [
            f"// Auto-translated from Python module: {analysis.name}",
            f"// Complexity: {analysis.complexity.value}",
            f"// Estimated effort: {analysis.estimated_effort}",
            "",
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "#include <memory>",
            ""
        ]
        
        # Add basic class/function stubs
        for class_name in analysis.translatable_classes:
            lines.extend([
                f"class {class_name} {{",
                "public:",
                f"    {class_name}();",
                f"    ~{class_name}();",
                "    // TODO: Implement class methods",
                "};",
                ""
            ])
        
        for func_name in analysis.translatable_functions:
            lines.extend([
                f"auto {func_name}() {{",
                "    // TODO: Implement function body",
                "    return 0;",
                "}",
                ""
            ])
        
        return '\n'.join(lines)
    
    def _generate_header(self, analysis: ModuleAnalysis) -> str:
        """Generate C++ header file for the module"""
        lines = [
            f"#pragma once",
            f"// Auto-generated header for Python module: {analysis.name}",
            "",
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "#include <memory>",
            ""
        ]
        
        # Forward declarations
        for class_name in analysis.translatable_classes:
            lines.append(f"class {class_name};")
        
        if analysis.translatable_classes:
            lines.append("")
        
        # Function declarations
        for func_name in analysis.translatable_functions:
            lines.append(f"auto {func_name}();")
        
        return '\n'.join(lines)
    
    def generate_analysis_report(self, analysis: ModuleAnalysis) -> str:
        """Generate a detailed analysis report"""
        lines = [
            f"# Module Analysis Report: {analysis.name}",
            "",
            f"**Complexity**: {analysis.complexity.value.upper()}",
            f"**Estimated Effort**: {analysis.estimated_effort.upper()}",
            f"**Suggested Approach**: {analysis.suggested_approach.replace('_', ' ').title()}",
            "",
            "## Translatable Components",
            "",
            f"**Functions**: {len(analysis.translatable_functions)}",
        ]
        
        if analysis.translatable_functions:
            for func in analysis.translatable_functions:
                lines.append(f"- {func}")
        
        lines.extend([
            "",
            f"**Classes**: {len(analysis.translatable_classes)}",
        ])
        
        if analysis.translatable_classes:
            for cls in analysis.translatable_classes:
                lines.append(f"- {cls}")
        
        lines.extend([
            "",
            "## Dependencies",
            ""
        ])
        
        if analysis.dependencies:
            for dep in analysis.dependencies:
                lines.append(f"- {dep}")
        else:
            lines.append("No external dependencies found")
        
        if analysis.c_extensions:
            lines.extend([
                "",
                "## C Extensions Detected",
                ""
            ])
            for ext in analysis.c_extensions:
                lines.append(f"- {ext} (requires C++ equivalent)")
        
        if analysis.translation_notes:
            lines.extend([
                "",
                "## Translation Notes",
                ""
            ])
            for note in analysis.translation_notes:
                lines.append(f"- {note}")
        
        lines.extend([
            "",
            "## Recommended Strategy",
            ""
        ])
        
        if analysis.suggested_approach == "automatic":
            lines.extend([
                "This module appears suitable for automatic translation.",
                "Most features should convert directly to C++."
            ])
        elif analysis.suggested_approach == "semi_automatic":
            lines.extend([
                "This module can be partially translated automatically.",
                "Some manual intervention may be required for complex features."
            ])
        elif analysis.suggested_approach == "manual_review":
            lines.extend([
                "This module requires careful manual review.",
                "Consider breaking it into smaller, simpler components."
            ])
        else:
            lines.extend([
                "This module cannot be automatically translated.",
                "Look for existing C++ alternatives or consider keeping it in Python."
            ])
        
        return '\n'.join(lines)
    
    def auto_translate_module(self, module_name: str, output_dir: Path = None) -> ConversionResult:
        """Automatically translate an unknown module to C++"""
        if not self.dynamic_analyzer:
            self.initialize_dynamic_analyzer()
        
        # First try to download the module
        downloaded_path = self.download_module_source(module_name)
        if not downloaded_path:
            return ConversionResult(
                success=False,
                module_name=module_name,
                error_message="Could not download module source from PyPI"
            )
        
        # Find main module file
        main_file = self._find_main_module_file(downloaded_path, module_name)
        if not main_file:
            return ConversionResult(
                success=False,
                module_name=module_name,
                error_message="Could not find main module file"
            )
        
        # Translate using dynamic analyzer
        return self.dynamic_analyzer.translate_module(main_file, output_dir or self.output_dir)
    
    def _find_main_module_file(self, directory: Path, module_name: str) -> Optional[Path]:
        """Find the main module file in a downloaded package"""
        possible_files = [
            directory / f"{module_name}.py",
            directory / f"{module_name}" / "__init__.py",
            directory / "src" / f"{module_name}.py",
            directory / "src" / f"{module_name}" / "__init__.py",
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return file_path
        
        # If no obvious match, look for Python files
        for file_path in directory.rglob("*.py"):
            if file_path.name == f"{module_name}.py" or file_path.parent.name == module_name:
                return file_path
        
        return None