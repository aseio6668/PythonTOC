"""
Module dependency manager for Python to C++ translation
"""

import ast
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import tempfile
import os


class DependencyInfo:
    """Information about a Python dependency"""
    def __init__(self, name: str, version: Optional[str] = None):
        self.name = name
        self.version = version
        self.is_standard_library = name in STANDARD_LIBRARY_MODULES
        self.is_c_extension = False
        self.cpp_equivalent = None
        self.source_url = None
        self.can_convert = False
        self.conversion_notes = []


class ModuleDependencyManager:
    """Manages Python module dependencies for C++ conversion"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.converted_modules: Dict[str, str] = {}  # module_name -> cpp_path
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Load known mappings
        self.load_known_mappings()
    
    def load_known_mappings(self):
        """Load known Python to C++ library mappings"""
        self.cpp_mappings = {
            # Standard library with C++ equivalents
            'math': {
                'cpp_lib': 'cmath',
                'type': 'standard',
                'notes': 'Direct mapping to C++ math functions'
            },
            'random': {
                'cpp_lib': 'random',
                'type': 'standard',
                'notes': 'Use C++11 random library'
            },
            'time': {
                'cpp_lib': 'chrono',
                'type': 'standard',
                'notes': 'Use C++11 chrono library'
            },
            'os': {
                'cpp_lib': 'filesystem',
                'type': 'standard',
                'notes': 'Use C++17 filesystem library'
            },
            're': {
                'cpp_lib': 'regex',
                'type': 'standard',
                'notes': 'Use C++11 regex library'
            },
            
            # Third-party libraries with known C++ equivalents
            'numpy': {
                'cpp_lib': 'eigen3',
                'type': 'external',
                'package_manager': {
                    'vcpkg': 'eigen3',
                    'conan': 'eigen/3.4.0',
                    'apt': 'libeigen3-dev',
                    'brew': 'eigen'
                },
                'notes': 'Use Eigen for linear algebra operations'
            },
            'opencv': {
                'cpp_lib': 'opencv',
                'type': 'external',
                'package_manager': {
                    'vcpkg': 'opencv',
                    'conan': 'opencv/4.5.5',
                    'apt': 'libopencv-dev',
                    'brew': 'opencv'
                },
                'notes': 'Direct C++ API mapping'
            },
            'requests': {
                'cpp_lib': 'libcurl',
                'type': 'external',
                'package_manager': {
                    'vcpkg': 'curl',
                    'conan': 'libcurl/7.80.0',
                    'apt': 'libcurl4-openssl-dev',
                    'brew': 'curl'
                },
                'notes': 'Use libcurl for HTTP requests'
            },
            'sqlite3': {
                'cpp_lib': 'sqlite3',
                'type': 'external',
                'package_manager': {
                    'vcpkg': 'sqlite3',
                    'conan': 'sqlite3/3.38.5',
                    'apt': 'libsqlite3-dev',
                    'brew': 'sqlite'
                },
                'notes': 'Direct C API mapping'
            },
            
            # Image processing libraries
            'PIL': {
                'cpp_lib': 'opencv',
                'type': 'external',
                'alternative': 'stb_image',
                'package_manager': {
                    'vcpkg': 'opencv[core,imgproc,imgcodecs]',
                    'conan': 'opencv/4.5.5',
                    'header_only': 'https://github.com/nothings/stb'
                },
                'notes': 'Use OpenCV for image processing or stb_image for simple operations'
            },
            'Pillow': {
                'cpp_lib': 'opencv',
                'type': 'external',
                'alternative': 'stb_image',
                'package_manager': {
                    'vcpkg': 'opencv[core,imgproc,imgcodecs]',
                    'conan': 'opencv/4.5.5'
                },
                'notes': 'Same as PIL - use OpenCV or stb_image'
            },
            
            # Scientific computing
            'scipy': {
                'cpp_lib': 'eigen3',
                'type': 'external',
                'alternative': 'gsl',
                'package_manager': {
                    'vcpkg': 'eigen3',
                    'conan': 'eigen/3.4.0',
                    'gsl_vcpkg': 'gsl',
                    'gsl_apt': 'libgsl-dev'
                },
                'notes': 'Use Eigen for linear algebra, GSL for scientific functions'
            },
            
            # CLI libraries
            'click': {
                'cpp_lib': 'cli11',
                'type': 'external',
                'alternative': 'argparse',
                'package_manager': {
                    'vcpkg': 'cli11',
                    'conan': 'cli11/2.3.2',
                    'header_only': 'https://github.com/CLIUtils/CLI11'
                },
                'notes': 'Use CLI11 for command line parsing'
            },
            'argparse': {
                'cpp_lib': 'cli11',
                'type': 'external',
                'package_manager': {
                    'vcpkg': 'cli11',
                    'conan': 'cli11/2.3.2'
                },
                'notes': 'Use CLI11 for command line parsing'
            },
            
            # Pure Python modules that can be converted
            'pure_python_candidates': {
                'dataclasses',
                'enum',
                'collections',
                'itertools',
                'functools',
                'typing'
            }
        }
    
    def analyze_dependencies(self, python_files: List[str]) -> Dict[str, DependencyInfo]:
        """Analyze dependencies from Python files"""
        all_imports = set()
        
        for file_path in python_files:
            imports = self._extract_imports(file_path)
            all_imports.update(imports)
        
        # Create dependency info for each import
        for module_name in all_imports:
            dep_info = DependencyInfo(module_name)
            
            # Check if it's a known mapping
            if module_name in self.cpp_mappings:
                dep_info.cpp_equivalent = self.cpp_mappings[module_name]
                dep_info.conversion_notes.append(f"Known mapping: {module_name} -> {dep_info.cpp_equivalent['cpp_lib']}")
            
            # Check if it's pure Python and convertible
            elif self._is_pure_python_module(module_name):
                dep_info.can_convert = True
                dep_info.conversion_notes.append(f"Pure Python module - can be converted")
                dep_info.source_url = f"https://pypi.org/pypi/{module_name}/json"
            
            # Check if it's a C extension
            elif self._is_c_extension(module_name):
                dep_info.is_c_extension = True
                dep_info.conversion_notes.append(f"C extension - requires manual porting or alternative")
            
            else:
                dep_info.conversion_notes.append(f"Unknown module - requires investigation")
            
            self.dependencies[module_name] = dep_info
        
        return self.dependencies
    
    def _extract_imports(self, file_path: str) -> Set[str]:
        """Extract all imports from a Python file"""
        imports = set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                            
            except SyntaxError as e:
                print(f"Warning: Could not parse {file_path}: {e}")
        
        return imports
    
    def _is_pure_python_module(self, module_name: str) -> bool:
        """Check if a module is pure Python and can potentially be converted"""
        # Try to import and check if it has __file__
        try:
            import importlib
            module = importlib.import_module(module_name)
            
            if hasattr(module, '__file__') and module.__file__:
                return module.__file__.endswith('.py')
            
            # Check against known pure Python modules
            return module_name in self.cpp_mappings.get('pure_python_candidates', set())
            
        except ImportError:
            # Module not installed, check against known list
            return module_name in self.cpp_mappings.get('pure_python_candidates', set())
    
    def _is_c_extension(self, module_name: str) -> bool:
        """Check if a module is a C extension"""
        try:
            import importlib
            module = importlib.import_module(module_name)
            
            if hasattr(module, '__file__') and module.__file__:
                return module.__file__.endswith(('.so', '.dll', '.pyd'))
            
            return False
        except ImportError:
            return False
    
    def download_and_convert_module(self, module_name: str, output_dir: Optional[str] = None) -> Optional[str]:
        """Download a pure Python module and convert it to C++"""
        if module_name not in self.dependencies:
            return None
        
        dep_info = self.dependencies[module_name]
        if not dep_info.can_convert:
            return None
        
        if output_dir is None:
            output_dir = self.workspace_path / "converted_modules"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get module source from PyPI
            source_files = self._download_module_source(module_name)
            
            if source_files:
                # Convert each Python file to C++
                from translator.translator import PythonToCppTranslator
                translator = PythonToCppTranslator(
                    include_headers=True,
                    namespace=f"py_{module_name}",
                    verbose=True
                )
                
                cpp_files = []
                for py_file in source_files:
                    try:
                        cpp_code = translator.translate_file(py_file)
                        
                        cpp_file = output_path / f"{Path(py_file).stem}.cpp"
                        with open(cpp_file, 'w') as f:
                            f.write(cpp_code)
                        
                        cpp_files.append(str(cpp_file))
                        
                    except Exception as e:
                        print(f"Warning: Could not convert {py_file}: {e}")
                
                if cpp_files:
                    # Generate a combined header and implementation
                    self._generate_module_wrapper(module_name, cpp_files, output_path)
                    self.converted_modules[module_name] = str(output_path / f"{module_name}.hpp")
                    return self.converted_modules[module_name]
            
        except Exception as e:
            print(f"Error converting module {module_name}: {e}")
        
        return None
    
    def _download_module_source(self, module_name: str) -> List[str]:
        """Download source code for a Python module from PyPI"""
        try:
            # Get package info from PyPI
            url = f"https://pypi.org/pypi/{module_name}/json"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read())
            
            # Find source distribution
            releases = data.get('releases', {})
            if not releases:
                return []
            
            latest_version = data['info']['version']
            files = releases.get(latest_version, [])
            
            source_file = None
            for file_info in files:
                if file_info['filename'].endswith('.tar.gz') and 'source' in file_info.get('packagetype', ''):
                    source_file = file_info
                    break
            
            if not source_file:
                return []
            
            # Download and extract
            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / source_file['filename']
                
                urllib.request.urlretrieve(source_file['url'], archive_path)
                
                # Extract and find Python files
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                # Find Python files
                python_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('test_'):
                            python_files.append(os.path.join(root, file))
                
                # Copy to a permanent location
                module_source_dir = self.workspace_path / "module_sources" / module_name
                module_source_dir.mkdir(parents=True, exist_ok=True)
                
                copied_files = []
                for py_file in python_files:
                    dest_file = module_source_dir / Path(py_file).name
                    with open(py_file, 'r', encoding='utf-8') as src:
                        with open(dest_file, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                    copied_files.append(str(dest_file))
                
                return copied_files
                
        except Exception as e:
            print(f"Error downloading {module_name}: {e}")
            return []
    
    def _generate_module_wrapper(self, module_name: str, cpp_files: List[str], output_dir: Path):
        """Generate a wrapper header/implementation for a converted module"""
        header_content = f"""#ifndef PY_{module_name.upper()}_HPP
#define PY_{module_name.upper()}_HPP

// Auto-generated C++ wrapper for Python module '{module_name}'
// Generated by Python to C++ Translator

#include <iostream>
#include <string>
#include <vector>
#include <memory>

namespace py_{module_name} {{

// Forward declarations
{self._generate_forward_declarations(cpp_files)}

// Include converted implementations
{self._generate_include_statements(cpp_files)}

}} // namespace py_{module_name}

#endif // PY_{module_name.upper()}_HPP
"""
        
        header_file = output_dir / f"{module_name}.hpp"
        with open(header_file, 'w') as f:
            f.write(header_content)
    
    def _generate_forward_declarations(self, cpp_files: List[str]) -> str:
        """Generate forward declarations from converted C++ files"""
        # This would analyze the C++ files and extract class/function declarations
        return "// TODO: Extract forward declarations from converted files"
    
    def _generate_include_statements(self, cpp_files: List[str]) -> str:
        """Generate include statements for converted C++ files"""
        includes = []
        for cpp_file in cpp_files:
            rel_path = Path(cpp_file).name
            includes.append(f'#include "{rel_path}"')
        return '\n'.join(includes)
    
    def generate_dependency_report(self) -> str:
        """Generate a comprehensive dependency analysis report"""
        report_lines = [
            "# Python to C++ Dependency Analysis Report",
            "",
            f"## Summary",
            f"- Total dependencies: {len(self.dependencies)}",
            f"- Convertible modules: {sum(1 for d in self.dependencies.values() if d.can_convert)}",
            f"- Known C++ mappings: {sum(1 for d in self.dependencies.values() if d.cpp_equivalent)}",
            f"- C extensions: {sum(1 for d in self.dependencies.values() if d.is_c_extension)}",
            "",
            "## Dependency Details",
            ""
        ]
        
        for name, dep_info in self.dependencies.items():
            report_lines.extend([
                f"### {name}",
                f"- **Type**: {'Standard Library' if dep_info.is_standard_library else 'Third-party'}",
                f"- **Can Convert**: {'Yes' if dep_info.can_convert else 'No'}",
                f"- **C Extension**: {'Yes' if dep_info.is_c_extension else 'No'}",
            ])
            
            if dep_info.cpp_equivalent:
                equiv = dep_info.cpp_equivalent
                report_lines.extend([
                    f"- **C++ Equivalent**: {equiv['cpp_lib']}",
                    f"- **Type**: {equiv['type']}",
                ])
                
                if 'package_manager' in equiv:
                    report_lines.append("- **Installation**:")
                    for pm, pkg in equiv['package_manager'].items():
                        report_lines.append(f"  - {pm}: `{pkg}`")
            
            if dep_info.conversion_notes:
                report_lines.extend([
                    "- **Notes**:",
                    *[f"  - {note}" for note in dep_info.conversion_notes]
                ])
            
            report_lines.append("")
        
        return '\n'.join(report_lines)
    
    def generate_build_configuration(self, build_system: str = "cmake") -> str:
        """Generate build configuration with dependencies"""
        if build_system.lower() == "cmake":
            return self._generate_cmake_with_deps()
        elif build_system.lower() == "conan":
            return self._generate_conanfile()
        else:
            return "# Unsupported build system"
    
    def _generate_cmake_with_deps(self) -> str:
        """Generate CMakeLists.txt with all dependencies"""
        cmake_lines = [
            "cmake_minimum_required(VERSION 3.12)",
            "project(converted_python_project)",
            "",
            "set(CMAKE_CXX_STANDARD 17)",
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
            "",
            "# Find packages"
        ]
        
        vcpkg_packages = []
        system_packages = []
        
        for name, dep_info in self.dependencies.items():
            if dep_info.cpp_equivalent and 'package_manager' in dep_info.cpp_equivalent:
                pm = dep_info.cpp_equivalent['package_manager']
                if 'vcpkg' in pm:
                    vcpkg_packages.append(pm['vcpkg'])
                elif 'apt' in pm:
                    system_packages.append(pm['apt'])
        
        # Add find_package calls
        for pkg in set(vcpkg_packages):
            cmake_lines.append(f"find_package({pkg} REQUIRED)")
        
        cmake_lines.extend([
            "",
            "# Source files",
            "file(GLOB_RECURSE SOURCES \"*.cpp\")",
            "",
            "# Add executable",
            "add_executable(main ${SOURCES})",
            "",
            "# Link libraries"
        ])
        
        # Add target_link_libraries calls
        if vcpkg_packages:
            for pkg in set(vcpkg_packages):
                cmake_lines.append(f"target_link_libraries(main PRIVATE {pkg}::{pkg})")
        
        cmake_lines.extend([
            "",
            "# Compiler options",
            "target_compile_options(main PRIVATE",
            "    $<$<CXX_COMPILER_ID:MSVC>:/W4>",
            "    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>",
            ")"
        ])
        
        return '\n'.join(cmake_lines)
    
    def _generate_conanfile(self) -> str:
        """Generate conanfile.txt with dependencies"""
        conan_deps = []
        
        for name, dep_info in self.dependencies.items():
            if dep_info.cpp_equivalent and 'package_manager' in dep_info.cpp_equivalent:
                pm = dep_info.cpp_equivalent['package_manager']
                if 'conan' in pm:
                    conan_deps.append(pm['conan'])
        
        conanfile_content = """[requires]
""" + '\n'.join(set(conan_deps)) + """

[generators]
CMakeDeps
CMakeToolchain

[options]

[imports]
"""
        
        return conanfile_content


# Standard library modules (Python 3.10+)
STANDARD_LIBRARY_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
    'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes', 'curses',
    'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email',
    'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass', 'gettext',
    'glob', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'imaplib',
    'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
    'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox',
    'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'multiprocessing',
    'netrc', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'pprint', 'profile', 'pstats', 'pty', 'pwd',
    'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline',
    'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
    'sndhdr', 'socket', 'socketserver', 'sqlite3', 'ssl', 'stat', 'statistics',
    'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symtable', 'sys',
    'sysconfig', 'tabnanny', 'tarfile', 'tempfile', 'termios', 'textwrap', 'threading',
    'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace', 'traceback',
    'tracemalloc', 'tty', 'turtle', 'types', 'typing', 'unicodedata', 'unittest',
    'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
    'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile',
    'zipimport', 'zlib'
}
