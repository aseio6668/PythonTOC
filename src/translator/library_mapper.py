"""
Library mapper for converting Python imports to C++ includes
"""

from typing import List, Dict, Set


class LibraryMapper:
    """Maps Python library imports to equivalent C++ includes and libraries"""
    
    def __init__(self):
        # Mapping of Python modules to C++ includes
        self.module_mappings = {
            # Standard library mappings
            'math': ['#include <cmath>'],
            'random': ['#include <random>'],
            'time': ['#include <chrono>', '#include <thread>'],
            'datetime': ['#include <chrono>'],
            'os': ['#include <filesystem>', '#include <cstdlib>'],
            'sys': ['#include <iostream>', '#include <cstdlib>'],
            'json': ['// TODO: Use nlohmann/json library'],
            're': ['#include <regex>'],
            'string': ['#include <string>', '#include <algorithm>'],
            'collections': ['#include <map>', '#include <set>', '#include <queue>', '#include <stack>'],
            'itertools': ['#include <algorithm>', '#include <numeric>'],
            'functools': ['#include <functional>'],
            'operator': ['#include <functional>'],
            'copy': ['#include <algorithm>'],
            'io': ['#include <iostream>', '#include <fstream>', '#include <sstream>'],
            'pathlib': ['#include <filesystem>'],
            'typing': [],  # No direct C++ equivalent
            
            # Popular third-party libraries
            'numpy': ['// TODO: Use Eigen or similar library'],
            'pandas': ['// TODO: Use DataFrame library or custom implementation'],
            'matplotlib': ['// TODO: Use plotting library like matplotlib-cpp'],
            'requests': ['// TODO: Use libcurl or similar HTTP library'],
            'flask': ['// TODO: Use web framework like crow or pistache'],
            'django': ['// TODO: Use web framework like crow or pistache'],
            'tkinter': ['// TODO: Use GUI framework like Qt or GTK'],
            'pygame': ['// TODO: Use game development library like SDL or SFML'],
            'opencv': ['#include <opencv2/opencv.hpp>'],
            'pil': ['// TODO: Use image processing library'],
            'sqlite3': ['#include <sqlite3.h>'],
            'threading': ['#include <thread>', '#include <mutex>'],
            'multiprocessing': ['#include <thread>', '#include <future>'],
            'asyncio': ['// TODO: Use coroutines (C++20) or async library'],
        }
        
        # Mapping of specific functions from modules
        self.function_mappings = {
            'math': {
                'sqrt': 'std::sqrt',
                'pow': 'std::pow',
                'sin': 'std::sin',
                'cos': 'std::cos',
                'tan': 'std::tan',
                'log': 'std::log',
                'log10': 'std::log10',
                'exp': 'std::exp',
                'floor': 'std::floor',
                'ceil': 'std::ceil',
                'abs': 'std::abs',
                'pi': 'M_PI',
                'e': 'M_E',
            },
            'random': {
                'random': 'std::uniform_real_distribution<double>(0.0, 1.0)',
                'randint': 'std::uniform_int_distribution',
                'choice': '// TODO: Implement choice function',
                'shuffle': 'std::shuffle',
                'seed': '// TODO: Set random seed',
            },
            'os': {
                'getcwd': 'std::filesystem::current_path()',
                'chdir': 'std::filesystem::current_path',
                'listdir': 'std::filesystem::directory_iterator',
                'mkdir': 'std::filesystem::create_directory',
                'rmdir': 'std::filesystem::remove',
                'remove': 'std::filesystem::remove',
                'rename': 'std::filesystem::rename',
                'path.join': 'std::filesystem::path operator/',
                'path.exists': 'std::filesystem::exists',
                'path.isfile': 'std::filesystem::is_regular_file',
                'path.isdir': 'std::filesystem::is_directory',
            },
            'sys': {
                'argv': '// Use main(int argc, char* argv[])',
                'exit': 'std::exit',
                'stdout': 'std::cout',
                'stderr': 'std::cerr',
                'stdin': 'std::cin',
            },
            'time': {
                'time': 'std::chrono::system_clock::now()',
                'sleep': 'std::this_thread::sleep_for',
                'perf_counter': 'std::chrono::high_resolution_clock::now()',
            },
            'string': {
                'ascii_lowercase': '"abcdefghijklmnopqrstuvwxyz"',
                'ascii_uppercase': '"ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
                'digits': '"0123456789"',
            },
        }
        
        # Common C++ includes that are often needed
        self.common_includes = {
            '#include <iostream>',
            '#include <string>',
            '#include <vector>',
            '#include <memory>',
            '#include <algorithm>',
            '#include <functional>',
        }
    
    def map_imports(self, imports: List[str], from_imports: Dict[str, List[str]]) -> List[str]:
        """
        Map Python imports to C++ includes
        
        Args:
            imports: List of imported modules (from import statements)
            from_imports: Dict of module -> list of imported names (from from-import statements)
            
        Returns:
            List of C++ include statements
        """
        cpp_includes = set()
        
        # Handle regular imports
        for module in imports:
            if module in self.module_mappings:
                cpp_includes.update(self.module_mappings[module])
            else:
                cpp_includes.add(f"// TODO: Map Python module '{module}' to C++ equivalent")
        
        # Handle from-imports
        for module, names in from_imports.items():
            if module in self.module_mappings:
                cpp_includes.update(self.module_mappings[module])
            else:
                cpp_includes.add(f"// TODO: Map Python module '{module}' to C++ equivalent")
            
            # Add specific function mappings if available
            if module in self.function_mappings:
                for name in names:
                    if name in self.function_mappings[module]:
                        mapping = self.function_mappings[module][name]
                        if mapping.startswith('//'):
                            cpp_includes.add(mapping)
        
        return sorted(list(cpp_includes))
    
    def map_function_call(self, module: str, function: str) -> str:
        """
        Map a Python function call to C++ equivalent
        
        Args:
            module: Python module name
            function: Function name
            
        Returns:
            C++ equivalent function call
        """
        if module in self.function_mappings:
            if function in self.function_mappings[module]:
                return self.function_mappings[module][function]
        
        return f"{module}::{function}"  # Default namespace-like mapping
    
    def get_suggested_libraries(self, imports: List[str]) -> List[str]:
        """
        Get suggested C++ libraries based on Python imports
        
        Args:
            imports: List of Python imports
            
        Returns:
            List of suggested C++ libraries
        """
        suggestions = []
        
        library_suggestions = {
            'numpy': 'Eigen (for linear algebra), Armadillo, or Blaze',
            'pandas': 'Custom DataFrame implementation or Apache Arrow',
            'matplotlib': 'matplotlib-cpp, gnuplot-iostream, or custom plotting',
            'requests': 'libcurl, cpprest, or httplib',
            'flask': 'crow, pistache, or beast',
            'django': 'crow, pistache, or drogon',
            'tkinter': 'Qt, GTK+, or Dear ImGui',
            'pygame': 'SDL2, SFML, or Allegro',
            'opencv': 'OpenCV C++ API',
            'pillow': 'FreeImage, DevIL, or SOIL',
            'sqlite3': 'SQLite C API or SQLiteModernCpp',
            'beautifulsoup4': 'pugixml, tinyxml2, or html parser',
            'lxml': 'pugixml, libxml2, or tinyxml2',
            'pytest': 'Google Test, Catch2, or doctest',
        }
        
        for module in imports:
            if module in library_suggestions:
                suggestions.append(f"{module} -> {library_suggestions[module]}")
        
        return suggestions
    
    def get_build_system_suggestions(self, imports: List[str]) -> Dict[str, List[str]]:
        """
        Get build system configuration suggestions based on imports
        
        Args:
            imports: List of Python imports
            
        Returns:
            Dict with build system configurations
        """
        cmake_packages = []
        vcpkg_packages = []
        conan_packages = []
        
        package_mappings = {
            'opencv': {
                'cmake': 'find_package(OpenCV REQUIRED)',
                'vcpkg': 'opencv',
                'conan': 'opencv/4.5.5'
            },
            'sqlite3': {
                'cmake': 'find_package(SQLite3 REQUIRED)',
                'vcpkg': 'sqlite3',
                'conan': 'sqlite3/3.38.5'
            },
            'requests': {
                'cmake': 'find_package(CURL REQUIRED)',
                'vcpkg': 'curl',
                'conan': 'libcurl/7.80.0'
            },
        }
        
        for module in imports:
            if module in package_mappings:
                mapping = package_mappings[module]
                cmake_packages.append(mapping['cmake'])
                vcpkg_packages.append(mapping['vcpkg'])
                conan_packages.append(mapping['conan'])
        
        return {
            'cmake': cmake_packages,
            'vcpkg': vcpkg_packages,
            'conan': conan_packages
        }
