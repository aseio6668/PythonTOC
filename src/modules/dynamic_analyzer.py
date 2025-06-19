"""
Dynamic Module Analysis and Translation System
Real-time analysis and C++ conversion of unknown Python modules
"""

import ast
import inspect
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import json
import requests
from urllib.parse import urljoin

@dataclass
class ModuleAnalysis:
    """Analysis result for a Python module"""
    name: str
    version: Optional[str] = None
    is_pure_python: bool = True
    has_c_extensions: bool = False
    functions: List[Dict] = None
    classes: List[Dict] = None
    constants: List[Dict] = None
    dependencies: List[str] = None
    complexity_score: int = 0  # 1-10 scale
    translation_feasibility: str = "unknown"  # "easy", "medium", "hard", "impossible"
    api_surface: Dict = None
    
    def __post_init__(self):
        if self.functions is None:
            self.functions = []
        if self.classes is None:
            self.classes = []
        if self.constants is None:
            self.constants = []
        if self.dependencies is None:
            self.dependencies = []
        if self.api_surface is None:
            self.api_surface = {}


class DynamicModuleAnalyzer:
    """
    Analyzes Python modules in real-time and generates C++ translations
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("dynamic_modules")
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_cache = {}
        
        # Pre-defined patterns for common Python constructs
        self.cpp_type_mappings = {
            'int': 'int',
            'float': 'double',
            'str': 'std::string',
            'bool': 'bool',
            'list': 'std::vector',
            'dict': 'std::unordered_map',
            'set': 'std::set',
            'tuple': 'std::tuple',
            'None': 'nullptr'
        }
    
    def analyze_module_live(self, module_name: str) -> ModuleAnalysis:
        """
        Analyze a module that's currently installed/importable
        """
        if module_name in self.analysis_cache:
            return self.analysis_cache[module_name]
        
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            analysis = ModuleAnalysis(name=module_name)
            
            # Get module info
            analysis.version = getattr(module, '__version__', None)
            
            # Analyze module structure
            self._analyze_module_structure(module, analysis)
            
            # Determine if it's pure Python
            analysis.is_pure_python = self._is_pure_python(module)
            analysis.has_c_extensions = not analysis.is_pure_python
            
            # Calculate complexity and feasibility
            analysis.complexity_score = self._calculate_complexity(analysis)
            analysis.translation_feasibility = self._assess_feasibility(analysis)
            
            self.analysis_cache[module_name] = analysis
            return analysis
            
        except ImportError:
            # Module not installed, try to analyze from source
            return self._analyze_from_pypi(module_name)
    
    def _analyze_module_structure(self, module: Any, analysis: ModuleAnalysis):
        """Analyze the structure of an imported module"""
        
        # Get all public attributes
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            try:
                attr = getattr(module, name)
                
                if inspect.isfunction(attr):
                    func_info = self._analyze_function(attr)
                    analysis.functions.append(func_info)
                    
                elif inspect.isclass(attr):
                    class_info = self._analyze_class(attr)
                    analysis.classes.append(class_info)
                    
                elif not inspect.ismodule(attr):
                    # Constant or variable
                    const_info = {
                        'name': name,
                        'type': type(attr).__name__,
                        'value': str(attr) if len(str(attr)) < 100 else f"{str(attr)[:100]}..."
                    }
                    analysis.constants.append(const_info)
                    
            except Exception as e:
                # Skip attributes that can't be analyzed
                continue
    
    def _analyze_function(self, func: callable) -> Dict:
        """Analyze a function and extract its signature and metadata"""
        try:
            sig = inspect.signature(func)
            
            return {
                'name': func.__name__,
                'parameters': [
                    {
                        'name': param.name,
                        'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'auto',
                        'default': str(param.default) if param.default != inspect.Parameter.empty else None
                    }
                    for param in sig.parameters.values()
                ],
                'return_type': str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else 'auto',
                'doc': inspect.getdoc(func) or '',
                'source_available': self._has_source(func)
            }
        except Exception:
            return {
                'name': func.__name__,
                'parameters': [],
                'return_type': 'auto',
                'doc': inspect.getdoc(func) or '',
                'source_available': False
            }
    
    def _analyze_class(self, cls: type) -> Dict:
        """Analyze a class and its methods"""
        try:
            methods = []
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if not name.startswith('_') or name in ['__init__', '__str__', '__repr__']:
                    methods.append(self._analyze_function(method))
            
            return {
                'name': cls.__name__,
                'base_classes': [base.__name__ for base in cls.__bases__],
                'methods': methods,
                'doc': inspect.getdoc(cls) or '',
                'source_available': self._has_source(cls)
            }
        except Exception:
            return {
                'name': cls.__name__,
                'base_classes': [],
                'methods': [],
                'doc': '',
                'source_available': False
            }
    
    def _has_source(self, obj) -> bool:
        """Check if source code is available for an object"""
        try:
            inspect.getsource(obj)
            return True
        except (OSError, TypeError):
            return False
    
    def _is_pure_python(self, module: Any) -> bool:
        """Determine if a module is pure Python"""
        try:
            # Check if module file is a .py file
            if hasattr(module, '__file__') and module.__file__:
                return module.__file__.endswith('.py')
            
            # Check for C extensions
            for name in dir(module):
                attr = getattr(module, name, None)
                if hasattr(attr, '__module__') and attr.__module__ == 'builtins':
                    continue
                if inspect.isbuiltin(attr) or inspect.ismethoddescriptor(attr):
                    return False
            
            return True
        except Exception:
            return False
    
    def _calculate_complexity(self, analysis: ModuleAnalysis) -> int:
        """Calculate complexity score (1-10)"""
        score = 1
        
        # Add complexity for number of functions/classes
        score += min(len(analysis.functions) // 10, 3)
        score += min(len(analysis.classes) // 5, 3)
        
        # Add complexity for C extensions
        if analysis.has_c_extensions:
            score += 3
        
        # Add complexity for complex function signatures
        for func in analysis.functions:
            if len(func['parameters']) > 5:
                score += 1
        
        return min(score, 10)
    
    def _assess_feasibility(self, analysis: ModuleAnalysis) -> str:
        """Assess translation feasibility"""
        if analysis.has_c_extensions:
            return "hard"
        elif analysis.complexity_score <= 3:
            return "easy"
        elif analysis.complexity_score <= 6:
            return "medium"
        else:
            return "hard"
    
    def _analyze_from_pypi(self, module_name: str) -> ModuleAnalysis:
        """Analyze module by downloading from PyPI"""
        analysis = ModuleAnalysis(name=module_name)
        
        try:
            # Get module info from PyPI API
            response = requests.get(f"https://pypi.org/pypi/{module_name}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                analysis.version = data['info']['version']
                
                # Check for C extensions
                for file_info in data['urls']:
                    if file_info['packagetype'] == 'sdist':
                        # Download and analyze source
                        return self._analyze_source_archive(module_name, file_info['url'], analysis)
                
        except Exception as e:
            print(f"Error analyzing {module_name} from PyPI: {e}")
        
        analysis.translation_feasibility = "unknown"
        return analysis
    
    def _analyze_source_archive(self, module_name: str, url: str, analysis: ModuleAnalysis) -> ModuleAnalysis:
        """Download and analyze source archive"""
        # This would download and analyze the source code
        # For now, return basic analysis
        analysis.translation_feasibility = "medium"
        return analysis
    
    def generate_cpp_translation(self, analysis: ModuleAnalysis) -> str:
        """Generate C++ code for a module based on analysis"""
        
        if analysis.translation_feasibility == "impossible":
            return f"// Module {analysis.name} cannot be automatically translated"
        
        cpp_code = []
        cpp_code.append(f"// Auto-generated C++ translation of Python module: {analysis.name}")
        cpp_code.append(f"// Version: {analysis.version or 'unknown'}")
        cpp_code.append(f"// Translation feasibility: {analysis.translation_feasibility}")
        cpp_code.append(f"// Complexity score: {analysis.complexity_score}/10")
        cpp_code.append("")
        
        # Standard includes
        cpp_code.extend([
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "#include <unordered_map>",
            "#include <memory>",
            "#include <functional>",
            ""
        ])
        
        # Namespace
        namespace = analysis.name.replace('.', '_').replace('-', '_')
        cpp_code.append(f"namespace {namespace} {{")
        cpp_code.append("")
        
        # Generate constants
        for const in analysis.constants:
            cpp_type = self._map_python_type_to_cpp(const['type'])
            cpp_code.append(f"const {cpp_type} {const['name']} = /* TODO: Initialize {const['name']} */;")
        
        if analysis.constants:
            cpp_code.append("")
        
        # Generate function declarations
        for func in analysis.functions:
            cpp_func = self._generate_cpp_function(func)
            cpp_code.append(cpp_func)
        
        # Generate class declarations
        for cls in analysis.classes:
            cpp_class = self._generate_cpp_class(cls)
            cpp_code.extend(cpp_class)
        
        cpp_code.append(f"}} // namespace {namespace}")
        
        return '\n'.join(cpp_code)
    
    def _map_python_type_to_cpp(self, py_type: str) -> str:
        """Map Python type to C++ type"""
        return self.cpp_type_mappings.get(py_type, 'auto')
    
    def _generate_cpp_function(self, func_info: Dict) -> str:
        """Generate C++ function declaration"""
        params = []
        for param in func_info['parameters']:
            cpp_type = self._map_python_type_to_cpp(param['type'])
            param_str = f"{cpp_type} {param['name']}"
            if param['default']:
                param_str += f" = /* default: {param['default']} */"
            params.append(param_str)
        
        return_type = self._map_python_type_to_cpp(func_info['return_type'])
        param_str = ', '.join(params)
        
        return f"{return_type} {func_info['name']}({param_str});"
    
    def _generate_cpp_class(self, class_info: Dict) -> List[str]:
        """Generate C++ class declaration"""
        lines = []
        
        # Class declaration
        if class_info['base_classes']:
            inheritance = ' : public ' + ', public '.join(class_info['base_classes'])
        else:
            inheritance = ''
        
        lines.append(f"class {class_info['name']}{inheritance} {{")
        lines.append("public:")
        
        # Generate method declarations
        for method in class_info['methods']:
            method_line = "    " + self._generate_cpp_function(method)
            lines.append(method_line)
        
        lines.append("};")
        lines.append("")
        
        return lines
    
    def create_smart_module_mapping(self, module_name: str) -> Dict:
        """Create intelligent mapping for unknown modules"""
        analysis = self.analyze_module_live(module_name)
        
        # Generate C++ translation
        cpp_code = self.generate_cpp_translation(analysis)
        
        # Create mapping entry
        mapping = {
            'name': f'Auto-generated {module_name}',
            'description': f'Automatically generated C++ equivalent for {module_name}',
            'analysis': {
                'complexity': analysis.complexity_score,
                'feasibility': analysis.translation_feasibility,
                'pure_python': analysis.is_pure_python,
                'function_count': len(analysis.functions),
                'class_count': len(analysis.classes)
            },
            'generated_code': cpp_code,
            'cmake_find': f'# find_package({module_name}) # Not available',
            'notes': [
                f'This is an automatically generated translation',
                f'Complexity: {analysis.complexity_score}/10',
                f'Feasibility: {analysis.translation_feasibility}',
                'Manual review and testing recommended'
            ]
        }
        
        if analysis.translation_feasibility == "easy":
            mapping['notes'].append('Good candidate for automatic translation')
        elif analysis.translation_feasibility == "hard":
            mapping['notes'].append('Consider using Python C API or keeping as Python module')
        
        return mapping


def demonstrate_dynamic_analysis():
    """Demonstrate the dynamic analysis system"""
    analyzer = DynamicModuleAnalyzer()
    
    # Example: analyze a custom module
    test_modules = ['json', 'datetime', 'collections']
    
    for module_name in test_modules:
        print(f"\\n=== Analyzing {module_name} ===")
        
        analysis = analyzer.analyze_module_live(module_name)
        print(f"Pure Python: {analysis.is_pure_python}")
        print(f"Functions: {len(analysis.functions)}")
        print(f"Classes: {len(analysis.classes)}")
        print(f"Complexity: {analysis.complexity_score}/10")
        print(f"Feasibility: {analysis.translation_feasibility}")
        
        # Generate mapping
        mapping = analyzer.create_smart_module_mapping(module_name)
        print(f"Generated mapping with {len(mapping['notes'])} notes")


if __name__ == "__main__":
    demonstrate_dynamic_analysis()
