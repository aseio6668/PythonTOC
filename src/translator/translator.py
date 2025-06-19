"""
Main translator class that coordinates the translation process
"""

import ast
from typing import Optional, List, Dict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.ast_parser import PythonASTParser
from parser.type_inferrer import TypeInferrer
from translator.cpp_generator import CppCodeGenerator
from translator.library_mapper import LibraryMapper
from modules.dependency_manager import ModuleDependencyManager, ModuleInfo


class PythonToCppTranslator:
    """Main translator class for converting Python code to C++"""
    
    def __init__(self, 
                 include_headers: bool = True,
                 namespace: Optional[str] = None,
                 verbose: bool = False,
                 manage_dependencies: bool = False,
                 output_dir: Optional[Path] = None):
        """
        Initialize the translator
        
        Args:
            include_headers: Whether to include standard C++ headers
            namespace: Optional namespace to wrap the generated code
            verbose: Enable verbose output
            manage_dependencies: Enable automatic dependency management
            output_dir: Output directory for generated modules
        """
        self.include_headers = include_headers
        self.namespace = namespace
        self.verbose = verbose
        self.manage_dependencies = manage_dependencies
        
        # Initialize components
        self.parser = PythonASTParser()
        self.type_inferrer = TypeInferrer()
        self.cpp_generator = CppCodeGenerator(
            include_headers=include_headers,
            namespace=namespace
        )
        self.library_mapper = LibraryMapper()
        
        # Initialize module dependency manager if enabled
        if manage_dependencies:
            self.dependency_manager = ModuleDependencyManager(output_dir)
        else:
            self.dependency_manager = None
    
    def translate_file(self, file_path: str) -> str:
        """
        Translate a Python file to C++
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Generated C++ code as string
        """
        if self.verbose:
            print(f"Parsing file: {file_path}")
        
        # Parse the Python file
        ast_tree = self.parser.parse_file(file_path)
        
        # Analyze dependencies if module management is enabled
        if self.dependency_manager:
            dependencies = self.dependency_manager.analyze_dependencies(Path(file_path))
            if self.verbose:
                print(f"Found {len(dependencies)} module dependencies")
                for dep in dependencies:
                    status = "✓" if dep.is_installed else "✗"
                    print(f"  {status} {dep.name} ({'builtin' if dep.is_builtin else 'external'})")
        
        if self.verbose:
            print(f"Found {len(self.parser.functions)} functions")
            print(f"Found {len(self.parser.classes)} classes")
            print(f"Found {len(self.parser.imports)} imports")
        
        # Generate C++ code
        return self.translate_ast(ast_tree)
    
    def translate_code(self, python_code: str) -> str:
        """
        Translate Python code string to C++
        
        Args:
            python_code: Python source code as string
            
        Returns:
            Generated C++ code as string
        """
        if self.verbose:
            print("Parsing Python code...")
        
        # Parse the Python code
        ast_tree = self.parser.parse_code(python_code)
        
        if self.verbose:
            print(f"Found {len(self.parser.functions)} functions")
            print(f"Found {len(self.parser.classes)} classes")
            print(f"Found {len(self.parser.imports)} imports")
        
        # Generate C++ code
        return self.translate_ast(ast_tree)
    
    def translate_ast(self, ast_tree: ast.AST) -> str:
        """
        Translate an AST to C++
        
        Args:
            ast_tree: Python AST
            
        Returns:
            Generated C++ code as string
        """
        # Map Python imports to C++ includes
        cpp_includes = self.library_mapper.map_imports(
            self.parser.imports,
            self.parser.from_imports
        )
        
        # Set up the generator with parsed information
        self.cpp_generator.set_imports(cpp_includes)
        self.cpp_generator.set_functions(self.parser.functions)
        self.cpp_generator.set_classes(self.parser.classes)
        
        # Perform type inference
        self._perform_type_inference(ast_tree)
        
        # Generate C++ code
        return self.cpp_generator.generate(ast_tree)
    
    def _perform_type_inference(self, ast_tree: ast.AST):
        """Perform type inference on the AST"""
        if self.verbose:
            print("Performing type inference...")
        
        # Walk through the AST and infer types
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                self._infer_function_types(node)
            elif isinstance(node, ast.ClassDef):
                self._infer_class_types(node)
            elif isinstance(node, ast.Assign):
                self._infer_assignment_types(node)
    
    def _infer_function_types(self, node: ast.FunctionDef):
        """Infer types for function parameters and return values"""
        # Check for type annotations
        for arg in node.args.args:
            if arg.annotation:
                type_info = self.type_inferrer.infer_from_annotation(arg.annotation)
                self.type_inferrer.register_variable_type(arg.arg, type_info)
        
        # Infer return type
        if node.returns:
            return_type = self.type_inferrer.infer_from_annotation(node.returns)
            self.type_inferrer.register_function_return_type(node.name, return_type)
        else:
            # Try to infer from return statements
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value:
                    return_type = self.type_inferrer.infer_type(child.value)
                    self.type_inferrer.register_function_return_type(node.name, return_type)
                    break
    
    def _infer_class_types(self, node: ast.ClassDef):
        """Infer types for class members"""
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._infer_function_types(item)
            elif isinstance(item, ast.Assign):
                self._infer_assignment_types(item)
    
    def _infer_assignment_types(self, node: ast.Assign):
        """Infer types for variable assignments"""
        if node.value:
            value_type = self.type_inferrer.infer_type(node.value)
            
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.type_inferrer.register_variable_type(target.id, value_type)
    
    def get_parser_info(self) -> Dict:
        """Get information from the parser for debugging"""
        return {
            'functions': self.parser.functions,
            'classes': self.parser.classes,
            'variables': self.parser.variables,
            'imports': self.parser.imports,
            'from_imports': self.parser.from_imports
        }
    
    def get_type_info(self) -> Dict:
        """Get type inference information for debugging"""
        return {
            'variable_types': {name: info.to_cpp_string() 
                             for name, info in self.type_inferrer.variable_types.items()},
            'function_return_types': {name: info.to_cpp_string() 
                                    for name, info in self.type_inferrer.function_return_types.items()}
        }
    
    def get_dependency_info(self) -> Optional[Dict]:
        """Get dependency analysis information"""
        if not self.dependency_manager:
            return None
        
        return {
            'dependencies': [
                {
                    'name': dep.name,
                    'is_builtin': dep.is_builtin,
                    'is_installed': dep.is_installed,
                    'is_pure_python': dep.is_pure_python,
                    'cpp_equivalent': dep.cpp_equivalent
                }
                for dep in self.dependency_manager.module_cache.values()
            ]
        }
    
    def generate_dependency_report(self, file_path: str) -> Optional[str]:
        """Generate a comprehensive dependency report"""
        if not self.dependency_manager:
            return None
        
        dependencies = self.dependency_manager.analyze_dependencies(Path(file_path))
        suggestions = self.dependency_manager.suggest_cpp_alternatives(dependencies)
        
        # Generate base report
        report = self.dependency_manager.generate_dependency_report(dependencies, suggestions)
        
        # Add ML-specific guidance if applicable
        ml_guide = self.dependency_manager.generate_ml_conversion_guide(dependencies)
        if ml_guide:
            report += "\n\n" + ml_guide
        
        return report
    
    def create_module_project(self, file_path: str, project_name: str = None) -> Optional[Path]:
        """Create a complete C++ project with dependency management"""
        if not self.dependency_manager:
            return None
        
        if not project_name:
            project_name = Path(file_path).stem
        
        dependencies = self.dependency_manager.analyze_dependencies(Path(file_path))
        return self.dependency_manager.create_module_project(dependencies, project_name)
    
    def analyze_and_convert_dependencies(self, python_files: List[str], 
                                        convert_pure_python: bool = False,
                                        download_sources: bool = False) -> Dict:
        """
        Analyze dependencies and optionally convert pure Python modules
        
        Args:
            python_files: List of Python files to analyze
            convert_pure_python: Whether to convert pure Python dependencies
            download_sources: Whether to download source code for conversion
            
        Returns:
            Dictionary with dependency analysis results
        """
        if self.verbose:
            print("Analyzing module dependencies...")
        
        # Analyze dependencies
        dependencies = self.module_manager.analyze_dependencies(python_files)
        
        results = {
            'dependencies': dependencies,
            'converted_modules': {},
            'cpp_mappings': {},
            'conversion_failures': []
        }
        
        # Convert pure Python modules if requested
        if convert_pure_python:
            if self.verbose:
                print("Converting pure Python dependencies...")
            
            for module_name, dep_info in dependencies.items():
                if dep_info.can_convert:
                    try:
                        if download_sources:
                            converted_path = self.module_manager.download_and_convert_module(module_name)
                            if converted_path:
                                results['converted_modules'][module_name] = converted_path
                                if self.verbose:
                                    print(f"  Converted {module_name} -> {converted_path}")
                            else:
                                results['conversion_failures'].append(module_name)
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"  Failed to convert {module_name}: {e}")
                        results['conversion_failures'].append(module_name)
                
                elif dep_info.cpp_equivalent:
                    results['cpp_mappings'][module_name] = dep_info.cpp_equivalent
        
        return results
