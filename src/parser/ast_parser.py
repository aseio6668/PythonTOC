"""
AST Parser for Python code analysis
"""

import ast
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    args: List[str]
    return_type: Optional[str] = None
    is_method: bool = False
    is_static: bool = False
    is_class_method: bool = False
    decorators: List[str] = None
    
    def __post_init__(self):
        if self.decorators is None:
            self.decorators = []


@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    bases: List[str]
    methods: List[FunctionInfo]
    attributes: List[str]
    decorators: List[str] = None
    
    def __post_init__(self):
        if self.decorators is None:
            self.decorators = []


@dataclass
class VariableInfo:
    """Information about a variable"""
    name: str
    type_hint: Optional[str] = None
    is_global: bool = False
    is_parameter: bool = False


class PythonASTParser:
    """Parser for Python AST to extract code structure information"""
    
    def __init__(self):
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        self.variables: List[VariableInfo] = []
        self.imports: List[str] = []
        self.from_imports: Dict[str, List[str]] = {}
        
    def parse_file(self, file_path: str) -> ast.AST:
        """Parse a Python file and return the AST"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse_code(content)
    
    def parse_code(self, code: str) -> ast.AST:
        """Parse Python code string and return the AST"""
        tree = ast.parse(code)
        self._analyze_ast(tree)
        return tree
    
    def _analyze_ast(self, tree: ast.AST):
        """Analyze the AST and extract information"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._analyze_function(node)
            elif isinstance(node, ast.ClassDef):
                self._analyze_class(node)
            elif isinstance(node, ast.Import):
                self._analyze_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._analyze_from_import(node)
            elif isinstance(node, ast.Assign):
                self._analyze_assignment(node)
    
    def _analyze_function(self, node: ast.FunctionDef, is_method: bool = False):
        """Analyze a function definition"""
        args = [arg.arg for arg in node.args.args]
        
        # Check for decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))
        
        # Determine function type
        is_static = 'staticmethod' in decorators
        is_class_method = 'classmethod' in decorators
        
        # Get return type annotation if available
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        func_info = FunctionInfo(
            name=node.name,
            args=args,
            return_type=return_type,
            is_method=is_method,
            is_static=is_static,
            is_class_method=is_class_method,
            decorators=decorators
        )
        
        self.functions.append(func_info)
    
    def _analyze_class(self, node: ast.ClassDef):
        """Analyze a class definition"""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(ast.unparse(base))
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))
        
        # Analyze methods and attributes
        methods = []
        attributes = set()
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Create a temporary parser for method analysis
                method_info = FunctionInfo(
                    name=item.name,
                    args=[arg.arg for arg in item.args.args],
                    is_method=True
                )
                
                # Check for decorators
                method_decorators = []
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name):
                        method_decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        method_decorators.append(ast.unparse(decorator))
                
                method_info.decorators = method_decorators
                method_info.is_static = 'staticmethod' in method_decorators
                method_info.is_class_method = 'classmethod' in method_decorators
                
                if item.returns:
                    method_info.return_type = ast.unparse(item.returns)
                
                methods.append(method_info)
            
            elif isinstance(item, ast.Assign):
                # Look for class attributes
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.add(target.id)
        
        class_info = ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            attributes=list(attributes),
            decorators=decorators
        )
        
        self.classes.append(class_info)
    
    def _analyze_import(self, node: ast.Import):
        """Analyze import statement"""
        for alias in node.names:
            self.imports.append(alias.name)
    
    def _analyze_from_import(self, node: ast.ImportFrom):
        """Analyze from-import statement"""
        if node.module:
            imported_names = [alias.name for alias in node.names]
            self.from_imports[node.module] = imported_names
    
    def _analyze_assignment(self, node: ast.Assign):
        """Analyze variable assignment"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_info = VariableInfo(
                    name=target.id,
                    is_global=True  # Simplified - would need scope analysis
                )
                self.variables.append(var_info)
    
    def get_all_functions(self) -> List[FunctionInfo]:
        """Get all functions including class methods"""
        all_functions = self.functions.copy()
        for class_info in self.classes:
            all_functions.extend(class_info.methods)
        return all_functions
    
    def get_function_by_name(self, name: str) -> Optional[FunctionInfo]:
        """Get function information by name"""
        for func in self.get_all_functions():
            if func.name == name:
                return func
        return None
    
    def get_class_by_name(self, name: str) -> Optional[ClassInfo]:
        """Get class information by name"""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None
