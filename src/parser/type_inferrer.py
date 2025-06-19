"""
Type inference engine for Python to C++ translation
"""

import ast
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass
from enum import Enum


class CppType(Enum):
    """C++ type mappings"""
    INT = "int"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    BOOL = "bool"
    STRING = "std::string"
    CHAR = "char"
    VOID = "void"
    AUTO = "auto"
    VECTOR = "std::vector"
    MAP = "std::map"
    SET = "std::set"
    UNORDERED_MAP = "std::unordered_map"
    UNORDERED_SET = "std::unordered_set"
    SHARED_PTR = "std::shared_ptr"
    UNIQUE_PTR = "std::unique_ptr"


@dataclass
class TypeInfo:
    """Information about a type"""
    cpp_type: CppType
    template_args: List['TypeInfo'] = None
    is_pointer: bool = False
    is_reference: bool = False
    is_const: bool = False
    custom_type: Optional[str] = None
    
    def __post_init__(self):
        if self.template_args is None:
            self.template_args = []
    
    def to_cpp_string(self) -> str:
        """Convert type info to C++ type string"""
        if self.custom_type:
            result = self.custom_type
        else:
            result = self.cpp_type.value
        
        # Handle template arguments
        if self.template_args:
            template_str = ", ".join(arg.to_cpp_string() for arg in self.template_args)
            result += f"<{template_str}>"
        
        # Handle const
        if self.is_const:
            result = f"const {result}"
        
        # Handle pointers and references
        if self.is_pointer:
            result += "*"
        elif self.is_reference:
            result += "&"
        
        return result


class TypeInferrer:
    """Infers types for Python variables and expressions"""
    
    def __init__(self):
        self.variable_types: Dict[str, TypeInfo] = {}
        self.function_return_types: Dict[str, TypeInfo] = {}
        self.builtin_type_mapping = {
            'int': TypeInfo(CppType.INT),
            'float': TypeInfo(CppType.DOUBLE),
            'str': TypeInfo(CppType.STRING),
            'bool': TypeInfo(CppType.BOOL),
            'list': TypeInfo(CppType.VECTOR),
            'dict': TypeInfo(CppType.MAP),
            'set': TypeInfo(CppType.SET),
            'tuple': TypeInfo(CppType.VECTOR),  # Simplified mapping
        }
    
    def infer_from_annotation(self, annotation: ast.expr) -> TypeInfo:
        """Infer type from Python type annotation"""
        if isinstance(annotation, ast.Name):
            type_name = annotation.id
            if type_name in self.builtin_type_mapping:
                return self.builtin_type_mapping[type_name]
            else:
                # Custom class type
                return TypeInfo(CppType.AUTO, custom_type=type_name)
        
        elif isinstance(annotation, ast.Subscript):
            # Handle generic types like List[int], Dict[str, int]
            if isinstance(annotation.value, ast.Name):
                container_type = annotation.value.id
                
                if container_type == 'List':
                    element_type = self.infer_from_annotation(annotation.slice)
                    return TypeInfo(CppType.VECTOR, template_args=[element_type])
                
                elif container_type == 'Dict':
                    if isinstance(annotation.slice, ast.Tuple):
                        key_type = self.infer_from_annotation(annotation.slice.elts[0])
                        value_type = self.infer_from_annotation(annotation.slice.elts[1])
                        return TypeInfo(CppType.MAP, template_args=[key_type, value_type])
                
                elif container_type == 'Set':
                    element_type = self.infer_from_annotation(annotation.slice)
                    return TypeInfo(CppType.SET, template_args=[element_type])
                
                elif container_type == 'Optional':
                    inner_type = self.infer_from_annotation(annotation.slice)
                    return TypeInfo(CppType.SHARED_PTR, template_args=[inner_type])
        
        # Default fallback
        return TypeInfo(CppType.AUTO)
    
    def infer_from_literal(self, node: ast.expr) -> TypeInfo:
        """Infer type from literal values"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return TypeInfo(CppType.INT)
            elif isinstance(node.value, float):
                return TypeInfo(CppType.DOUBLE)
            elif isinstance(node.value, str):
                return TypeInfo(CppType.STRING)
            elif isinstance(node.value, bool):
                return TypeInfo(CppType.BOOL)
        
        elif isinstance(node, ast.List):
            if node.elts:
                # Infer element type from first element
                element_type = self.infer_from_literal(node.elts[0])
                return TypeInfo(CppType.VECTOR, template_args=[element_type])
            else:
                # Empty list, use auto
                return TypeInfo(CppType.VECTOR, template_args=[TypeInfo(CppType.AUTO)])
        
        elif isinstance(node, ast.Dict):
            if node.keys and node.values:
                key_type = self.infer_from_literal(node.keys[0])
                value_type = self.infer_from_literal(node.values[0])
                return TypeInfo(CppType.MAP, template_args=[key_type, value_type])
            else:
                return TypeInfo(CppType.MAP, template_args=[TypeInfo(CppType.AUTO), TypeInfo(CppType.AUTO)])
        
        elif isinstance(node, ast.Set):
            if node.elts:
                element_type = self.infer_from_literal(node.elts[0])
                return TypeInfo(CppType.SET, template_args=[element_type])
            else:
                return TypeInfo(CppType.SET, template_args=[TypeInfo(CppType.AUTO)])
        
        # Default fallback
        return TypeInfo(CppType.AUTO)
    
    def infer_from_call(self, node: ast.Call) -> TypeInfo:
        """Infer type from function call"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Built-in constructors
            if func_name in self.builtin_type_mapping:
                return self.builtin_type_mapping[func_name]
            
            # Check if we know the return type
            if func_name in self.function_return_types:
                return self.function_return_types[func_name]
        
        # Default fallback
        return TypeInfo(CppType.AUTO)
    
    def infer_from_binop(self, node: ast.BinOp) -> TypeInfo:
        """Infer type from binary operation"""
        left_type = self.infer_type(node.left)
        right_type = self.infer_type(node.right)
        
        # String concatenation
        if isinstance(node.op, ast.Add):
            if (left_type.cpp_type == CppType.STRING or 
                right_type.cpp_type == CppType.STRING):
                return TypeInfo(CppType.STRING)
        
        # Numeric operations
        if (left_type.cpp_type in [CppType.INT, CppType.FLOAT, CppType.DOUBLE] and
            right_type.cpp_type in [CppType.INT, CppType.FLOAT, CppType.DOUBLE]):
            
            # Division always returns float
            if isinstance(node.op, ast.Div):
                return TypeInfo(CppType.DOUBLE)
            
            # If either operand is double/float, result is double
            if (left_type.cpp_type in [CppType.FLOAT, CppType.DOUBLE] or
                right_type.cpp_type in [CppType.FLOAT, CppType.DOUBLE]):
                return TypeInfo(CppType.DOUBLE)
            
            # Both integers
            return TypeInfo(CppType.INT)
        
        # Comparison operations
        if isinstance(node.op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return TypeInfo(CppType.BOOL)
        
        # Default fallback
        return TypeInfo(CppType.AUTO)
    
    def infer_type(self, node: ast.expr) -> TypeInfo:
        """Main type inference method"""
        if isinstance(node, ast.Name):
            # Check if we already know the type
            if node.id in self.variable_types:
                return self.variable_types[node.id]
            else:
                return TypeInfo(CppType.AUTO)
        
        elif isinstance(node, ast.Constant):
            return self.infer_from_literal(node)
        
        elif isinstance(node, (ast.List, ast.Dict, ast.Set)):
            return self.infer_from_literal(node)
        
        elif isinstance(node, ast.Call):
            return self.infer_from_call(node)
        
        elif isinstance(node, ast.BinOp):
            return self.infer_from_binop(node)
        
        elif isinstance(node, ast.UnaryOp):
            operand_type = self.infer_type(node.operand)
            if isinstance(node.op, ast.Not):
                return TypeInfo(CppType.BOOL)
            else:
                return operand_type
        
        elif isinstance(node, ast.Compare):
            return TypeInfo(CppType.BOOL)
        
        elif isinstance(node, ast.Attribute):
            # For method calls or attribute access
            return TypeInfo(CppType.AUTO)
        
        # Default fallback
        return TypeInfo(CppType.AUTO)
    
    def register_variable_type(self, name: str, type_info: TypeInfo):
        """Register a variable's type"""
        self.variable_types[name] = type_info
    
    def register_function_return_type(self, name: str, type_info: TypeInfo):
        """Register a function's return type"""
        self.function_return_types[name] = type_info
    
    def get_variable_type(self, name: str) -> Optional[TypeInfo]:
        """Get a variable's type"""
        return self.variable_types.get(name)
    
    def get_function_return_type(self, name: str) -> Optional[TypeInfo]:
        """Get a function's return type"""
        return self.function_return_types.get(name)
