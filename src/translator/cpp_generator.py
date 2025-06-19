"""
C++ code generator that converts Python AST to C++ code
"""

import ast
from typing import List, Dict, Optional, Set
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.ast_parser import FunctionInfo, ClassInfo
from parser.type_inferrer import TypeInfo, CppType
from utils.templates import CppTemplates


class CppCodeGenerator:
    """Generates C++ code from Python AST and parsed information"""
    
    def __init__(self, 
                 include_headers: bool = True,
                 namespace: Optional[str] = None,
                 indent: str = "    "):
        """
        Initialize the C++ code generator
        
        Args:
            include_headers: Whether to include standard headers
            namespace: Optional namespace to wrap code
            indent: Indentation string
        """
        self.include_headers = include_headers
        self.namespace = namespace
        self.indent = indent
        
        # Code components
        self.includes: Set[str] = set()
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        
        # Templates
        self.templates = CppTemplates()
        
        # Current indentation level
        self.indent_level = 0
        
        # Required includes based on used types
        self.required_includes = {
            CppType.STRING: "#include <string>",
            CppType.VECTOR: "#include <vector>",
            CppType.MAP: "#include <map>",
            CppType.SET: "#include <set>",
            CppType.UNORDERED_MAP: "#include <unordered_map>",
            CppType.UNORDERED_SET: "#include <unordered_set>",
            CppType.SHARED_PTR: "#include <memory>",
            CppType.UNIQUE_PTR: "#include <memory>",
        }
    
    def set_imports(self, cpp_includes: List[str]):
        """Set the C++ includes needed"""
        self.includes.update(cpp_includes)
    
    def set_functions(self, functions: List[FunctionInfo]):
        """Set the functions to generate"""
        self.functions = functions
    
    def set_classes(self, classes: List[ClassInfo]):
        """Set the classes to generate"""
        self.classes = classes
    
    def generate(self, ast_tree: ast.AST) -> str:
        """Generate complete C++ code"""
        code_parts = []
        
        # Add includes
        if self.include_headers:
            code_parts.append(self._generate_includes())
        
        # Add namespace opening
        if self.namespace:
            code_parts.append(f"namespace {self.namespace} {{")
            self.indent_level += 1
        
        # Generate forward declarations
        forward_decls = self._generate_forward_declarations()
        if forward_decls:
            code_parts.append(forward_decls)
        
        # Generate classes
        for class_info in self.classes:
            code_parts.append(self._generate_class(class_info))
        
        # Generate standalone functions
        standalone_functions = [f for f in self.functions if not any(f in c.methods for c in self.classes)]
        for func_info in standalone_functions:
            code_parts.append(self._generate_function_declaration(func_info))
        
        # Generate function implementations
        for func_info in standalone_functions:
            func_node = self._find_function_node(ast_tree, func_info.name)
            if func_node:
                code_parts.append(self._generate_function_implementation(func_node, func_info))
        
        # Add namespace closing
        if self.namespace:
            self.indent_level -= 1
            code_parts.append("}")
        
        return "\n\n".join(filter(None, code_parts))
    
    def _generate_includes(self) -> str:
        """Generate include statements"""
        standard_includes = [
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "#include <memory>"
        ]
        
        all_includes = standard_includes + list(self.includes)
        return "\n".join(all_includes)
    
    def _generate_forward_declarations(self) -> str:
        """Generate forward declarations for classes"""
        if not self.classes:
            return ""
        
        decls = []
        for class_info in self.classes:
            decls.append(f"class {class_info.name};")
        
        return "\n".join(decls)
    
    def _generate_class(self, class_info: ClassInfo) -> str:
        """Generate a C++ class"""
        lines = []
        
        # Class declaration
        if class_info.bases:
            inheritance = " : " + ", ".join(f"public {base}" for base in class_info.bases)
        else:
            inheritance = ""
        
        lines.append(f"class {class_info.name}{inheritance} {{")
        
        # Generate sections
        private_members = []
        public_members = []
        
        # Constructor
        constructor = self._generate_constructor(class_info)
        if constructor:
            public_members.append(constructor)
        
        # Methods
        for method in class_info.methods:
            method_code = self._generate_method_declaration(method)
            if method.name.startswith('_') and method.name != '__init__':
                private_members.append(method_code)
            else:
                public_members.append(method_code)
        
        # Private section
        if private_members or class_info.attributes:
            lines.append("private:")
            
            # Attributes
            for attr in class_info.attributes:
                lines.append(f"    int {attr};  // TODO: infer correct type")
            
            # Private methods
            for member in private_members:
                lines.append(f"    {member}")
        
        # Public section
        if public_members:
            lines.append("public:")
            for member in public_members:
                lines.append(f"    {member}")
        
        lines.append("};")
        
        return "\n".join(lines)
    
    def _generate_constructor(self, class_info: ClassInfo) -> Optional[str]:
        """Generate constructor for a class"""
        init_method = None
        for method in class_info.methods:
            if method.name == "__init__":
                init_method = method
                break
        
        if not init_method:
            return f"{class_info.name}() = default;"
        
        # Generate constructor with parameters
        params = []
        for arg in init_method.args[1:]:  # Skip 'self'
            params.append(f"auto {arg}")  # TODO: Use proper type inference
        
        param_str = ", ".join(params)
        return f"{class_info.name}({param_str});"
    
    def _generate_method_declaration(self, method: FunctionInfo) -> str:
        """Generate method declaration"""
        if method.name == "__init__":
            return ""  # Constructor handled separately
        
        # Convert special methods
        if method.name == "__str__":
            return "std::string toString() const;"
        elif method.name == "__repr__":
            return "std::string repr() const;"
        
        return_type = "auto"  # TODO: Use type inference
        
        params = []
        for arg in method.args[1:] if method.is_method else method.args:  # Skip 'self' for methods
            params.append(f"auto {arg}")  # TODO: Use proper type inference
        
        param_str = ", ".join(params)
        
        modifiers = ""
        if method.is_static:
            modifiers = "static "
        
        return f"{modifiers}{return_type} {method.name}({param_str});"
    
    def _generate_function_declaration(self, func_info: FunctionInfo) -> str:
        """Generate function declaration"""
        return_type = "auto"  # TODO: Use type inference
        
        params = []
        for arg in func_info.args:
            params.append(f"auto {arg}")  # TODO: Use proper type inference
        
        param_str = ", ".join(params)
        return f"{return_type} {func_info.name}({param_str});"
    
    def _generate_function_implementation(self, node: ast.FunctionDef, func_info: FunctionInfo) -> str:
        """Generate function implementation"""
        return_type = "auto"  # TODO: Use type inference
        
        params = []
        for arg in func_info.args:
            params.append(f"auto {arg}")  # TODO: Use proper type inference
        
        param_str = ", ".join(params)
        
        lines = [f"{return_type} {func_info.name}({param_str}) {{"]
        
        # Generate function body
        body = self._generate_statements(node.body)
        for line in body:
            lines.append(f"    {line}")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_statements(self, statements: List[ast.stmt]) -> List[str]:
        """Generate C++ statements from Python AST statements"""
        lines = []
        
        for stmt in statements:
            if isinstance(stmt, ast.Return):
                lines.append(self._generate_return(stmt))
            elif isinstance(stmt, ast.Assign):
                lines.append(self._generate_assignment(stmt))
            elif isinstance(stmt, ast.If):
                lines.extend(self._generate_if(stmt))
            elif isinstance(stmt, ast.For):
                lines.extend(self._generate_for(stmt))
            elif isinstance(stmt, ast.While):
                lines.extend(self._generate_while(stmt))
            elif isinstance(stmt, ast.Expr):
                lines.append(self._generate_expression(stmt.value) + ";")
            else:
                lines.append(f"// TODO: Implement {type(stmt).__name__}")
        
        return lines
    
    def _generate_return(self, node: ast.Return) -> str:
        """Generate return statement"""
        if node.value:
            return f"return {self._generate_expression(node.value)};"
        else:
            return "return;"
    
    def _generate_assignment(self, node: ast.Assign) -> str:
        """Generate assignment statement"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            value = self._generate_expression(node.value)
            return f"auto {target} = {value};"
        else:
            return "// TODO: Complex assignment"
    
    def _generate_if(self, node: ast.If) -> List[str]:
        """Generate if statement"""
        lines = []
        condition = self._generate_expression(node.test)
        lines.append(f"if ({condition}) {{")
        
        # If body
        body = self._generate_statements(node.body)
        for line in body:
            lines.append(f"    {line}")
        
        lines.append("}")
        
        # Else clause
        if node.orelse:
            lines.append("else {")
            else_body = self._generate_statements(node.orelse)
            for line in else_body:
                lines.append(f"    {line}")
            lines.append("}")
        
        return lines
    
    def _generate_for(self, node: ast.For) -> List[str]:
        """Generate for loop"""
        lines = []
        
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                # Range-based for loop
                target = node.target.id
                if len(node.iter.args) == 1:
                    # range(n)
                    end = self._generate_expression(node.iter.args[0])
                    lines.append(f"for (int {target} = 0; {target} < {end}; ++{target}) {{")
                elif len(node.iter.args) == 2:
                    # range(start, end)
                    start = self._generate_expression(node.iter.args[0])
                    end = self._generate_expression(node.iter.args[1])
                    lines.append(f"for (int {target} = {start}; {target} < {end}; ++{target}) {{")
                else:
                    lines.append("// TODO: range with step")
            else:
                # Iterator-based for loop
                target = node.target.id
                container = self._generate_expression(node.iter)
                lines.append(f"for (const auto& {target} : {container}) {{")
        else:
            lines.append("// TODO: Complex for loop")
        
        # Loop body
        body = self._generate_statements(node.body)
        for line in body:
            lines.append(f"    {line}")
        
        lines.append("}")
        return lines
    
    def _generate_while(self, node: ast.While) -> List[str]:
        """Generate while loop"""
        lines = []
        condition = self._generate_expression(node.test)
        lines.append(f"while ({condition}) {{")
        
        # Loop body
        body = self._generate_statements(node.body)
        for line in body:
            lines.append(f"    {line}")
        
        lines.append("}")
        return lines
    
    def _generate_expression(self, node: ast.expr) -> str:
        """Generate C++ expression from Python expression"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            elif isinstance(node.value, bool):
                return "true" if node.value else "false"
            else:
                return str(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._generate_expression(node.left)
            right = self._generate_expression(node.right)
            op = self._get_binary_operator(node.op)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.Compare):
            return self._generate_comparison(node)
        elif isinstance(node, ast.Call):
            return self._generate_call(node)
        elif isinstance(node, ast.List):
            elements = [self._generate_expression(elt) for elt in node.elts]
            return "{" + ", ".join(elements) + "}"
        elif isinstance(node, ast.Attribute):
            obj = self._generate_expression(node.value)
            return f"{obj}.{node.attr}"
        else:
            return f"/* TODO: {type(node).__name__} */"
    
    def _generate_comparison(self, node: ast.Compare) -> str:
        """Generate comparison expression"""
        left = self._generate_expression(node.left)
        parts = [left]
        
        for op, comparator in zip(node.ops, node.comparators):
            op_str = self._get_comparison_operator(op)
            comp_str = self._generate_expression(comparator)
            parts.append(f"{op_str} {comp_str}")
        
        return " ".join(parts)
    
    def _generate_call(self, node: ast.Call) -> str:
        """Generate function call"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
              # Handle built-in functions
            if func_name == "print":
                if node.args:
                    args = [self._generate_expression(arg) for arg in node.args]
                    separator = ' << " " << '
                    return f'std::cout << {separator.join(args)} << std::endl'
                else:
                    return "std::cout << std::endl"
            elif func_name == "len":
                if node.args:
                    arg = self._generate_expression(node.args[0])
                    return f"{arg}.size()"
            elif func_name == "str":
                if node.args:
                    arg = self._generate_expression(node.args[0])
                    return f"std::to_string({arg})"
            
            # Regular function call
            args = [self._generate_expression(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"
        
        elif isinstance(node.func, ast.Attribute):
            obj = self._generate_expression(node.func.value)
            method = node.func.attr
            args = [self._generate_expression(arg) for arg in node.args]
            
            # Handle list methods
            if method == "append":
                return f"{obj}.push_back({args[0]})" if args else f"{obj}.push_back()"
            elif method == "pop":
                return f"{obj}.pop_back()"
            
            return f"{obj}.{method}({', '.join(args)})"
        
        return "/* TODO: Complex call */"
    
    def _get_binary_operator(self, op: ast.operator) -> str:
        """Convert Python binary operator to C++"""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",  # Note: integer division in C++
            ast.Mod: "%",
            ast.Pow: "/* TODO: pow */",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
        }
        return op_map.get(type(op), "/* unknown op */")
    
    def _get_comparison_operator(self, op: ast.cmpop) -> str:
        """Convert Python comparison operator to C++"""
        op_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "==",  # Simplified
            ast.IsNot: "!=",  # Simplified
        }
        return op_map.get(type(op), "/* unknown comparison */")
    
    def _find_function_node(self, tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
        """Find a function node in the AST by name"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None
