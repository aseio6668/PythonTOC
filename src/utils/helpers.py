"""
Helper functions for the Python to C++ translator
"""

import ast
from typing import List, Optional, Any
from pathlib import Path


def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path"""
    return Path(file_path).suffix


def is_python_file(file_path: str) -> bool:
    """Check if a file is a Python file"""
    return get_file_extension(file_path).lower() == '.py'


def is_cpp_file(file_path: str) -> bool:
    """Check if a file is a C++ file"""
    cpp_extensions = {'.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h', '.hxx', '.h++'}
    return get_file_extension(file_path).lower() in cpp_extensions


def sanitize_cpp_identifier(name: str) -> str:
    """
    Sanitize a Python identifier to be valid in C++
    
    Args:
        name: Python identifier
        
    Returns:
        Valid C++ identifier
    """
    # Replace invalid characters
    sanitized = name.replace('-', '_').replace(' ', '_')
    
    # Handle Python keywords that are invalid in C++
    cpp_keywords = {
        'and', 'or', 'not', 'class', 'struct', 'union', 'enum',
        'template', 'typename', 'namespace', 'using', 'auto',
        'const', 'static', 'extern', 'inline', 'virtual', 'override',
        'final', 'public', 'private', 'protected', 'friend',
        'new', 'delete', 'this', 'operator', 'sizeof', 'typeid'
    }
    
    if sanitized in cpp_keywords:
        sanitized += '_'
    
    # Ensure it starts with a letter or underscore
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
        sanitized = '_' + sanitized
    
    return sanitized or '_unnamed'


def get_cpp_type_from_python_type(python_type: str) -> str:
    """
    Get C++ type from Python type string
    
    Args:
        python_type: Python type as string
        
    Returns:
        Corresponding C++ type
    """
    type_mapping = {
        'int': 'int',
        'float': 'double',
        'str': 'std::string',
        'bool': 'bool',
        'list': 'std::vector',
        'dict': 'std::map',
        'set': 'std::set',
        'tuple': 'std::tuple',
        'None': 'void',
        'Any': 'auto',
    }
    
    return type_mapping.get(python_type, 'auto')


def extract_docstring(node: ast.FunctionDef) -> Optional[str]:
    """
    Extract docstring from a function node
    
    Args:
        node: Function AST node
        
    Returns:
        Docstring if present, None otherwise
    """
    if (node.body and 
        isinstance(node.body[0], ast.Expr) and 
        isinstance(node.body[0].value, ast.Constant) and 
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None


def format_cpp_comment(text: str, style: str = "//") -> List[str]:
    """
    Format text as C++ comments
    
    Args:
        text: Text to format as comments
        style: Comment style ("//", "/*", or "/**")
        
    Returns:
        List of comment lines
    """
    lines = text.strip().split('\n')
    
    if style == "//":
        return [f"// {line}" for line in lines]
    elif style == "/*":
        if len(lines) == 1:
            return [f"/* {lines[0]} */"]
        else:
            result = ["/*"]
            result.extend(f" * {line}" for line in lines)
            result.append(" */")
            return result
    elif style == "/**":
        result = ["/**"]
        result.extend(f" * {line}" for line in lines)
        result.append(" */")
        return result
    else:
        return [f"// {line}" for line in lines]


def indent_code(code: str, indent_level: int = 1, indent_str: str = "    ") -> str:
    """
    Indent code by specified level
    
    Args:
        code: Code to indent
        indent_level: Number of indentation levels
        indent_str: Indentation string
        
    Returns:
        Indented code
    """
    lines = code.split('\n')
    indent = indent_str * indent_level
    return '\n'.join(indent + line if line.strip() else line for line in lines)


def get_python_version_info() -> str:
    """Get Python version information for generated C++ comments"""
    import sys
    return f"Python {sys.version}"


def create_header_guard(filename: str) -> tuple[str, str]:
    """
    Create header guard macros for C++ header files
    
    Args:
        filename: Header file name
        
    Returns:
        Tuple of (opening guard, closing guard)
    """
    guard_name = Path(filename).stem.upper() + "_H"
    guard_name = sanitize_cpp_identifier(guard_name)
    
    opening = f"#ifndef {guard_name}\n#define {guard_name}"
    closing = f"#endif // {guard_name}"
    
    return opening, closing


def analyze_complexity(node: ast.AST) -> dict:
    """
    Analyze code complexity metrics
    
    Args:
        node: AST node to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    metrics = {
        'lines': 0,
        'functions': 0,
        'classes': 0,
        'loops': 0,
        'conditions': 0,
        'max_depth': 0
    }
    
    def visit_node(n, depth=0):
        metrics['max_depth'] = max(metrics['max_depth'], depth)
        
        if isinstance(n, ast.FunctionDef):
            metrics['functions'] += 1
        elif isinstance(n, ast.ClassDef):
            metrics['classes'] += 1
        elif isinstance(n, (ast.For, ast.While)):
            metrics['loops'] += 1
        elif isinstance(n, (ast.If, ast.IfExp)):
            metrics['conditions'] += 1
        
        for child in ast.iter_child_nodes(n):
            visit_node(child, depth + 1)
    
    visit_node(node)
    
    # Estimate lines (rough approximation)
    metrics['lines'] = len(ast.unparse(node).split('\n'))
    
    return metrics


def generate_translation_report(parser_info: dict, type_info: dict, complexity: dict) -> str:
    """
    Generate a translation report
    
    Args:
        parser_info: Information from the parser
        type_info: Type inference information
        complexity: Complexity metrics
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "# Python to C++ Translation Report",
        "",
        "## Code Analysis",
        f"- Functions: {len(parser_info.get('functions', []))}",
        f"- Classes: {len(parser_info.get('classes', []))}",
        f"- Imports: {len(parser_info.get('imports', []))}",
        f"- Estimated lines: {complexity.get('lines', 0)}",
        f"- Max nesting depth: {complexity.get('max_depth', 0)}",
        "",
        "## Type Inference",
        f"- Variables with inferred types: {len(type_info.get('variable_types', {}))}",
        f"- Functions with return types: {len(type_info.get('function_return_types', {}))}",
        "",
        "## Translation Notes",
        "- Memory management: Consider using smart pointers for object lifetimes",
        "- Error handling: Python exceptions should be converted to C++ exceptions",
        "- Performance: Review generated code for optimization opportunities",
        "",
        "## Recommended Next Steps",
        "1. Review generated C++ code for correctness",
        "2. Add proper type annotations to improve translation",
        "3. Test the generated code with appropriate inputs",
        "4. Consider adding CMake build configuration",
        "5. Add unit tests for the translated functions",
    ]
    
    return '\n'.join(report_lines)
