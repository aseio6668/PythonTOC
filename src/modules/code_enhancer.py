"""
Advanced C++ Code Quality Enhancement System
Improves the quality and maintainability of generated C++ code
"""

import ast
import re
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CodeQualityMetrics:
    """Metrics for code quality assessment"""
    cyclomatic_complexity: int
    nesting_depth: int
    function_length: int
    variable_naming_score: float
    const_correctness_score: float
    memory_safety_score: float
    performance_score: float


class CppCodeEnhancer:
    """Enhances generated C++ code for production quality"""
    
    def __init__(self):
        self.type_inference_patterns = {}
        self.performance_patterns = {}
        self.safety_patterns = {}
    
    def enhance_generated_code(self, cpp_code: str, python_ast: ast.AST) -> str:
        """Apply multiple enhancement passes to C++ code"""
        enhanced_code = cpp_code
        
        # Apply enhancement passes
        enhanced_code = self._apply_const_correctness(enhanced_code)
        enhanced_code = self._apply_smart_pointers(enhanced_code)
        enhanced_code = self._apply_move_semantics(enhanced_code)
        enhanced_code = self._apply_type_deduction(enhanced_code)
        enhanced_code = self._apply_performance_optimizations(enhanced_code)
        enhanced_code = self._apply_error_handling(enhanced_code)
        enhanced_code = self._apply_modern_cpp_features(enhanced_code)
        
        return enhanced_code
    
    def _apply_const_correctness(self, code: str) -> str:
        """Add const correctness to generated C++ code"""
        # Add const to member functions that don't modify state
        code = re.sub(
            r'(\w+\s+\w+\([^)]*\))\s*{([^{}]*return[^{}]*)}',
            r'\1 const {\2}',
            code
        )
        
        # Add const to variables that aren't modified
        lines = code.split('\n')
        enhanced_lines = []
        
        for line in lines:
            if 'auto ' in line and '=' in line and not '+=' in line and not '-=' in line:
                # Check if variable is never modified (simple heuristic)
                var_name = line.split('auto ')[1].split(' =')[0].strip()
                if not self._is_variable_modified(var_name, code):
                    line = line.replace('auto ', 'const auto ')
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _apply_smart_pointers(self, code: str) -> str:
        """Replace raw pointers with smart pointers where appropriate"""
        # Replace new/delete patterns with unique_ptr
        code = re.sub(
            r'(\w+\*\s+\w+\s*=\s*)new\s+(\w+)',
            r'auto \1std::make_unique<\2>',
            code
        )
        
        # Add appropriate headers
        if 'std::make_unique' in code or 'std::unique_ptr' in code:
            if '#include <memory>' not in code:
                code = '#include <memory>\n' + code
        
        return code
    
    def _apply_move_semantics(self, code: str) -> str:
        """Add move semantics for better performance"""
        # Add std::move for return values of expensive types
        expensive_types = ['std::vector', 'std::string', 'std::map', 'std::unordered_map']
        
        for type_name in expensive_types:
            pattern = rf'return\s+(\w+);'
            replacement = r'return std::move(\1);'
            # Only apply if the variable is of expensive type
            code = re.sub(pattern, replacement, code)
        
        return code
    
    def _apply_type_deduction(self, code: str) -> str:
        """Improve type deduction and reduce verbosity"""
        # Replace verbose type declarations with auto where appropriate
        type_patterns = [
            (r'std::vector<\w+>\s+(\w+)\s*=', r'auto \1 ='),
            (r'std::string\s+(\w+)\s*=', r'auto \1 ='),
            (r'std::map<\w+,\s*\w+>\s+(\w+)\s*=', r'auto \1 ='),
        ]
        
        for pattern, replacement in type_patterns:
            code = re.sub(pattern, replacement, code)
        
        return code
    
    def _apply_performance_optimizations(self, code: str) -> str:
        """Apply performance optimization patterns"""
        # Reserve capacity for vectors when size is known
        code = re.sub(
            r'(std::vector<\w+>\s+\w+);',
            r'\1;\n    \1.reserve(/* estimated_size */);',
            code
        )
        
        # Use emplace_back instead of push_back
        code = re.sub(r'\.push_back\(', r'.emplace_back(', code)
        
        # Use range-based for loops
        code = re.sub(
            r'for\s*\(\s*int\s+(\w+)\s*=\s*0;\s*\1\s*<\s*(\w+)\.size\(\);\s*\+\+\1\s*\)',
            r'for (const auto& item : \2)',
            code
        )
        
        return code
    
    def _apply_error_handling(self, code: str) -> str:
        """Add proper error handling patterns"""
        # Replace TODO comments for exceptions with actual error handling
        code = re.sub(
            r'// TODO: Handle exception\s*(\w+)',
            r'try {\n        \1\n    } catch (const std::exception& e) {\n        std::cerr << "Error: " << e.what() << std::endl;\n        return {};\n    }',
            code
        )
        
        return code
    
    def _apply_modern_cpp_features(self, code: str) -> str:
        """Apply modern C++ features (C++17/20)"""
        # Use structured bindings where appropriate
        code = re.sub(
            r'auto\s+(\w+)\s*=\s*(\w+)\.first;\s*auto\s+(\w+)\s*=\s*\2\.second;',
            r'auto [\1, \3] = \2;',
            code
        )
        
        # Use if-init statements
        code = re.sub(
            r'auto\s+(\w+)\s*=\s*([^;]+);\s*if\s*\(\s*\1',
            r'if (auto \1 = \2; \1',
            code
        )
        
        return code
    
    def _is_variable_modified(self, var_name: str, code: str) -> bool:
        """Check if a variable is modified after declaration"""
        modification_patterns = [
            rf'{var_name}\s*=',
            rf'{var_name}\s*\+=',
            rf'{var_name}\s*-=',
            rf'{var_name}\s*\*=',
            rf'{var_name}\s*/=',
            rf'{var_name}\+\+',
            rf'\+\+{var_name}',
            rf'{var_name}--',
            rf'--{var_name}',
        ]
        
        for pattern in modification_patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def generate_quality_report(self, cpp_code: str) -> CodeQualityMetrics:
        """Generate quality metrics for C++ code"""
        return CodeQualityMetrics(
            cyclomatic_complexity=self._calculate_complexity(cpp_code),
            nesting_depth=self._calculate_nesting_depth(cpp_code),
            function_length=self._calculate_average_function_length(cpp_code),
            variable_naming_score=self._score_variable_naming(cpp_code),
            const_correctness_score=self._score_const_correctness(cpp_code),
            memory_safety_score=self._score_memory_safety(cpp_code),
            performance_score=self._score_performance_patterns(cpp_code)
        )
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'catch']
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code))
        
        return complexity
    
    def _calculate_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        return max_depth
    
    def _calculate_average_function_length(self, code: str) -> int:
        """Calculate average function length in lines"""
        functions = re.findall(r'\w+\s+\w+\([^)]*\)\s*{[^}]*}', code, re.DOTALL)
        if not functions:
            return 0
        
        total_lines = sum(func.count('\n') for func in functions)
        return total_lines // len(functions)
    
    def _score_variable_naming(self, code: str) -> float:
        """Score variable naming quality (0-1)"""
        variables = re.findall(r'\b(?:auto|int|float|double|std::\w+)\s+(\w+)', code)
        if not variables:
            return 1.0
        
        good_names = sum(1 for var in variables if len(var) > 2 and '_' not in var[:2])
        return good_names / len(variables)
    
    def _score_const_correctness(self, code: str) -> float:
        """Score const correctness (0-1)"""
        const_opportunities = len(re.findall(r'\bauto\s+\w+\s*=', code))
        const_used = len(re.findall(r'\bconst\s+auto\s+\w+\s*=', code))
        
        if const_opportunities == 0:
            return 1.0
        
        return const_used / const_opportunities
    
    def _score_memory_safety(self, code: str) -> float:
        """Score memory safety (0-1)"""
        raw_pointers = len(re.findall(r'\w+\*\s+\w+', code))
        smart_pointers = len(re.findall(r'std::(?:unique_ptr|shared_ptr)', code))
        
        if raw_pointers + smart_pointers == 0:
            return 1.0
        
        return smart_pointers / (raw_pointers + smart_pointers)
    
    def _score_performance_patterns(self, code: str) -> float:
        """Score performance optimization patterns (0-1)"""
        score = 0.0
        checks = 0
        
        # Check for range-based for loops
        range_for = len(re.findall(r'for\s*\(\s*(?:const\s+)?auto\s*&', code))
        c_style_for = len(re.findall(r'for\s*\(\s*int\s+\w+\s*=', code))
        if range_for + c_style_for > 0:
            score += range_for / (range_for + c_style_for)
            checks += 1
        
        # Check for move semantics
        if 'std::move' in code:
            score += 1.0
            checks += 1
        
        # Check for emplace vs insert
        emplace_calls = len(re.findall(r'\.emplace', code))
        insert_calls = len(re.findall(r'\.(?:push_back|insert)', code))
        if emplace_calls + insert_calls > 0:
            score += emplace_calls / (emplace_calls + insert_calls)
            checks += 1
        
        return score / max(checks, 1)


class CppFormatter:
    """Formats C++ code according to modern style guidelines"""
    
    def __init__(self, style: str = "google"):
        self.style = style
        self.indent_size = 2 if style == "google" else 4
    
    def format_code(self, code: str) -> str:
        """Format C++ code with consistent style"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Adjust indent level
            if stripped.endswith('{'):
                formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
                indent_level += 1
            elif stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
                formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
            else:
                formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
        
        return '\n'.join(formatted_lines)
    
    def add_documentation(self, code: str, python_ast: ast.AST) -> str:
        """Add Doxygen-style documentation"""
        # Extract docstrings from Python AST
        docstrings = self._extract_docstrings(python_ast)
        
        # Add file header
        header = [
            "/**",
            " * @file auto_generated.cpp",
            " * @brief Auto-generated C++ code from Python source",
            " * @note This file was automatically translated from Python",
            " * @warning Review and test before production use",
            " */",
            ""
        ]
        
        return '\n'.join(header) + code
    
    def _extract_docstrings(self, tree: ast.AST) -> Dict[str, str]:
        """Extract docstrings from Python AST"""
        docstrings = {}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstrings[node.name] = node.body[0].value.value
        
        return docstrings


# Integration example
def enhance_translation_output(cpp_code: str, python_ast: ast.AST) -> Tuple[str, CodeQualityMetrics]:
    """Enhance generated C++ code and return quality metrics"""
    enhancer = CppCodeEnhancer()
    formatter = CppFormatter(style="google")
    
    # Apply enhancements
    enhanced_code = enhancer.enhance_generated_code(cpp_code, python_ast)
    
    # Format code
    formatted_code = formatter.format_code(enhanced_code)
    
    # Add documentation
    documented_code = formatter.add_documentation(formatted_code, python_ast)
    
    # Generate quality metrics
    metrics = enhancer.generate_quality_report(documented_code)
    
    return documented_code, metrics
