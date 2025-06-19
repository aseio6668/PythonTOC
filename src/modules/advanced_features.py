"""
Advanced Python Feature Translator
Handles complex Python constructs that are challenging to translate to C++
"""

import ast
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class TranslationStrategy:
    """Strategy for translating a specific Python feature"""
    feature_name: str
    cpp_equivalent: str
    requires_headers: List[str]
    complexity_score: int  # 1-10, 10 being most complex
    notes: str


class AdvancedFeatureTranslator:
    """Translates advanced Python features to C++"""
    
    def __init__(self):
        self.translation_strategies = self._initialize_strategies()
        self.required_headers = set()
        self.helper_functions = set()
    
    def _initialize_strategies(self) -> Dict[str, TranslationStrategy]:
        """Initialize translation strategies for Python features"""
        return {
            'list_comprehension': TranslationStrategy(
                feature_name="List Comprehension",
                cpp_equivalent="std::transform or range-based for loop",
                requires_headers=["<algorithm>", "<vector>"],
                complexity_score=3,
                notes="Convert to functional style or traditional loop"
            ),
            'dict_comprehension': TranslationStrategy(
                feature_name="Dictionary Comprehension",
                cpp_equivalent="std::transform with std::map",
                requires_headers=["<algorithm>", "<map>"],
                complexity_score=4,
                notes="Use std::map with transform or manual loop"
            ),
            'generator_expression': TranslationStrategy(
                feature_name="Generator Expression",
                cpp_equivalent="Custom iterator or lazy evaluation",
                requires_headers=["<iterator>", "<memory>"],
                complexity_score=8,
                notes="Requires custom iterator implementation"
            ),
            'lambda_function': TranslationStrategy(
                feature_name="Lambda Function",
                cpp_equivalent="C++ lambda or std::function",
                requires_headers=["<functional>"],
                complexity_score=2,
                notes="Direct translation to C++ lambda"
            ),
            'decorator': TranslationStrategy(
                feature_name="Decorator",
                cpp_equivalent="Template or wrapper class",
                requires_headers=["<functional>", "<type_traits>"],
                complexity_score=9,
                notes="Complex - use templates or design patterns"
            ),
            'context_manager': TranslationStrategy(
                feature_name="Context Manager (with statement)",
                cpp_equivalent="RAII pattern",
                requires_headers=["<memory>"],
                complexity_score=6,
                notes="Use RAII destructors for cleanup"
            ),
            'async_function': TranslationStrategy(
                feature_name="Async Function",
                cpp_equivalent="std::future or coroutines",
                requires_headers=["<future>", "<coroutine>"],
                complexity_score=10,
                notes="Use C++20 coroutines or std::async"
            ),
            'f_string': TranslationStrategy(
                feature_name="F-string",
                cpp_equivalent="std::format or stringstream",
                requires_headers=["<format>", "<sstream>"],
                complexity_score=3,
                notes="Use std::format (C++20) or stringstream"
            ),
            'multiple_assignment': TranslationStrategy(
                feature_name="Multiple Assignment/Unpacking",
                cpp_equivalent="std::tuple with structured bindings",
                requires_headers=["<tuple>"],
                complexity_score=4,
                notes="Use std::tuple and C++17 structured bindings"
            ),
            'yield_statement': TranslationStrategy(
                feature_name="Yield Statement/Generator",
                cpp_equivalent="Custom generator class",
                requires_headers=["<iterator>", "<optional>"],
                complexity_score=9,
                notes="Implement custom generator with iterator pattern"
            )
        }
    
    def translate_list_comprehension(self, node: ast.ListComp) -> str:
        """Translate list comprehension to C++ code"""
        self.required_headers.update(["<algorithm>", "<vector>"])
        
        # Extract components
        element = self._translate_expression(node.elt)
        target = node.generators[0].target.id if isinstance(node.generators[0].target, ast.Name) else "item"
        iter_expr = self._translate_expression(node.generators[0].iter)
        
        # Check if there are conditions
        conditions = []
        for generator in node.generators:
            for if_clause in generator.ifs:
                conditions.append(self._translate_expression(if_clause))
        
        if conditions:
            # Use traditional for loop with conditions
            condition_str = " && ".join(conditions)
            return f"""{{
    std::vector<auto> result;
    for (const auto& {target} : {iter_expr}) {{
        if ({condition_str}) {{
            result.push_back({element});
        }}
    }}
    return result;
}}()"""
        else:
            # Use std::transform
            return f"""{{
    std::vector<auto> result;
    std::transform({iter_expr}.begin(), {iter_expr}.end(), 
                   std::back_inserter(result), 
                   [](const auto& {target}) {{ return {element}; }});
    return result;
}}()"""
    
    def translate_lambda_function(self, node: ast.Lambda) -> str:
        """Translate lambda function to C++ lambda"""
        self.required_headers.add("<functional>")
        
        # Extract parameters
        params = [arg.arg for arg in node.args.args]
        param_list = ", ".join(f"auto {param}" for param in params)
        
        # Extract body
        body = self._translate_expression(node.body)
        
        return f"[&]({param_list}) {{ return {body}; }}"
    
    def translate_f_string(self, node: ast.JoinedStr) -> str:
        """Translate f-string to C++ format string"""
        self.required_headers.add("<format>")
        
        format_parts = []
        args = []
        
        for value in node.values:
            if isinstance(value, ast.Constant):
                format_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                format_parts.append("{}")
                args.append(self._translate_expression(value.value))
        
        format_string = "".join(format_parts)
        
        if args:
            args_str = ", " + ", ".join(args)
        else:
            args_str = ""
        
        return f'std::format("{format_string}"{args_str})'
    
    def translate_context_manager(self, node: ast.With) -> str:
        """Translate with statement to RAII pattern"""
        self.required_headers.add("<memory>")
        
        # Extract context manager
        context_expr = self._translate_expression(node.items[0].context_expr)
        var_name = node.items[0].optional_vars.id if node.items[0].optional_vars else "resource"
        
        # Generate RAII wrapper
        body_lines = [self._translate_statement(stmt) for stmt in node.body]
        body = "\\n    ".join(body_lines)
        
        return f"""{{
    auto {var_name} = {context_expr};
    // RAII: resource will be automatically cleaned up
    {body}
}}"""
    
    def translate_multiple_assignment(self, node: ast.Assign) -> str:
        """Translate multiple assignment to structured bindings"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            self.required_headers.add("<tuple>")
            
            # Extract target names
            targets = [name.id for name in node.targets[0].elts if isinstance(name, ast.Name)]
            value = self._translate_expression(node.value)
            
            # Use structured bindings (C++17)
            target_list = ", ".join(targets)
            return f"auto [{target_list}] = {value};"
        
        return self._translate_statement(node)
    
    def translate_generator_expression(self, node: ast.GeneratorExp) -> str:
        """Translate generator expression to custom iterator"""
        self.required_headers.update(["<iterator>", "<optional>"])
        self.helper_functions.add("generator_helper")
        
        # This is complex - create a generator class
        element = self._translate_expression(node.elt)
        target = node.generators[0].target.id if isinstance(node.generators[0].target, ast.Name) else "item"
        iter_expr = self._translate_expression(node.generators[0].iter)
        
        class_name = f"Generator_{hash(str(node)) % 10000}"
        
        return f"""{{
    class {class_name} {{
    private:
        decltype({iter_expr}) source;
        typename decltype({iter_expr})::iterator current;
        
    public:
        {class_name}(decltype({iter_expr}) src) : source(std::move(src)), current(source.begin()) {{}}
        
        std::optional<decltype({element})> next() {{
            if (current != source.end()) {{
                auto {target} = *current++;
                return {element};
            }}
            return std::nullopt;
        }}
    }};
    
    return {class_name}({iter_expr});
}}()"""
    
    def translate_yield_function(self, node: ast.FunctionDef) -> str:
        """Translate generator function with yield to C++ generator class"""
        self.required_headers.update(["<iterator>", "<optional>", "<memory>"])
        
        func_name = node.name
        class_name = f"{func_name}_generator"
        
        # Find yield statements
        yield_nodes = []
        for child in ast.walk(node):
            if isinstance(child, ast.Yield):
                yield_nodes.append(child)
        
        if not yield_nodes:
            return self._translate_function(node)
        
        # Generate generator class
        return f"""
class {class_name} {{
private:
    int state = 0;
    // Add member variables for function parameters and locals
    
public:
    std::optional<auto> next() {{
        switch (state) {{
            case 0:
                // Function body up to first yield
                state = 1;
                return /* first yield value */;
            case 1:
                // Continue after first yield
                state = 2;
                return /* second yield value */;
            default:
                return std::nullopt;
        }}
    }}
    
    // Iterator interface
    class iterator {{
        {class_name}* gen;
        std::optional<auto> current;
        
    public:
        iterator({class_name}* g) : gen(g) {{ current = gen->next(); }}
        auto operator*() {{ return *current; }}
        iterator& operator++() {{ current = gen->next(); return *this; }}
        bool operator!=(const iterator& other) {{ return current.has_value(); }}
    }};
    
    iterator begin() {{ return iterator(this); }}
    iterator end() {{ return iterator(nullptr); }}
}};

{class_name} {func_name}(/* parameters */) {{
    return {class_name}(/* parameter values */);
}}"""
    
    def translate_decorator(self, node: ast.FunctionDef) -> str:
        """Translate decorated function to C++ template/wrapper"""
        if not node.decorator_list:
            return self._translate_function(node)
        
        self.required_headers.update(["<functional>", "<type_traits>"])
        
        func_name = node.name
        decorator_names = [self._translate_expression(dec) for dec in node.decorator_list]
        
        # Simple approach: create wrapper templates
        wrappers = []
        for i, decorator in enumerate(decorator_names):
            wrapper_name = f"{func_name}_wrapped_{i}"
            wrappers.append(f"""
template<typename Func>
auto {decorator}_wrapper(Func&& func) {{
    return [func = std::forward<Func>(func)](auto&&... args) {{
        // Decorator logic for {decorator}
        return func(std::forward<decltype(args)>(args)...);
    }};
}}""")
        
        # Apply decorators in reverse order
        final_function = func_name
        for decorator in reversed(decorator_names):
            final_function = f"{decorator}_wrapper({final_function})"
        
        return "\\n".join(wrappers) + f"\\nauto {func_name}_decorated = {final_function};"
    
    def translate_async_function(self, node: ast.AsyncFunctionDef) -> str:
        """Translate async function to C++ coroutine or future"""
        self.required_headers.update(["<future>", "<coroutine>"])
        
        func_name = node.name
        
        # C++20 coroutine approach
        return f"""
#include <coroutine>

struct {func_name}_awaitable {{
    struct promise_type {{
        {func_name}_awaitable get_return_object() {{
            return {func_name}_awaitable{{std::coroutine_handle<promise_type>::from_promise(*this)}};
        }}
        std::suspend_never initial_suspend() {{ return {{}}; }}
        std::suspend_never final_suspend() noexcept {{ return {{}}; }}
        void return_void() {{}}
        void unhandled_exception() {{}}
    }};
    
    std::coroutine_handle<promise_type> handle;
    
    {func_name}_awaitable(std::coroutine_handle<promise_type> h) : handle(h) {{}}
    ~{func_name}_awaitable() {{ if (handle) handle.destroy(); }}
}};

{func_name}_awaitable {func_name}() {{
    // Async function body
    co_return;
}}"""
    
    def _translate_expression(self, node: ast.AST) -> str:
        """Translate an AST expression to C++ (simplified)"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return str(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._translate_expression(node.left)
            right = self._translate_expression(node.right)
            op = self._translate_operator(node.op)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.Call):
            func = self._translate_expression(node.func)
            args = [self._translate_expression(arg) for arg in node.args]
            return f"{func}({', '.join(args)})"
        else:
            return "/* TODO: Complex expression */"
    
    def _translate_statement(self, node: ast.AST) -> str:
        """Translate an AST statement to C++ (simplified)"""
        if isinstance(node, ast.Expr):
            return self._translate_expression(node.value) + ";"
        elif isinstance(node, ast.Return):
            if node.value:
                return f"return {self._translate_expression(node.value)};"
            return "return;"
        else:
            return "/* TODO: Complex statement */"
    
    def _translate_function(self, node: ast.FunctionDef) -> str:
        """Translate a regular function (placeholder)"""
        return f"auto {node.name}() {{ /* TODO: Function body */ }}"
    
    def _translate_operator(self, op: ast.operator) -> str:
        """Translate Python operator to C++"""
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.Pow: '^', ast.LShift: '<<', ast.RShift: '>>',
            ast.BitOr: '|', ast.BitXor: '^', ast.BitAnd: '&'
        }
        return op_map.get(type(op), '/*unknown_op*/')
    
    def get_required_headers(self) -> List[str]:
        """Get all headers required by translated features"""
        return sorted(list(self.required_headers))
    
    def get_helper_functions(self) -> List[str]:
        """Get helper functions needed for complex translations"""
        helpers = []
        if "generator_helper" in self.helper_functions:
            helpers.append("""
// Helper for generator expressions
template<typename T>
class optional_iterator {
    std::optional<T> current;
    std::function<std::optional<T>()> next_func;
    
public:
    optional_iterator(std::function<std::optional<T>()> func) : next_func(func) {
        current = next_func();
    }
    
    T operator*() { return *current; }
    optional_iterator& operator++() { current = next_func(); return *this; }
    bool operator!=(const optional_iterator& other) { return current.has_value(); }
};""")
        
        return helpers
    
    def analyze_advanced_features(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze which advanced features are used in the code"""
        feature_counts = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                feature_counts['list_comprehension'] = feature_counts.get('list_comprehension', 0) + 1
            elif isinstance(node, ast.DictComp):
                feature_counts['dict_comprehension'] = feature_counts.get('dict_comprehension', 0) + 1
            elif isinstance(node, ast.GeneratorExp):
                feature_counts['generator_expression'] = feature_counts.get('generator_expression', 0) + 1
            elif isinstance(node, ast.Lambda):
                feature_counts['lambda_function'] = feature_counts.get('lambda_function', 0) + 1
            elif isinstance(node, ast.FunctionDef) and node.decorator_list:
                feature_counts['decorator'] = feature_counts.get('decorator', 0) + 1
            elif isinstance(node, ast.With):
                feature_counts['context_manager'] = feature_counts.get('context_manager', 0) + 1
            elif isinstance(node, ast.AsyncFunctionDef):
                feature_counts['async_function'] = feature_counts.get('async_function', 0) + 1
            elif isinstance(node, ast.JoinedStr):
                feature_counts['f_string'] = feature_counts.get('f_string', 0) + 1
            elif isinstance(node, ast.Yield):
                feature_counts['yield_statement'] = feature_counts.get('yield_statement', 0) + 1
        
        return feature_counts
    
    def generate_feature_report(self, feature_counts: Dict[str, int]) -> str:
        """Generate a report of advanced features and their translation complexity"""
        lines = [
            "# Advanced Python Features Analysis",
            "",
            "This report analyzes advanced Python features in your code and their C++ translation complexity.",
            ""
        ]
        
        if not feature_counts:
            lines.append("✅ No advanced Python features detected. Translation should be straightforward.")
            return '\\n'.join(lines)
        
        total_complexity = 0
        for feature, count in feature_counts.items():
            strategy = self.translation_strategies.get(feature)
            if strategy:
                complexity = strategy.complexity_score * count
                total_complexity += complexity
                
                lines.extend([
                    f"## {strategy.feature_name} ({count} occurrences)",
                    f"- **Complexity Score**: {strategy.complexity_score}/10",
                    f"- **C++ Equivalent**: {strategy.cpp_equivalent}",
                    f"- **Required Headers**: {', '.join(strategy.requires_headers)}",
                    f"- **Notes**: {strategy.notes}",
                    ""
                ])
        
        # Overall assessment
        lines.extend([
            "## Overall Assessment",
            f"- **Total Complexity Score**: {total_complexity}",
            ""
        ])
        
        if total_complexity <= 10:
            lines.append("✅ **Low complexity** - Most features can be translated automatically.")
        elif total_complexity <= 30:
            lines.append("⚠️ **Medium complexity** - Some manual intervention may be required.")
        else:
            lines.append("🔴 **High complexity** - Significant manual work needed for proper translation.")
        
        return '\\n'.join(lines)


# Example integration function
def analyze_and_translate_advanced_features(python_code: str) -> Tuple[str, str]:
    """Analyze advanced features and provide translation guidance"""
    tree = ast.parse(python_code)
    translator = AdvancedFeatureTranslator()
    
    # Analyze features
    feature_counts = translator.analyze_advanced_features(tree)
    
    # Generate report
    report = translator.generate_feature_report(feature_counts)
    
    # Get required headers for translation
    headers = translator.get_required_headers()
    header_includes = '\\n'.join(f'#include {header}' for header in headers)
    
    return report, header_includes
