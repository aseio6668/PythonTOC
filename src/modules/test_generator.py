"""
Automatic Test Generation System for Python-to-C++ Translation
Generates comprehensive test suites for translated code validation
"""

import ast
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json


@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    function_name: str
    inputs: List[Any]
    expected_output: Any
    test_type: str  # 'unit', 'integration', 'performance'
    description: str


@dataclass
class TestSuite:
    """Collection of test cases for a module"""
    module_name: str
    test_cases: List[TestCase]
    setup_code: str = ""
    teardown_code: str = ""


@dataclass
class TestResult:
    """Results from running a test"""
    test_name: str
    passed: bool
    python_output: Any
    cpp_output: Any
    execution_time_python: float
    execution_time_cpp: float
    error_message: Optional[str] = None


class TestGenerator:
    """Generates comprehensive test suites for translated code"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def analyze_python_functions(self, python_file: Path) -> List[Dict]:
        """Analyze Python functions to understand their signature and behavior"""
        with open(python_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'arg_types': self._infer_argument_types(node),
                    'return_type': self._infer_return_type(node),
                    'docstring': ast.get_docstring(node),
                    'complexity': self._estimate_complexity(node)
                }
                functions.append(func_info)
        
        return functions
    
    def generate_test_cases(self, python_file: Path, num_tests_per_function: int = 5) -> TestSuite:
        """Generate comprehensive test cases for a Python module"""
        functions = self.analyze_python_functions(python_file)
        test_cases = []
        
        for func in functions:
            # Skip private functions and __init__
            if func['name'].startswith('_'):
                continue
            
            func_tests = self._generate_function_tests(func, num_tests_per_function)
            test_cases.extend(func_tests)
        
        return TestSuite(
            module_name=python_file.stem,
            test_cases=test_cases,
            setup_code=self._generate_setup_code(functions),
            teardown_code=""
        )
    
    def _generate_function_tests(self, func_info: Dict, num_tests: int) -> List[TestCase]:
        """Generate test cases for a specific function"""
        test_cases = []
        
        for i in range(num_tests):
            inputs = self._generate_test_inputs(func_info['args'], func_info['arg_types'])
            
            test_case = TestCase(
                name=f"test_{func_info['name']}_{i+1}",
                function_name=func_info['name'],
                inputs=inputs,
                expected_output=None,  # Will be filled by running Python version
                test_type='unit',
                description=f"Test case {i+1} for {func_info['name']}"
            )
            test_cases.append(test_case)
        
        # Add edge cases
        edge_cases = self._generate_edge_cases(func_info)
        test_cases.extend(edge_cases)
        
        return test_cases
    
    def _generate_test_inputs(self, arg_names: List[str], arg_types: Dict) -> List[Any]:
        """Generate appropriate test inputs based on argument types"""
        inputs = []
        
        for arg_name in arg_names:
            arg_type = arg_types.get(arg_name, 'auto')
            
            if arg_type in ['int', 'auto']:
                inputs.append(self._random_int())
            elif arg_type == 'float':
                inputs.append(self._random_float())
            elif arg_type == 'str':
                inputs.append(self._random_string())
            elif arg_type == 'list':
                inputs.append(self._random_list())
            elif arg_type == 'bool':
                inputs.append(self._random_bool())
            else:
                inputs.append(42)  # Default fallback
        
        return inputs
    
    def _generate_edge_cases(self, func_info: Dict) -> List[TestCase]:
        """Generate edge case tests for a function"""
        edge_cases = []
        
        # Common edge cases based on function signature
        if any('int' in str(arg_type) for arg_type in func_info['arg_types'].values()):
            # Test with zero, negative, large numbers
            edge_inputs = [
                [0] * len(func_info['args']),
                [-1] * len(func_info['args']),
                [1000000] * len(func_info['args'])
            ]
            
            for i, inputs in enumerate(edge_inputs):
                edge_case = TestCase(
                    name=f"test_{func_info['name']}_edge_{i+1}",
                    function_name=func_info['name'],
                    inputs=inputs,
                    expected_output=None,
                    test_type='edge_case',
                    description=f"Edge case {i+1} for {func_info['name']}"
                )
                edge_cases.append(edge_case)
        
        return edge_cases
    
    def run_python_tests(self, python_file: Path, test_suite: TestSuite) -> Dict[str, Any]:
        """Run test cases against the original Python code to get expected outputs"""
        results = {}
        
        # Create test runner script
        test_script = self._create_python_test_script(python_file, test_suite)
        test_file = self.temp_dir / "test_runner.py"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        # Run tests
        try:
            result = subprocess.run(
                ['python', str(test_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse JSON output
                results = json.loads(result.stdout)
        except Exception as e:
            print(f"Error running Python tests: {e}")
        
        return results
    
    def run_cpp_tests(self, cpp_file: Path, test_suite: TestSuite) -> Dict[str, Any]:
        """Run test cases against the translated C++ code"""
        results = {}
        
        # Create C++ test file
        test_cpp = self._create_cpp_test_file(cpp_file, test_suite)
        test_file = self.temp_dir / "test_runner.cpp"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_cpp)
        
        # Compile and run
        executable = self._compile_cpp_tests(test_file)
        if executable:
            try:
                result = subprocess.run(
                    [str(executable)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    results = json.loads(result.stdout)
            except Exception as e:
                print(f"Error running C++ tests: {e}")
        
        return results
    
    def compare_results(self, python_results: Dict, cpp_results: Dict, 
                       test_suite: TestSuite) -> List[TestResult]:
        """Compare Python and C++ test results"""
        comparison_results = []
        
        for test_case in test_suite.test_cases:
            test_name = test_case.name
            
            python_result = python_results.get(test_name, {})
            cpp_result = cpp_results.get(test_name, {})
            
            passed = self._results_match(
                python_result.get('output'),
                cpp_result.get('output')
            )
            
            test_result = TestResult(
                test_name=test_name,
                passed=passed,
                python_output=python_result.get('output'),
                cpp_output=cpp_result.get('output'),
                execution_time_python=python_result.get('execution_time', 0.0),
                execution_time_cpp=cpp_result.get('execution_time', 0.0),
                error_message=cpp_result.get('error') if not passed else None
            )
            comparison_results.append(test_result)
        
        return comparison_results
    
    def generate_test_report(self, test_results: List[TestResult], 
                           output_file: Path) -> str:
        """Generate a comprehensive test report"""
        passed_tests = [r for r in test_results if r.passed]
        failed_tests = [r for r in test_results if not r.passed]
        
        total_tests = len(test_results)
        pass_rate = len(passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        # Calculate performance statistics
        avg_speedup = 0.0
        if passed_tests:
            speedups = []
            for result in passed_tests:
                if result.execution_time_python > 0 and result.execution_time_cpp > 0:
                    speedup = result.execution_time_python / result.execution_time_cpp
                    speedups.append(speedup)
            
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        
        report_lines = [
            "# Translation Test Report",
            "",
            "## Summary",
            f"- **Total Tests**: {total_tests}",
            f"- **Passed**: {len(passed_tests)} ({pass_rate:.1f}%)",
            f"- **Failed**: {len(failed_tests)}",
            f"- **Average Speedup**: {avg_speedup:.2f}x",
            "",
            "## Test Results",
            ""
        ]
        
        # Passed tests
        if passed_tests:
            report_lines.extend([
                "### ✅ Passed Tests",
                ""
            ])
            for result in passed_tests:
                speedup = (result.execution_time_python / result.execution_time_cpp 
                          if result.execution_time_cpp > 0 else 0)
                report_lines.append(
                    f"- **{result.test_name}**: "
                    f"Output matches (Speedup: {speedup:.2f}x)"
                )
            report_lines.append("")
        
        # Failed tests
        if failed_tests:
            report_lines.extend([
                "### ❌ Failed Tests",
                ""
            ])
            for result in failed_tests:
                report_lines.extend([
                    f"#### {result.test_name}",
                    f"- **Python Output**: `{result.python_output}`",
                    f"- **C++ Output**: `{result.cpp_output}`",
                    f"- **Error**: {result.error_message or 'Output mismatch'}",
                    ""
                ])
        
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if pass_rate < 80:
            report_lines.extend([
                "- ⚠️ **Low pass rate**: Review translation logic for accuracy",
                "- Consider adding more type hints to improve translation"
            ])
        
        if avg_speedup < 1.5:
            report_lines.extend([
                "- ⚠️ **Low performance gain**: Optimize generated C++ code",
                "- Consider using compiler optimizations (-O2, -O3)"
            ])
        
        if failed_tests:
            report_lines.extend([
                "- 🔧 **Fix failed tests**: Review specific translation issues",
                "- Add manual test cases for complex scenarios"
            ])
        
        report_content = '\n'.join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    def generate_google_test_suite(self, test_suite: TestSuite, 
                                 python_results: Dict, output_file: Path):
        """Generate Google Test compatible C++ test suite"""
        test_code = [
            "#include <gtest/gtest.h>",
            "#include <iostream>",
            "#include <vector>",
            "#include <string>",
            "",
            "// Include the translated module",
            f'#include "{test_suite.module_name}.h"',
            "",
            f"class {test_suite.module_name.capitalize()}Test : public ::testing::Test {{",
            "protected:",
            "    void SetUp() override {",
            "        // Setup code",
            "    }",
            "",
            "    void TearDown() override {",
            "        // Cleanup code", 
            "    }",
            "};",
            ""
        ]
        
        for test_case in test_suite.test_cases:
            python_result = python_results.get(test_case.name, {})
            expected_output = python_result.get('output')
            
            if expected_output is not None:
                test_method = [
                    f"TEST_F({test_suite.module_name.capitalize()}Test, {test_case.name}) {{",
                    f"    // {test_case.description}",
                ]
                
                # Generate input parameters
                if test_case.inputs:
                    for i, input_val in enumerate(test_case.inputs):
                        if isinstance(input_val, str):
                            test_method.append(f'    auto arg{i} = std::string("{input_val}");')
                        else:
                            test_method.append(f'    auto arg{i} = {input_val};')
                
                # Generate function call
                args_str = ", ".join([f"arg{i}" for i in range(len(test_case.inputs))])
                test_method.append(f"    auto result = {test_case.function_name}({args_str});")
                
                # Generate assertion
                if isinstance(expected_output, (int, float)):
                    test_method.append(f"    EXPECT_EQ(result, {expected_output});")
                elif isinstance(expected_output, str):
                    test_method.append(f'    EXPECT_EQ(result, std::string("{expected_output}"));')
                elif isinstance(expected_output, bool):
                    test_method.append(f"    EXPECT_EQ(result, {'true' if expected_output else 'false'});")
                else:
                    test_method.append(f"    // TODO: Add assertion for complex type")
                
                test_method.append("}")
                test_method.append("")
                
                test_code.extend(test_method)
        
        test_code.extend([
            "int main(int argc, char **argv) {",
            "    ::testing::InitGoogleTest(&argc, argv);",
            "    return RUN_ALL_TESTS();",
            "}"
        ])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_code))
    
    # Helper methods
    def _infer_argument_types(self, func_node: ast.FunctionDef) -> Dict[str, str]:
        """Infer argument types from function definition"""
        types = {}
        for arg in func_node.args.args:
            if arg.annotation:
                types[arg.arg] = self._annotation_to_string(arg.annotation)
            else:
                types[arg.arg] = 'auto'
        return types
    
    def _infer_return_type(self, func_node: ast.FunctionDef) -> str:
        """Infer return type from function definition"""
        if func_node.returns:
            return self._annotation_to_string(func_node.returns)
        return 'auto'
    
    def _annotation_to_string(self, annotation) -> str:
        """Convert AST annotation to string"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            return f"{annotation.value.id}[{annotation.slice.id}]"
        return 'auto'
    
    def _estimate_complexity(self, func_node: ast.FunctionDef) -> int:
        """Estimate function complexity based on AST"""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def _generate_setup_code(self, functions: List[Dict]) -> str:
        """Generate setup code for tests"""
        return "# Test setup code\nimport sys\nimport os\n"
    
    def _random_int(self) -> int:
        import random
        return random.randint(-100, 100)
    
    def _random_float(self) -> float:
        import random
        return random.uniform(-100.0, 100.0)
    
    def _random_string(self) -> str:
        import random
        import string
        length = random.randint(1, 10)
        return ''.join(random.choices(string.ascii_letters, k=length))
    
    def _random_list(self) -> List[int]:
        import random
        length = random.randint(1, 5)
        return [random.randint(1, 10) for _ in range(length)]
    
    def _random_bool(self) -> bool:
        import random
        return random.choice([True, False])
    
    def _create_python_test_script(self, python_file: Path, test_suite: TestSuite) -> str:
        """Create Python test runner script"""
        script_lines = [
            "import json",
            "import time", 
            "import sys",
            "import os",
            "",
            f"# Import the module to test",
            f"sys.path.append('{python_file.parent}')",
        ]
        
        # Import functions from the module
        with open(python_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract function definitions
        tree = ast.parse(content)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if functions:
            import_line = f"from {python_file.stem} import " + ", ".join(functions)
            script_lines.append(import_line)
        
        script_lines.extend([
            "",
            "results = {}",
            ""
        ])
        
        # Add test cases
        for test_case in test_suite.test_cases:
            script_lines.extend([
                f"# Test: {test_case.name}",
                "try:",
                "    start_time = time.perf_counter()",
            ])
            
            # Generate function call
            args_str = ", ".join([repr(arg) for arg in test_case.inputs])
            script_lines.append(f"    result = {test_case.function_name}({args_str})")
            
            script_lines.extend([
                "    end_time = time.perf_counter()",
                f"    results['{test_case.name}'] = {{",
                "        'output': result,",
                "        'execution_time': (end_time - start_time) * 1000,",
                "        'error': None",
                "    }",
                "except Exception as e:",
                f"    results['{test_case.name}'] = {{",
                "        'output': None,",
                "        'execution_time': 0.0,",
                "        'error': str(e)",
                "    }",
                ""
            ])
        
        script_lines.extend([
            "print(json.dumps(results))"
        ])
        
        return '\n'.join(script_lines)
    
    def _create_cpp_test_file(self, cpp_file: Path, test_suite: TestSuite) -> str:
        """Create C++ test runner"""
        # This would need to be adapted based on the generated C++ structure
        cpp_lines = [
            "#include <iostream>",
            "#include <string>",
            "#include <chrono>",
            "#include <iomanip>",
            f'#include "{cpp_file.stem}.h"',
            "",
            "int main() {",
            '    std::cout << "{" << std::endl;',
            ""
        ]
        
        for i, test_case in enumerate(test_suite.test_cases):
            if i > 0:
                cpp_lines.append('    std::cout << "," << std::endl;')
            
            cpp_lines.extend([
                f'    std::cout << "  \\"{test_case.name}\\": {{" << std::endl;',
                "    try {",
                "        auto start = std::chrono::high_resolution_clock::now();"
            ])
            
            # Generate function call (simplified)
            args_str = ", ".join([str(arg) for arg in test_case.inputs])
            cpp_lines.append(f"        auto result = {test_case.function_name}({args_str});")
            
            cpp_lines.extend([
                "        auto end = std::chrono::high_resolution_clock::now();",
                "        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);",
                '        std::cout << "    \\"output\\": " << result << "," << std::endl;',
                '        std::cout << "    \\"execution_time\\": " << duration.count() / 1000.0 << "," << std::endl;',
                '        std::cout << "    \\"error\\": null" << std::endl;',
                "    } catch (const std::exception& e) {",
                '        std::cout << "    \\"output\\": null," << std::endl;',
                '        std::cout << "    \\"execution_time\\": 0.0," << std::endl;',
                '        std::cout << "    \\"error\\": \\"" << e.what() << "\\"" << std::endl;',
                "    }",
                '    std::cout << "  }" << std::endl;'
            ])
        
        cpp_lines.extend([
            '    std::cout << "}" << std::endl;',
            "    return 0;",
            "}"
        ])
        
        return '\n'.join(cpp_lines)
    
    def _compile_cpp_tests(self, test_file: Path) -> Optional[Path]:
        """Compile C++ test file"""
        executable = test_file.with_suffix('.exe')
        
        try:
            result = subprocess.run(
                ['g++', '-std=c++17', '-O2', str(test_file), '-o', str(executable)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and executable.exists():
                return executable
        except FileNotFoundError:
            pass
        
        return None
    
    def _results_match(self, python_output, cpp_output) -> bool:
        """Compare Python and C++ outputs for equality"""
        if python_output is None or cpp_output is None:
            return False
        
        # Handle floating point comparison
        if isinstance(python_output, float) and isinstance(cpp_output, float):
            return abs(python_output - cpp_output) < 1e-6
        
        return python_output == cpp_output
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Integration function
def test_translated_code(python_file: Path, cpp_file: Path, output_dir: Path) -> str:
    """Generate and run comprehensive tests for translated code"""
    generator = TestGenerator()
    
    try:
        # Generate test suite
        test_suite = generator.generate_test_cases(python_file)
        
        # Run Python tests to get expected outputs
        python_results = generator.run_python_tests(python_file, test_suite)
        
        # Run C++ tests
        cpp_results = generator.run_cpp_tests(cpp_file, test_suite)
        
        # Compare results
        test_results = generator.compare_results(python_results, cpp_results, test_suite)
        
        # Generate reports
        report_file = output_dir / f"{python_file.stem}_test_report.md"
        report = generator.generate_test_report(test_results, report_file)
        
        # Generate Google Test suite
        gtest_file = output_dir / f"{python_file.stem}_gtest.cpp"
        generator.generate_google_test_suite(test_suite, python_results, gtest_file)
        
        return report
    
    finally:
        generator.cleanup()


if __name__ == "__main__":
    # Example usage
    python_file = Path("examples/simple_functions.py")
    cpp_file = Path("examples/cpp_output/simple_functions.cpp") 
    output_dir = Path("test_output")
    
    output_dir.mkdir(exist_ok=True)
    
    report = test_translated_code(python_file, cpp_file, output_dir)
    print("Test report generated!")
    print(report)
