"""
Performance Benchmarking System for Python-to-C++ Translation
Measures and compares performance between Python and translated C++ code
"""

import time
import subprocess
import statistics
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    language: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    iterations: int
    std_deviation: float
    min_time: float
    max_time: float


@dataclass
class ComparisonReport:
    """Performance comparison between Python and C++"""
    python_result: BenchmarkResult
    cpp_result: BenchmarkResult
    speedup_factor: float
    memory_improvement: float
    overall_score: float


class PerformanceBenchmarker:
    """Benchmarks performance of Python vs translated C++ code"""
    
    def __init__(self, iterations: int = 10):
        self.iterations = iterations
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def benchmark_translation(self, python_file: Path, cpp_file: Path, 
                            test_data: Optional[str] = None) -> ComparisonReport:
        """Benchmark Python vs C++ performance"""
        
        # Benchmark Python
        python_result = self._benchmark_python(python_file, test_data)
        
        # Compile and benchmark C++
        cpp_result = self._benchmark_cpp(cpp_file, test_data)
        
        # Calculate comparison metrics
        speedup = python_result.execution_time_ms / cpp_result.execution_time_ms
        memory_improvement = (python_result.memory_usage_mb - cpp_result.memory_usage_mb) / python_result.memory_usage_mb
        overall_score = (speedup * 0.7) + (memory_improvement * 0.3)
        
        return ComparisonReport(
            python_result=python_result,
            cpp_result=cpp_result,
            speedup_factor=speedup,
            memory_improvement=memory_improvement,
            overall_score=overall_score
        )
    
    def _benchmark_python(self, python_file: Path, test_data: Optional[str]) -> BenchmarkResult:
        """Benchmark Python code execution"""
        times = []
        
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            
            # Execute Python code
            cmd = ["python", str(python_file)]
            if test_data:
                cmd.extend(test_data.split())
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(execution_time)
        
        return BenchmarkResult(
            language="Python",
            execution_time_ms=statistics.mean(times),
            memory_usage_mb=self._estimate_memory_usage("python", python_file),
            cpu_usage_percent=0.0,  # Would need psutil for accurate measurement
            iterations=self.iterations,
            std_deviation=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times)
        )
    
    def _benchmark_cpp(self, cpp_file: Path, test_data: Optional[str]) -> BenchmarkResult:
        """Benchmark C++ code execution"""
        # Compile C++ code
        executable = self._compile_cpp(cpp_file)
        
        if not executable:
            return BenchmarkResult(
                language="C++",
                execution_time_ms=float('inf'),
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                iterations=0,
                std_deviation=0.0,
                min_time=0.0,
                max_time=0.0
            )
        
        times = []
        
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            
            # Execute C++ code
            cmd = [str(executable)]
            if test_data:
                cmd.extend(test_data.split())
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(execution_time)
        
        return BenchmarkResult(
            language="C++",
            execution_time_ms=statistics.mean(times),
            memory_usage_mb=self._estimate_memory_usage("cpp", executable),
            cpu_usage_percent=0.0,
            iterations=self.iterations,
            std_deviation=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times)
        )
    
    def _compile_cpp(self, cpp_file: Path) -> Optional[Path]:
        """Compile C++ file and return executable path"""
        executable = self.temp_dir / f"{cpp_file.stem}_bench"
        
        # Try different compilation approaches
        compile_commands = [
            ["g++", "-std=c++17", "-O3", "-DNDEBUG", str(cpp_file), "-o", str(executable)],
            ["clang++", "-std=c++17", "-O3", "-DNDEBUG", str(cpp_file), "-o", str(executable)],
            ["cl", "/std:c++17", "/O2", "/DNDEBUG", str(cpp_file), f"/Fe:{executable}.exe"]
        ]
        
        for cmd in compile_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Check if executable exists
                    if executable.exists():
                        return executable
                    elif Path(f"{executable}.exe").exists():
                        return Path(f"{executable}.exe")
            except FileNotFoundError:
                continue
        
        print(f"Failed to compile {cpp_file}")
        return None
    
    def _estimate_memory_usage(self, language: str, file_path: Path) -> float:
        """Estimate memory usage (simplified approach)"""
        # This is a simplified estimation
        # In practice, you'd use tools like valgrind, /usr/bin/time, or psutil
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if language == "python":
            return file_size_mb * 5  # Python typically uses more memory
        else:
            return file_size_mb * 2  # C++ is more memory efficient
    
    def generate_benchmark_report(self, comparison: ComparisonReport) -> str:
        """Generate a detailed benchmark report"""
        lines = [
            "# Performance Benchmark Report",
            "",
            "## Summary",
            f"- **Speedup Factor**: {comparison.speedup_factor:.2f}x",
            f"- **Memory Improvement**: {comparison.memory_improvement*100:.1f}%",
            f"- **Overall Score**: {comparison.overall_score:.2f}",
            "",
            "## Detailed Results",
            "",
            "### Python Performance",
            f"- **Average Execution Time**: {comparison.python_result.execution_time_ms:.2f} ms",
            f"- **Memory Usage**: {comparison.python_result.memory_usage_mb:.2f} MB",
            f"- **Standard Deviation**: {comparison.python_result.std_deviation:.2f} ms",
            f"- **Min/Max Time**: {comparison.python_result.min_time:.2f} / {comparison.python_result.max_time:.2f} ms",
            "",
            "### C++ Performance",
            f"- **Average Execution Time**: {comparison.cpp_result.execution_time_ms:.2f} ms",
            f"- **Memory Usage**: {comparison.cpp_result.memory_usage_mb:.2f} MB",
            f"- **Standard Deviation**: {comparison.cpp_result.std_deviation:.2f} ms",
            f"- **Min/Max Time**: {comparison.cpp_result.min_time:.2f} / {comparison.cpp_result.max_time:.2f} ms",
            "",
            "## Analysis",
            ""
        ]
        
        # Performance analysis
        if comparison.speedup_factor > 2.0:
            lines.append("🚀 **Excellent speedup!** C++ version is significantly faster.")
        elif comparison.speedup_factor > 1.2:
            lines.append("✅ **Good speedup.** C++ version shows measurable improvement.")
        elif comparison.speedup_factor > 0.8:
            lines.append("⚡ **Comparable performance.** Similar execution times.")
        else:
            lines.append("⚠️ **C++ version is slower.** Consider optimization or review translation.")
        
        lines.append("")
        
        # Memory analysis
        if comparison.memory_improvement > 0.2:
            lines.append("💾 **Significant memory savings** with C++ version.")
        elif comparison.memory_improvement > 0:
            lines.append("💾 **Some memory savings** with C++ version.")
        else:
            lines.append("💾 **C++ version uses more memory.** Review memory management.")
        
        lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        # Recommendations based on results
        if comparison.speedup_factor < 1.5:
            lines.extend([
                "- Consider enabling compiler optimizations (-O3, -march=native)",
                "- Review generated C++ code for performance bottlenecks",
                "- Check if expensive operations can be optimized"
            ])
        
        if comparison.memory_improvement < 0:
            lines.extend([
                "- Review memory allocation patterns",
                "- Consider using more efficient data structures",
                "- Check for memory leaks in generated code"
            ])
        
        if comparison.cpp_result.std_deviation > comparison.python_result.std_deviation * 2:
            lines.extend([
                "- C++ version shows high variance - investigate consistency",
                "- Consider warm-up runs to improve measurement accuracy"
            ])
        
        return '\n'.join(lines)
    
    def run_comprehensive_benchmark(self, python_file: Path, cpp_file: Path, 
                                  test_cases: List[str] = None) -> Dict[str, ComparisonReport]:
        """Run benchmarks with multiple test cases"""
        results = {}
        
        test_cases = test_cases or [""]  # Default to no arguments
        
        for i, test_case in enumerate(test_cases):
            test_name = f"test_case_{i}" if test_case else "default"
            results[test_name] = self.benchmark_translation(python_file, cpp_file, test_case)
        
        return results
    
    def save_benchmark_results(self, results: Dict[str, ComparisonReport], output_file: Path):
        """Save benchmark results to JSON file"""
        serializable_results = {}
        
        for test_name, comparison in results.items():
            serializable_results[test_name] = {
                'python': {
                    'execution_time_ms': comparison.python_result.execution_time_ms,
                    'memory_usage_mb': comparison.python_result.memory_usage_mb,
                    'std_deviation': comparison.python_result.std_deviation,
                    'iterations': comparison.python_result.iterations
                },
                'cpp': {
                    'execution_time_ms': comparison.cpp_result.execution_time_ms,
                    'memory_usage_mb': comparison.cpp_result.memory_usage_mb,
                    'std_deviation': comparison.cpp_result.std_deviation,
                    'iterations': comparison.cpp_result.iterations
                },
                'metrics': {
                    'speedup_factor': comparison.speedup_factor,
                    'memory_improvement': comparison.memory_improvement,
                    'overall_score': comparison.overall_score
                }
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class BenchmarkSuite:
    """Collection of standard benchmarks for testing translation quality"""
    
    @staticmethod
    def create_algorithm_benchmarks() -> List[Tuple[str, str]]:
        """Create algorithm-focused benchmark cases"""
        benchmarks = [
            ("Fibonacci", """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

if __name__ == "__main__":
    result = fibonacci(30)
    print(result)
"""),
            ("Bubble Sort", """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

if __name__ == "__main__":
    data = list(range(1000, 0, -1))
    result = bubble_sort(data)
    print(len(result))
"""),
            ("Matrix Multiplication", """
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    for _ in range(1000):
        result = matrix_multiply(A, B)
    print(result[0][0])
""")
        ]
        
        return benchmarks
    
    @staticmethod
    def save_benchmark_cases(output_dir: Path):
        """Save benchmark cases to files"""
        output_dir.mkdir(exist_ok=True)
        
        for name, code in BenchmarkSuite.create_algorithm_benchmarks():
            filename = name.lower().replace(" ", "_") + "_benchmark.py"
            with open(output_dir / filename, 'w') as f:
                f.write(code)


# Example usage integration
def benchmark_translated_code(python_file: Path, cpp_file: Path) -> str:
    """Benchmark a translated Python file against its C++ equivalent"""
    benchmarker = PerformanceBenchmarker(iterations=5)
    
    try:
        comparison = benchmarker.benchmark_translation(python_file, cpp_file)
        report = benchmarker.generate_benchmark_report(comparison)
        return report
    finally:
        benchmarker.cleanup()


if __name__ == "__main__":
    # Create sample benchmark cases
    BenchmarkSuite.save_benchmark_cases(Path("benchmark_cases"))
