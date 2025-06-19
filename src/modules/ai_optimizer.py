"""
AI-Powered Code Optimization System

Uses machine learning techniques to analyze and optimize generated C++ code.
Features:
- Performance pattern analysis
- Memory usage optimization
- Algorithmic complexity improvements
- Code style and best practices enforcement
- Automatic refactoring suggestions
"""

import os
import sys
import re
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """Represents a code optimization suggestion"""
    type: str  # 'performance', 'memory', 'style', 'algorithmic'
    priority: int  # 1-10, higher is more important
    description: str
    original_code: str
    optimized_code: str
    reasoning: str
    estimated_improvement: Optional[str] = None
    complexity_before: Optional[str] = None
    complexity_after: Optional[str] = None


@dataclass
class CodeMetrics:
    """Code quality and performance metrics"""
    lines_of_code: int
    cyclomatic_complexity: int
    nesting_depth: int
    function_count: int
    class_count: int
    loop_count: int
    memory_allocations: int
    potential_memory_leaks: int
    performance_hotspots: List[str]
    code_smells: List[str]
    maintainability_score: float  # 0-100


class PatternDatabase:
    """Database of optimization patterns and anti-patterns"""
    
    def __init__(self):
        self.performance_patterns = self._load_performance_patterns()
        self.anti_patterns = self._load_anti_patterns()
        self.best_practices = self._load_best_practices()
        self.vectorizer = None
        self.pattern_vectors = None
        
        if ML_AVAILABLE:
            self._initialize_ml_components()
    
    def _load_performance_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known performance optimization patterns"""
        return {
            "loop_unrolling": {
                "pattern": r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\d+)\s*;\s*\+\+\w+\s*\)",
                "condition": lambda m: int(m.group(1)) <= 8,
                "optimization": "unroll_small_loop",
                "improvement": "10-30% faster execution",
                "description": "Small loops can be unrolled for better performance"
            },
            "vector_operations": {
                "pattern": r"for\s*\([^}]+\)\s*{\s*\w+\[\w+\]\s*[+\-*/]=\s*\w+\[\w+\]\s*[+\-*/]\s*\w+\[\w+\]\s*;",
                "optimization": "vectorize_operation",
                "improvement": "2-4x faster with SIMD",
                "description": "Array operations can be vectorized"
            },
            "string_concatenation": {
                "pattern": r"std::string\s+\w+\s*;\s*(\w+\s*\+=\s*[^;]+;\s*){3,}",
                "optimization": "use_string_builder",
                "improvement": "Reduces memory allocations",
                "description": "Multiple string concatenations should use stringstream or reserve"
            },
            "unnecessary_copying": {
                "pattern": r"(\w+)\s*=\s*(\w+);\s*//.*copy",
                "optimization": "use_references",
                "improvement": "Eliminates unnecessary copies",
                "description": "Use references or move semantics instead of copying"
            },
            "inefficient_container_lookup": {
                "pattern": r"std::map<[^>]+>\s+\w+.*\w+\.find\([^)]+\)\s*!=\s*\w+\.end\(\)",
                "optimization": "use_unordered_map",
                "improvement": "O(1) vs O(log n) lookup",
                "description": "Consider using unordered_map for frequent lookups"
            }
        }
    
    def _load_anti_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known anti-patterns to avoid"""
        return {
            "raw_pointers": {
                "pattern": r"(\w+\s*\*\s*\w+\s*=\s*new\s+\w+|delete\s+\w+)",
                "severity": "high",
                "description": "Raw pointers should be replaced with smart pointers",
                "solution": "Use std::unique_ptr or std::shared_ptr"
            },
            "c_style_casts": {
                "pattern": r"\(\s*\w+\s*\)\s*\w+",
                "severity": "medium",
                "description": "C-style casts should be replaced with C++ casts",
                "solution": "Use static_cast, dynamic_cast, or const_cast"
            },
            "magic_numbers": {
                "pattern": r"\b\d{2,}\b",
                "severity": "low",
                "description": "Magic numbers should be replaced with named constants",
                "solution": "Define const or constexpr variables"
            },
            "global_variables": {
                "pattern": r"^(?!.*static).*\b\w+\s+\w+\s*=",
                "severity": "medium",
                "description": "Global variables should be avoided",
                "solution": "Use dependency injection or singleton pattern"
            }
        }
    
    def _load_best_practices(self) -> Dict[str, Dict[str, Any]]:
        """Load C++ best practices"""
        return {
            "const_correctness": {
                "check": "missing_const",
                "description": "Add const qualifiers where appropriate",
                "benefit": "Improves code safety and enables optimizations"
            },
            "move_semantics": {
                "check": "missing_move",
                "description": "Use move semantics for expensive-to-copy objects",
                "benefit": "Reduces unnecessary copying and improves performance"
            },
            "range_based_loops": {
                "check": "old_style_loop",
                "description": "Use range-based for loops when possible",
                "benefit": "More readable and less error-prone"
            },
            "auto_keyword": {
                "check": "verbose_types",
                "description": "Use auto for complex type declarations",
                "benefit": "Improves maintainability and readability"
            }
        }
    
    def _initialize_ml_components(self):
        """Initialize machine learning components for pattern analysis"""
        try:
            # Create feature vectors from patterns
            pattern_texts = []
            for category in [self.performance_patterns, self.anti_patterns, self.best_practices]:
                for pattern_name, pattern_data in category.items():
                    if isinstance(pattern_data, dict) and 'description' in pattern_data:
                        pattern_texts.append(pattern_data['description'])
            
            if pattern_texts:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.pattern_vectors = self.vectorizer.fit_transform(pattern_texts)
                logger.info("ML components initialized for pattern matching")
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")


class AlgorithmicComplexityAnalyzer:
    """Analyzes and suggests improvements for algorithmic complexity"""
    
    def __init__(self):
        self.complexity_patterns = {
            # Nested loops
            r"for\s*\([^}]+\)\s*{\s*[^}]*for\s*\([^}]+\)\s*{": "O(n²)",
            r"for\s*\([^}]+\)\s*{\s*[^}]*for\s*\([^}]+\)\s*{\s*[^}]*for\s*\([^}]+\)\s*{": "O(n³)",
            
            # Map/set operations in loops
            r"for\s*\([^}]+\)\s*{\s*[^}]*\.find\s*\(": "O(n log n)",
            r"for\s*\([^}]+\)\s*{\s*[^}]*\.insert\s*\(": "O(n log n)",
            
            # Linear search patterns
            r"std::find\s*\([^}]+\)": "O(n)",
            r"for\s*\([^}]+\)\s*{\s*[^}]*if\s*\([^}]+==": "O(n)",
            
            # Sorting patterns
            r"std::sort\s*\([^}]+\)": "O(n log n)",
            r"std::stable_sort\s*\([^}]+\)": "O(n log n)",
        }
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze the algorithmic complexity of code"""
        complexities = []
        
        for pattern, complexity in self.complexity_patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                complexities.append({
                    "pattern": pattern,
                    "complexity": complexity,
                    "location": match.span(),
                    "code_snippet": match.group()
                })
        
        # Determine overall complexity
        if any("O(n³)" in c["complexity"] for c in complexities):
            overall = "O(n³)"
        elif any("O(n²)" in c["complexity"] for c in complexities):
            overall = "O(n²)"
        elif any("O(n log n)" in c["complexity"] for c in complexities):
            overall = "O(n log n)"
        else:
            overall = "O(n)"
        
        return {
            "overall_complexity": overall,
            "detailed_analysis": complexities,
            "suggestions": self._generate_complexity_suggestions(complexities)
        }
    
    def _generate_complexity_suggestions(self, complexities: List[Dict[str, Any]]) -> List[OptimizationSuggestion]:
        """Generate suggestions to improve algorithmic complexity"""
        suggestions = []
        
        for comp in complexities:
            if "O(n²)" in comp["complexity"] or "O(n³)" in comp["complexity"]:
                suggestions.append(OptimizationSuggestion(
                    type="algorithmic",
                    priority=9,
                    description="High complexity nested loop detected",
                    original_code=comp["code_snippet"],
                    optimized_code="// Consider using algorithms or data structures to reduce complexity",
                    reasoning="Nested loops often indicate inefficient algorithms",
                    complexity_before=comp["complexity"],
                    complexity_after="O(n) or O(n log n) with better algorithm"
                ))
        
        return suggestions


class MemoryOptimizer:
    """Analyzes and optimizes memory usage patterns"""
    
    def __init__(self):
        self.memory_patterns = {
            "unnecessary_allocation": r"new\s+\w+\[.*\]",
            "potential_leak": r"new\s+\w+.*(?!.*delete)",
            "large_stack_allocation": r"\w+\s+\w+\[\s*\d{4,}\s*\]",
            "copy_constructor_call": r"\w+\s+\w+\s*=\s*\w+\s*\(",
            "unnecessary_temporary": r"return\s+\w+\s*\([^)]*\)\s*\+",
        }
    
    def analyze_memory_usage(self, code: str) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        issues = []
        
        for pattern_name, pattern in self.memory_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                issues.append({
                    "type": pattern_name,
                    "location": match.span(),
                    "code": match.group(),
                    "severity": self._get_severity(pattern_name)
                })
        
        return {
            "total_issues": len(issues),
            "issues_by_type": Counter(issue["type"] for issue in issues),
            "detailed_issues": issues,
            "suggestions": self._generate_memory_suggestions(issues)
        }
    
    def _get_severity(self, pattern_name: str) -> str:
        severity_map = {
            "potential_leak": "critical",
            "unnecessary_allocation": "high",
            "large_stack_allocation": "medium",
            "copy_constructor_call": "low",
            "unnecessary_temporary": "medium"
        }
        return severity_map.get(pattern_name, "low")
    
    def _generate_memory_suggestions(self, issues: List[Dict[str, Any]]) -> List[OptimizationSuggestion]:
        """Generate memory optimization suggestions"""
        suggestions = []
        
        for issue in issues:
            if issue["type"] == "potential_leak":
                suggestions.append(OptimizationSuggestion(
                    type="memory",
                    priority=10,
                    description="Potential memory leak detected",
                    original_code=issue["code"],
                    optimized_code="Use smart pointers (std::unique_ptr or std::shared_ptr)",
                    reasoning="Raw pointers without matching delete can cause memory leaks",
                    estimated_improvement="Eliminates memory leaks"
                ))
            elif issue["type"] == "unnecessary_allocation":
                suggestions.append(OptimizationSuggestion(
                    type="memory",
                    priority=7,
                    description="Unnecessary dynamic allocation",
                    original_code=issue["code"],
                    optimized_code="Use stack allocation or containers like std::vector",
                    reasoning="Dynamic allocation is slower and more error-prone",
                    estimated_improvement="Faster allocation and automatic cleanup"
                ))
        
        return suggestions


class PerformanceProfiler:
    """Profiles code for performance hotspots"""
    
    def __init__(self):
        self.hotspot_patterns = {
            "expensive_string_ops": r"std::string.*\+.*std::string",
            "frequent_allocation": r"new\s+\w+.*(?=.*for.*{)",
            "inefficient_io": r"std::cout\s*<<.*(?=.*for.*{)",
            "virtual_calls_in_loop": r"for.*{.*\w+\->\w+\(",
            "exception_in_loop": r"for.*{.*try.*catch",
        }
    
    def profile_code(self, code: str) -> Dict[str, Any]:
        """Profile code for performance issues"""
        hotspots = []
        
        for pattern_name, pattern in self.hotspot_patterns.items():
            matches = re.finditer(pattern, code, re.DOTALL)
            for match in matches:
                hotspots.append({
                    "type": pattern_name,
                    "location": match.span(),
                    "code": match.group(),
                    "impact": self._get_performance_impact(pattern_name)
                })
        
        return {
            "hotspot_count": len(hotspots),
            "hotspots": hotspots,
            "performance_score": self._calculate_performance_score(hotspots),
            "suggestions": self._generate_performance_suggestions(hotspots)
        }
    
    def _get_performance_impact(self, pattern_name: str) -> str:
        impact_map = {
            "expensive_string_ops": "high",
            "frequent_allocation": "critical",
            "inefficient_io": "medium",
            "virtual_calls_in_loop": "medium",
            "exception_in_loop": "high"
        }
        return impact_map.get(pattern_name, "low")
    
    def _calculate_performance_score(self, hotspots: List[Dict[str, Any]]) -> int:
        """Calculate overall performance score (0-100)"""
        if not hotspots:
            return 100
        
        penalty = 0
        for hotspot in hotspots:
            impact = hotspot["impact"]
            if impact == "critical":
                penalty += 25
            elif impact == "high":
                penalty += 15
            elif impact == "medium":
                penalty += 10
            else:
                penalty += 5
        
        return max(0, 100 - penalty)
    
    def _generate_performance_suggestions(self, hotspots: List[Dict[str, Any]]) -> List[OptimizationSuggestion]:
        """Generate performance optimization suggestions"""
        suggestions = []
        
        for hotspot in hotspots:
            if hotspot["type"] == "frequent_allocation":
                suggestions.append(OptimizationSuggestion(
                    type="performance",
                    priority=9,
                    description="Frequent allocation in loop",
                    original_code=hotspot["code"],
                    optimized_code="Pre-allocate outside loop or use object pool",
                    reasoning="Memory allocation in loops causes performance degradation",
                    estimated_improvement="10-50% faster execution"
                ))
            elif hotspot["type"] == "expensive_string_ops":
                suggestions.append(OptimizationSuggestion(
                    type="performance",
                    priority=6,
                    description="Expensive string concatenation",
                    original_code=hotspot["code"],
                    optimized_code="Use std::stringstream or reserve capacity",
                    reasoning="String concatenation creates temporary objects",
                    estimated_improvement="2-5x faster for multiple concatenations"
                ))
        
        return suggestions


class AICodeOptimizer:
    """Main AI-powered code optimization system"""
    
    def __init__(self):
        self.pattern_db = PatternDatabase()
        self.complexity_analyzer = AlgorithmicComplexityAnalyzer()
        self.memory_optimizer = MemoryOptimizer()
        self.performance_profiler = PerformanceProfiler()
        self.optimization_history = []
    
    def analyze_code(self, code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive code analysis"""
        logger.info(f"Analyzing code: {file_path or 'inline'}")
        
        # Basic metrics
        metrics = self._calculate_metrics(code)
        
        # Complexity analysis
        complexity_analysis = self.complexity_analyzer.analyze_complexity(code)
        
        # Memory analysis
        memory_analysis = self.memory_optimizer.analyze_memory_usage(code)
        
        # Performance analysis
        performance_analysis = self.performance_profiler.profile_code(code)
        
        # Pattern matching
        pattern_suggestions = self._match_patterns(code)
        
        # Combine all suggestions
        all_suggestions = []
        all_suggestions.extend(complexity_analysis.get("suggestions", []))
        all_suggestions.extend(memory_analysis.get("suggestions", []))
        all_suggestions.extend(performance_analysis.get("suggestions", []))
        all_suggestions.extend(pattern_suggestions)
        
        # Sort by priority
        all_suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return {
            "file_path": file_path,
            "metrics": metrics,
            "complexity": complexity_analysis,
            "memory": memory_analysis,
            "performance": performance_analysis,
            "suggestions": all_suggestions,
            "overall_score": self._calculate_overall_score(metrics, performance_analysis),
            "timestamp": str(Path().cwd() / "analysis_results" / f"analysis_{hash(code)}.json")
        }
    
    def optimize_code(self, code: str, apply_suggestions: bool = False) -> Dict[str, Any]:
        """Optimize code based on analysis"""
        analysis = self.analyze_code(code)
        suggestions = analysis["suggestions"]
        
        if not apply_suggestions:
            return {
                "original_code": code,
                "analysis": analysis,
                "optimized_code": None,
                "applied_suggestions": []
            }
        
        # Apply suggestions in priority order
        optimized_code = code
        applied_suggestions = []
        
        for suggestion in suggestions[:10]:  # Limit to top 10 suggestions
            try:
                if self._can_apply_suggestion(suggestion, optimized_code):
                    optimized_code = self._apply_suggestion(suggestion, optimized_code)
                    applied_suggestions.append(suggestion)
            except Exception as e:
                logger.warning(f"Failed to apply suggestion: {e}")
        
        return {
            "original_code": code,
            "optimized_code": optimized_code,
            "analysis": analysis,
            "applied_suggestions": applied_suggestions,
            "improvement_summary": self._generate_improvement_summary(applied_suggestions)
        }
    
    def batch_optimize(self, file_paths: List[str]) -> Dict[str, Any]:
        """Optimize multiple files"""
        results = {}
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                result = self.optimize_code(code, apply_suggestions=True)
                results[file_path] = result
                
                # Save optimized version
                if result["optimized_code"]:
                    optimized_path = file_path.replace('.cpp', '_optimized.cpp')
                    with open(optimized_path, 'w', encoding='utf-8') as f:
                        f.write(result["optimized_code"])
                    
                    logger.info(f"Optimized {file_path} -> {optimized_path}")
                
            except Exception as e:
                logger.error(f"Failed to optimize {file_path}: {e}")
                results[file_path] = {"error": str(e)}
        
        return results
    
    def _calculate_metrics(self, code: str) -> CodeMetrics:
        """Calculate basic code metrics"""
        lines = code.splitlines()
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        
        # Count various elements
        function_count = len(re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*{', code))
        class_count = len(re.findall(r'\bclass\s+\w+', code))
        loop_count = len(re.findall(r'\b(for|while)\s*\(', code))
        
        # Estimate complexity (simplified)
        cyclomatic_complexity = 1 + len(re.findall(r'\b(if|else|while|for|switch|case|catch)\b', code))
        
        # Nesting depth (simplified)
        max_depth = 0
        current_depth = 0
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        # Memory allocations
        memory_allocations = len(re.findall(r'\bnew\b', code))
        potential_memory_leaks = len(re.findall(r'\bnew\b', code)) - len(re.findall(r'\bdelete\b', code))
        
        # Performance hotspots and code smells (simplified)
        performance_hotspots = []
        if re.search(r'for.*for', code):
            performance_hotspots.append("nested loops")
        if re.search(r'std::string.*\+', code):
            performance_hotspots.append("string concatenation")
        
        code_smells = []
        if len(lines) > 1000:
            code_smells.append("large file")
        if cyclomatic_complexity > 50:
            code_smells.append("high complexity")
        
        # Maintainability score (simplified)
        maintainability_score = max(0, 100 - cyclomatic_complexity - max_depth * 5 - len(code_smells) * 10)
        
        return CodeMetrics(
            lines_of_code=lines_of_code,
            cyclomatic_complexity=cyclomatic_complexity,
            nesting_depth=max_depth,
            function_count=function_count,
            class_count=class_count,
            loop_count=loop_count,
            memory_allocations=memory_allocations,
            potential_memory_leaks=max(0, potential_memory_leaks),
            performance_hotspots=performance_hotspots,
            code_smells=code_smells,
            maintainability_score=maintainability_score
        )
    
    def _match_patterns(self, code: str) -> List[OptimizationSuggestion]:
        """Match code against known patterns"""
        suggestions = []
        
        # Check performance patterns
        for pattern_name, pattern_data in self.pattern_db.performance_patterns.items():
            pattern = pattern_data["pattern"]
            matches = re.finditer(pattern, code)
            
            for match in matches:
                # Check condition if exists
                if "condition" in pattern_data:
                    if not pattern_data["condition"](match):
                        continue
                
                suggestions.append(OptimizationSuggestion(
                    type="performance",
                    priority=7,
                    description=pattern_data["description"],
                    original_code=match.group(),
                    optimized_code=f"// Apply {pattern_data['optimization']}",
                    reasoning=pattern_data["description"],
                    estimated_improvement=pattern_data.get("improvement", "Performance gain")
                ))
        
        # Check anti-patterns
        for pattern_name, pattern_data in self.pattern_db.anti_patterns.items():
            pattern = pattern_data["pattern"]
            matches = re.finditer(pattern, code)
            
            for match in matches:
                priority = {"high": 8, "medium": 5, "low": 3}.get(pattern_data["severity"], 3)
                
                suggestions.append(OptimizationSuggestion(
                    type="style",
                    priority=priority,
                    description=pattern_data["description"],
                    original_code=match.group(),
                    optimized_code=pattern_data["solution"],
                    reasoning=pattern_data["description"]
                ))
        
        return suggestions
    
    def _can_apply_suggestion(self, suggestion: OptimizationSuggestion, code: str) -> bool:
        """Check if a suggestion can be safely applied"""
        # Simple heuristic - check if the original code still exists
        return suggestion.original_code in code
    
    def _apply_suggestion(self, suggestion: OptimizationSuggestion, code: str) -> str:
        """Apply a suggestion to the code"""
        # Simple replacement for now - could be more sophisticated
        if suggestion.type == "style" and "smart pointer" in suggestion.optimized_code.lower():
            # Replace new/delete with smart pointers
            return re.sub(
                r'(\w+\s*\*\s*\w+\s*=\s*)new\s+(\w+)',
                r'\1std::make_unique<\2>',
                code
            )
        elif suggestion.type == "performance" and "stringstream" in suggestion.optimized_code.lower():
            # Replace string concatenation
            return re.sub(
                r'std::string\s+(\w+)\s*;\s*((\1\s*\+=\s*[^;]+;\s*)+)',
                r'std::stringstream \1_ss;\n// Add to stringstream and convert: std::string \1 = \1_ss.str();',
                code
            )
        else:
            # Add comment with suggestion
            return code.replace(
                suggestion.original_code,
                f"// TODO: {suggestion.description}\n{suggestion.original_code}"
            )
    
    def _calculate_overall_score(self, metrics: CodeMetrics, performance_analysis: Dict[str, Any]) -> int:
        """Calculate overall code quality score"""
        base_score = metrics.maintainability_score
        performance_score = performance_analysis.get("performance_score", 100)
        
        # Weight the scores
        overall = (base_score * 0.6 + performance_score * 0.4)
        return int(overall)
    
    def _generate_improvement_summary(self, applied_suggestions: List[OptimizationSuggestion]) -> Dict[str, Any]:
        """Generate a summary of improvements"""
        if not applied_suggestions:
            return {"message": "No optimizations applied"}
        
        improvements_by_type = defaultdict(list)
        for suggestion in applied_suggestions:
            improvements_by_type[suggestion.type].append(suggestion)
        
        return {
            "total_optimizations": len(applied_suggestions),
            "by_type": {k: len(v) for k, v in improvements_by_type.items()},
            "estimated_improvements": [s.estimated_improvement for s in applied_suggestions if s.estimated_improvement],
            "priority_distribution": Counter(s.priority for s in applied_suggestions)
        }
    
    def save_analysis_report(self, analysis: Dict[str, Any], output_path: str):
        """Save detailed analysis report"""
        # Convert suggestions to dictionaries for JSON serialization
        analysis_copy = analysis.copy()
        analysis_copy["suggestions"] = [asdict(s) for s in analysis["suggestions"]]
        analysis_copy["metrics"] = asdict(analysis["metrics"])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_copy, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_path}")


def main():
    """Demo the AI code optimization system"""
    print("AI-Powered Code Optimization System Demo")
    print("=" * 50)
    
    # Sample C++ code to optimize
    sample_code = '''
#include <iostream>
#include <string>
#include <vector>

class DataProcessor {
public:
    std::string processData(std::vector<int> data) {
        std::string result;
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.size(); j++) {
                if (data[i] > data[j]) {
                    std::string temp = "Item " + std::to_string(i) + " > " + std::to_string(j);
                    result += temp + "\\n";
                }
            }
        }
        
        int* buffer = new int[1000];
        // Process buffer...
        // Missing delete[] buffer; - memory leak!
        
        return result;
    }
};
'''
    
    # Initialize optimizer
    optimizer = AICodeOptimizer()
    
    # Analyze the code
    print("Analyzing sample code...")
    analysis = optimizer.analyze_code(sample_code, "sample.cpp")
    
    print(f"\\nCode Metrics:")
    print(f"  Lines of Code: {analysis['metrics'].lines_of_code}")
    print(f"  Cyclomatic Complexity: {analysis['metrics'].cyclomatic_complexity}")
    print(f"  Function Count: {analysis['metrics'].function_count}")
    print(f"  Performance Score: {analysis['performance']['performance_score']}")
    print(f"  Overall Score: {analysis['overall_score']}")
    
    print(f"\\nFound {len(analysis['suggestions'])} optimization suggestions:")
    for i, suggestion in enumerate(analysis['suggestions'][:5], 1):
        print(f"\\n{i}. {suggestion.description} (Priority: {suggestion.priority})")
        print(f"   Type: {suggestion.type}")
        print(f"   Reasoning: {suggestion.reasoning}")
        if suggestion.estimated_improvement:
            print(f"   Improvement: {suggestion.estimated_improvement}")
    
    # Demonstrate optimization
    print("\\n" + "=" * 50)
    print("Applying optimizations...")
    
    optimization_result = optimizer.optimize_code(sample_code, apply_suggestions=True)
    
    if optimization_result["applied_suggestions"]:
        print(f"Applied {len(optimization_result['applied_suggestions'])} optimizations")
        print("\\nImprovement Summary:")
        summary = optimization_result["improvement_summary"]
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save results
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save optimized code
        with open(output_dir / "optimized_sample.cpp", 'w') as f:
            f.write(optimization_result["optimized_code"])
        
        # Save analysis report
        optimizer.save_analysis_report(analysis, str(output_dir / "analysis_report.json"))
        
        print(f"\\nResults saved to {output_dir}/")
    else:
        print("No optimizations were applied")
    
    print("\\nAI Code Optimization Demo completed!")


if __name__ == "__main__":
    main()
