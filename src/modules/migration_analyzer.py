"""
Advanced Code Migration Analyzer for Python-to-C++ Translation
Provides detailed migration strategies, risk assessment, and optimization recommendations
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import importlib.util


class MigrationComplexity(Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MigrationStrategy(Enum):
    DIRECT_TRANSLATION = "direct_translation"
    REFACTOR_THEN_TRANSLATE = "refactor_then_translate"
    HYBRID_APPROACH = "hybrid_approach"
    MANUAL_REWRITE = "manual_rewrite"
    KEEP_PYTHON = "keep_python"


@dataclass
class CodePattern:
    """Represents a detected code pattern"""
    pattern_type: str
    description: str
    complexity: MigrationComplexity
    risk_level: RiskLevel
    line_numbers: List[int]
    code_snippets: List[str]
    migration_notes: str
    suggested_approach: str


@dataclass
class DependencyAnalysis:
    """Analysis of external dependencies"""
    name: str
    version: Optional[str]
    usage_patterns: List[str]
    cpp_alternatives: List[str]
    migration_effort: MigrationComplexity
    compatibility_notes: str


@dataclass
class FunctionAnalysis:
    """Detailed analysis of a function"""
    name: str
    line_number: int
    complexity_score: int
    parameter_count: int
    return_complexity: str
    uses_globals: bool
    side_effects: List[str]
    migration_complexity: MigrationComplexity
    suggested_changes: List[str]


@dataclass
class ClassAnalysis:
    """Detailed analysis of a class"""
    name: str
    line_number: int
    inheritance_depth: int
    method_count: int
    uses_metaclass: bool
    dynamic_attributes: bool
    special_methods: List[str]
    migration_complexity: MigrationComplexity
    suggested_strategy: str


@dataclass
class MigrationReport:
    """Comprehensive migration analysis report"""
    source_file: Path
    total_lines: int
    overall_complexity: MigrationComplexity
    estimated_effort_hours: Tuple[int, int]  # (min, max)
    recommended_strategy: MigrationStrategy
    risk_assessment: RiskLevel
    
    # Detailed analysis
    functions: List[FunctionAnalysis] = field(default_factory=list)
    classes: List[ClassAnalysis] = field(default_factory=list)
    dependencies: List[DependencyAnalysis] = field(default_factory=list)
    patterns: List[CodePattern] = field(default_factory=list)
    
    # Recommendations
    preparation_steps: List[str] = field(default_factory=list)
    migration_phases: List[Dict[str, Any]] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)


class AdvancedMigrationAnalyzer:
    """Advanced analyzer for Python-to-C++ migration planning"""
    
    def __init__(self):
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.complexity_weights = self._initialize_complexity_weights()
        self.dependency_mappings = self._load_dependency_mappings()
        
    def analyze_migration(self, python_file: Path) -> MigrationReport:
        """Perform comprehensive migration analysis"""
        
        with open(python_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return self._create_error_report(python_file, f"Syntax error: {e}")
        
        # Initialize report
        report = MigrationReport(
            source_file=python_file,
            total_lines=len(source_code.split('\n')),
            overall_complexity=MigrationComplexity.SIMPLE,
            estimated_effort_hours=(1, 3),
            recommended_strategy=MigrationStrategy.DIRECT_TRANSLATION,
            risk_assessment=RiskLevel.LOW
        )
        
        # Perform detailed analysis
        report.functions = self._analyze_functions(tree, source_code)
        report.classes = self._analyze_classes(tree, source_code)
        report.dependencies = self._analyze_dependencies(tree)
        report.patterns = self._detect_patterns(tree, source_code)
        
        # Calculate overall metrics
        self._calculate_overall_metrics(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def compare_migration_strategies(self, python_file: Path) -> Dict[str, Any]:
        """Compare different migration strategies for a file"""
        
        base_analysis = self.analyze_migration(python_file)
        
        strategies = {
            MigrationStrategy.DIRECT_TRANSLATION: {
                "description": "Translate Python code directly to C++",
                "effort": base_analysis.estimated_effort_hours,
                "accuracy": self._estimate_translation_accuracy(base_analysis),
                "maintenance": "Low - single codebase",
                "performance": "High - native C++",
                "risks": self._get_direct_translation_risks(base_analysis)
            },
            MigrationStrategy.REFACTOR_THEN_TRANSLATE: {
                "description": "Refactor Python code first, then translate",
                "effort": (base_analysis.estimated_effort_hours[0] * 1.5, 
                          base_analysis.estimated_effort_hours[1] * 2),
                "accuracy": min(1.0, self._estimate_translation_accuracy(base_analysis) + 0.2),
                "maintenance": "Low - single codebase",
                "performance": "High - optimized C++",
                "risks": self._get_refactor_risks(base_analysis)
            },
            MigrationStrategy.HYBRID_APPROACH: {
                "description": "Keep critical parts in Python, translate performance parts",
                "effort": (base_analysis.estimated_effort_hours[0] * 0.6,
                          base_analysis.estimated_effort_hours[1] * 1.2),
                "accuracy": 0.9,
                "maintenance": "Medium - dual codebase",
                "performance": "Medium - mixed performance",
                "risks": ["Interface complexity", "Debugging across languages"]
            },
            MigrationStrategy.MANUAL_REWRITE: {
                "description": "Rewrite from scratch in C++ using Python as reference",
                "effort": (base_analysis.estimated_effort_hours[0] * 2,
                          base_analysis.estimated_effort_hours[1] * 4),
                "accuracy": 1.0,
                "maintenance": "Low - single codebase",
                "performance": "Very High - optimized C++",
                "risks": ["High initial effort", "Potential feature gaps"]
            }
        }
        
        # Recommend best strategy
        best_strategy = self._recommend_strategy(base_analysis, strategies)
        
        return {
            "analysis": base_analysis,
            "strategies": strategies,
            "recommended": best_strategy,
            "comparison_matrix": self._create_comparison_matrix(strategies)
        }
    
    def generate_migration_roadmap(self, python_files: List[Path]) -> Dict[str, Any]:
        """Generate a comprehensive migration roadmap for multiple files"""
        
        analyses = [self.analyze_migration(f) for f in python_files]
        
        # Sort by complexity and dependencies
        sorted_analyses = self._sort_by_migration_order(analyses)
        
        roadmap = {
            "overview": {
                "total_files": len(python_files),
                "total_effort_range": self._calculate_total_effort(analyses),
                "average_complexity": self._calculate_average_complexity(analyses),
                "high_risk_files": len([a for a in analyses if a.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
            },
            "phases": self._create_migration_phases(sorted_analyses),
            "dependencies": self._analyze_cross_file_dependencies(analyses),
            "resource_requirements": self._estimate_resource_requirements(analyses),
            "timeline": self._create_migration_timeline(sorted_analyses),
            "risk_mitigation": self._create_risk_mitigation_plan(analyses)
        }
        
        return roadmap
    
    def _analyze_functions(self, tree: ast.AST, source_code: str) -> List[FunctionAnalysis]:
        """Analyze all functions in the code"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis = FunctionAnalysis(
                    name=node.name,
                    line_number=node.lineno,
                    complexity_score=self._calculate_cyclomatic_complexity(node),
                    parameter_count=len(node.args.args),
                    return_complexity=self._analyze_return_complexity(node),
                    uses_globals=self._uses_global_variables(node),
                    side_effects=self._detect_side_effects(node),
                    migration_complexity=self._assess_function_complexity(node),
                    suggested_changes=self._suggest_function_improvements(node)
                )
                functions.append(analysis)
        
        return functions
    
    def _analyze_classes(self, tree: ast.AST, source_code: str) -> List[ClassAnalysis]:
        """Analyze all classes in the code"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis = ClassAnalysis(
                    name=node.name,
                    line_number=node.lineno,
                    inheritance_depth=len(node.bases),
                    method_count=len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    uses_metaclass=self._uses_metaclass(node),
                    dynamic_attributes=self._has_dynamic_attributes(node),
                    special_methods=self._get_special_methods(node),
                    migration_complexity=self._assess_class_complexity(node),
                    suggested_strategy=self._suggest_class_strategy(node)
                )
                classes.append(analysis)
        
        return classes
    
    def _analyze_dependencies(self, tree: ast.AST) -> List[DependencyAnalysis]:
        """Analyze external dependencies"""
        dependencies = []
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        for imp in imports:
            if imp in self.dependency_mappings:
                mapping = self.dependency_mappings[imp]
                analysis = DependencyAnalysis(
                    name=imp,
                    version=None,  # Would need to detect from requirements.txt
                    usage_patterns=mapping.get("patterns", []),
                    cpp_alternatives=mapping.get("alternatives", []),
                    migration_effort=MigrationComplexity(mapping.get("complexity", "moderate")),
                    compatibility_notes=mapping.get("notes", "")
                )
                dependencies.append(analysis)
        
        return dependencies
    
    def _detect_patterns(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect various code patterns that affect migration"""
        patterns = []
        
        for detector in self.pattern_detectors:
            detected = detector(tree, source_code)
            patterns.extend(detected)
        
        return patterns
    
    def _calculate_overall_metrics(self, report: MigrationReport):
        """Calculate overall complexity and effort metrics"""
        
        # Complexity scoring
        complexity_scores = []
        
        for func in report.functions:
            complexity_scores.append(self.complexity_weights[func.migration_complexity])
        
        for cls in report.classes:
            complexity_scores.append(self.complexity_weights[cls.migration_complexity])
        
        for pattern in report.patterns:
            complexity_scores.append(self.complexity_weights[pattern.complexity])
        
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            if avg_complexity <= 1.5:
                report.overall_complexity = MigrationComplexity.SIMPLE
                report.estimated_effort_hours = (2, 8)
            elif avg_complexity <= 2.5:
                report.overall_complexity = MigrationComplexity.MODERATE
                report.estimated_effort_hours = (8, 24)
            elif avg_complexity <= 3.5:
                report.overall_complexity = MigrationComplexity.COMPLEX
                report.estimated_effort_hours = (24, 80)
            else:
                report.overall_complexity = MigrationComplexity.EXPERT
                report.estimated_effort_hours = (80, 200)
        
        # Risk assessment
        high_risk_patterns = [p for p in report.patterns if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        complex_dependencies = [d for d in report.dependencies if d.migration_effort in [MigrationComplexity.COMPLEX, MigrationComplexity.EXPERT]]
        
        if len(high_risk_patterns) > 3 or len(complex_dependencies) > 2:
            report.risk_assessment = RiskLevel.HIGH
        elif len(high_risk_patterns) > 1 or len(complex_dependencies) > 0:
            report.risk_assessment = RiskLevel.MEDIUM
        else:
            report.risk_assessment = RiskLevel.LOW
    
    def _generate_recommendations(self, report: MigrationReport):
        """Generate migration recommendations"""
        
        # Preparation steps
        report.preparation_steps = [
            "Add comprehensive type hints to all functions",
            "Add docstrings with parameter and return type documentation",
            "Write unit tests for all functions to validate translation",
            "Refactor complex functions into smaller, focused functions"
        ]
        
        # Add specific recommendations based on analysis
        if any(func.uses_globals for func in report.functions):
            report.preparation_steps.append("Eliminate global variable dependencies")
        
        if any(cls.uses_metaclass for cls in report.classes):
            report.preparation_steps.append("Simplify metaclass usage or plan manual C++ implementation")
        
        # Migration phases
        report.migration_phases = [
            {
                "phase": 1,
                "name": "Foundation",
                "description": "Set up C++ project structure and basic types",
                "effort_percentage": 20,
                "deliverables": ["Project template", "Build system", "Basic data structures"]
            },
            {
                "phase": 2,
                "name": "Core Translation",
                "description": "Translate main functionality",
                "effort_percentage": 50,
                "deliverables": ["Translated functions", "Core classes", "Main algorithms"]
            },
            {
                "phase": 3,
                "name": "Integration",
                "description": "Integrate components and handle dependencies",
                "effort_percentage": 20,
                "deliverables": ["Integrated system", "Dependency handling", "Interface compatibility"]
            },
            {
                "phase": 4,
                "name": "Optimization",
                "description": "Optimize performance and finalize",
                "effort_percentage": 10,
                "deliverables": ["Performance optimizations", "Final testing", "Documentation"]
            }
        ]
        
        # Optimization opportunities
        report.optimization_opportunities = [
            "Use const references for large objects",
            "Implement move semantics for expensive operations",
            "Use smart pointers for automatic memory management",
            "Consider using std::string_view for string parameters"
        ]
        
        if any(func.complexity_score > 10 for func in report.functions):
            report.optimization_opportunities.append("Break down complex functions for better maintainability")
        
        # Potential issues
        report.potential_issues = []
        
        if report.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            report.potential_issues.extend([
                "High complexity may lead to translation errors",
                "Manual verification of complex logic required",
                "Consider incremental migration approach"
            ])
        
        complex_patterns = [p for p in report.patterns if p.complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.EXPERT]]
        if complex_patterns:
            report.potential_issues.append(f"Complex patterns detected: {', '.join(p.pattern_type for p in complex_patterns)}")
    
    def generate_detailed_report(self, report: MigrationReport, output_file: Path) -> str:
        """Generate detailed migration report"""
        
        lines = [
            f"# Migration Analysis Report: {report.source_file.name}",
            "",
            "## Executive Summary",
            "",
            f"- **Overall Complexity**: {report.overall_complexity.value.title()}",
            f"- **Estimated Effort**: {report.estimated_effort_hours[0]}-{report.estimated_effort_hours[1]} hours",
            f"- **Risk Level**: {report.risk_assessment.value.title()}",
            f"- **Recommended Strategy**: {report.recommended_strategy.value.replace('_', ' ').title()}",
            "",
            "## Code Analysis",
            "",
            f"- **Total Lines**: {report.total_lines}",
            f"- **Functions**: {len(report.functions)}",
            f"- **Classes**: {len(report.classes)}",
            f"- **Dependencies**: {len(report.dependencies)}",
            f"- **Detected Patterns**: {len(report.patterns)}",
            ""
        ]
        
        # Function analysis
        if report.functions:
            lines.extend([
                "## Function Analysis",
                ""
            ])
            
            for func in sorted(report.functions, key=lambda f: f.complexity_score, reverse=True)[:10]:
                lines.extend([
                    f"### {func.name} (Line {func.line_number})",
                    f"- **Complexity Score**: {func.complexity_score}",
                    f"- **Parameters**: {func.parameter_count}",
                    f"- **Migration Complexity**: {func.migration_complexity.value}",
                    f"- **Uses Globals**: {'Yes' if func.uses_globals else 'No'}",
                ])
                
                if func.side_effects:
                    lines.append(f"- **Side Effects**: {', '.join(func.side_effects)}")
                
                if func.suggested_changes:
                    lines.extend([
                        "- **Suggested Changes**:",
                        *[f"  - {change}" for change in func.suggested_changes]
                    ])
                
                lines.append("")
        
        # Class analysis
        if report.classes:
            lines.extend([
                "## Class Analysis",
                ""
            ])
            
            for cls in report.classes:
                lines.extend([
                    f"### {cls.name} (Line {cls.line_number})",
                    f"- **Inheritance Depth**: {cls.inheritance_depth}",
                    f"- **Methods**: {cls.method_count}",
                    f"- **Uses Metaclass**: {'Yes' if cls.uses_metaclass else 'No'}",
                    f"- **Dynamic Attributes**: {'Yes' if cls.dynamic_attributes else 'No'}",
                    f"- **Migration Complexity**: {cls.migration_complexity.value}",
                    f"- **Suggested Strategy**: {cls.suggested_strategy}",
                ])
                
                if cls.special_methods:
                    lines.append(f"- **Special Methods**: {', '.join(cls.special_methods)}")
                
                lines.append("")
        
        # Dependencies
        if report.dependencies:
            lines.extend([
                "## Dependency Analysis",
                ""
            ])
            
            for dep in report.dependencies:
                lines.extend([
                    f"### {dep.name}",
                    f"- **Migration Effort**: {dep.migration_effort.value}",
                    f"- **C++ Alternatives**: {', '.join(dep.cpp_alternatives) if dep.cpp_alternatives else 'None identified'}",
                ])
                
                if dep.compatibility_notes:
                    lines.append(f"- **Notes**: {dep.compatibility_notes}")
                
                lines.append("")
        
        # Detected patterns
        if report.patterns:
            lines.extend([
                "## Detected Patterns",
                ""
            ])
            
            pattern_groups = {}
            for pattern in report.patterns:
                if pattern.pattern_type not in pattern_groups:
                    pattern_groups[pattern.pattern_type] = []
                pattern_groups[pattern.pattern_type].append(pattern)
            
            for pattern_type, patterns in pattern_groups.items():
                lines.extend([
                    f"### {pattern_type.replace('_', ' ').title()}",
                    f"- **Count**: {len(patterns)}",
                    f"- **Complexity**: {patterns[0].complexity.value}",
                    f"- **Risk Level**: {patterns[0].risk_level.value}",
                    f"- **Description**: {patterns[0].description}",
                    f"- **Migration Notes**: {patterns[0].migration_notes}",
                    f"- **Suggested Approach**: {patterns[0].suggested_approach}",
                    ""
                ])
        
        # Recommendations
        lines.extend([
            "## Preparation Steps",
            ""
        ])
        
        for i, step in enumerate(report.preparation_steps, 1):
            lines.append(f"{i}. {step}")
        
        lines.extend([
            "",
            "## Migration Phases",
            ""
        ])
        
        for phase in report.migration_phases:
            lines.extend([
                f"### Phase {phase['phase']}: {phase['name']}",
                f"- **Description**: {phase['description']}",
                f"- **Effort**: {phase['effort_percentage']}% of total",
                f"- **Deliverables**: {', '.join(phase['deliverables'])}",
                ""
            ])
        
        # Optimization opportunities
        if report.optimization_opportunities:
            lines.extend([
                "## Optimization Opportunities",
                ""
            ])
            
            for opportunity in report.optimization_opportunities:
                lines.append(f"- {opportunity}")
            
            lines.append("")
        
        # Potential issues
        if report.potential_issues:
            lines.extend([
                "## Potential Issues",
                ""
            ])
            
            for issue in report.potential_issues:
                lines.append(f"- ⚠️ {issue}")
        
        report_content = '\n'.join(lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    # Helper methods for analysis
    def _calculate_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _analyze_return_complexity(self, func_node: ast.FunctionDef) -> str:
        """Analyze return type complexity"""
        returns = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Tuple):
                    returns.append("tuple")
                elif isinstance(node.value, ast.List):
                    returns.append("list")
                elif isinstance(node.value, ast.Dict):
                    returns.append("dict")
                else:
                    returns.append("simple")
        
        if not returns:
            return "void"
        elif len(set(returns)) == 1:
            return returns[0]
        else:
            return "mixed"
    
    def _uses_global_variables(self, func_node: ast.FunctionDef) -> bool:
        """Check if function uses global variables"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Global):
                return True
        return False
    
    def _detect_side_effects(self, func_node: ast.FunctionDef) -> List[str]:
        """Detect potential side effects in function"""
        side_effects = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'print':
                        side_effects.append("console output")
                    elif node.func.id in ['open', 'write']:
                        side_effects.append("file I/O")
            elif isinstance(node, ast.Global):
                side_effects.append("global variable modification")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        side_effects.append("object state modification")
        
        return list(set(side_effects))
    
    def _assess_function_complexity(self, func_node: ast.FunctionDef) -> MigrationComplexity:
        """Assess migration complexity of a function"""
        complexity_score = self._calculate_cyclomatic_complexity(func_node)
        param_count = len(func_node.args.args)
        
        if complexity_score <= 5 and param_count <= 3:
            return MigrationComplexity.SIMPLE
        elif complexity_score <= 10 and param_count <= 6:
            return MigrationComplexity.MODERATE
        elif complexity_score <= 20:
            return MigrationComplexity.COMPLEX
        else:
            return MigrationComplexity.EXPERT
    
    def _suggest_function_improvements(self, func_node: ast.FunctionDef) -> List[str]:
        """Suggest improvements for function translation"""
        suggestions = []
        
        if len(func_node.args.args) > 5:
            suggestions.append("Consider using a struct/class for multiple parameters")
        
        if not func_node.returns:
            suggestions.append("Add return type annotation")
        
        if self._calculate_cyclomatic_complexity(func_node) > 10:
            suggestions.append("Break down into smaller functions")
        
        return suggestions
    
    def _uses_metaclass(self, class_node: ast.ClassDef) -> bool:
        """Check if class uses metaclass"""
        for keyword in class_node.keywords:
            if keyword.arg == 'metaclass':
                return True
        return False
    
    def _has_dynamic_attributes(self, class_node: ast.ClassDef) -> bool:
        """Check if class uses dynamic attributes"""
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'setattr':
                    return True
        return False
    
    def _get_special_methods(self, class_node: ast.ClassDef) -> List[str]:
        """Get list of special methods in class"""
        special_methods = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('__') and node.name.endswith('__'):
                    special_methods.append(node.name)
        
        return special_methods
    
    def _assess_class_complexity(self, class_node: ast.ClassDef) -> MigrationComplexity:
        """Assess migration complexity of a class"""
        method_count = len([n for n in class_node.body if isinstance(n, ast.FunctionDef)])
        inheritance_depth = len(class_node.bases)
        
        if self._uses_metaclass(class_node):
            return MigrationComplexity.EXPERT
        elif self._has_dynamic_attributes(class_node):
            return MigrationComplexity.COMPLEX
        elif method_count > 15 or inheritance_depth > 2:
            return MigrationComplexity.COMPLEX
        elif method_count > 8 or inheritance_depth > 1:
            return MigrationComplexity.MODERATE
        else:
            return MigrationComplexity.SIMPLE
    
    def _suggest_class_strategy(self, class_node: ast.ClassDef) -> str:
        """Suggest migration strategy for class"""
        if self._uses_metaclass(class_node):
            return "Manual implementation required - metaclasses have no C++ equivalent"
        elif self._has_dynamic_attributes(class_node):
            return "Consider using std::map for dynamic attributes or redesign"
        elif len(class_node.bases) > 1:
            return "Multiple inheritance - may need interface design in C++"
        else:
            return "Direct translation with minor adjustments"
    
    def _initialize_pattern_detectors(self) -> List:
        """Initialize pattern detection functions"""
        return [
            self._detect_list_comprehensions,
            self._detect_generators,
            self._detect_decorators,
            self._detect_context_managers,
            self._detect_async_patterns,
            self._detect_dynamic_typing,
            self._detect_string_formatting,
            self._detect_exception_handling
        ]
    
    def _detect_list_comprehensions(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect list comprehensions"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                pattern = CodePattern(
                    pattern_type="list_comprehension",
                    description="List/dict/set comprehension detected",
                    complexity=MigrationComplexity.MODERATE,
                    risk_level=RiskLevel.MEDIUM,
                    line_numbers=[node.lineno],
                    code_snippets=[ast.unparse(node) if hasattr(ast, 'unparse') else "comprehension"],
                    migration_notes="Comprehensions need to be converted to traditional loops or std::transform",
                    suggested_approach="Use std::transform with lambda functions or traditional for loops"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_generators(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect generator functions"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                pattern = CodePattern(
                    pattern_type="generator",
                    description="Generator function with yield detected",
                    complexity=MigrationComplexity.COMPLEX,
                    risk_level=RiskLevel.HIGH,
                    line_numbers=[node.lineno],
                    code_snippets=["yield statement"],
                    migration_notes="Generators require coroutines or iterator pattern in C++",
                    suggested_approach="Consider using C++20 coroutines or implement iterator pattern"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_decorators(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect decorators"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.decorator_list:
                pattern = CodePattern(
                    pattern_type="decorator",
                    description=f"{len(node.decorator_list)} decorator(s) found",
                    complexity=MigrationComplexity.COMPLEX,
                    risk_level=RiskLevel.HIGH,
                    line_numbers=[node.lineno],
                    code_snippets=[f"@{dec.id if isinstance(dec, ast.Name) else 'decorator'}" 
                                 for dec in node.decorator_list],
                    migration_notes="Decorators need manual implementation in C++",
                    suggested_approach="Implement decorator pattern or inline decorator functionality"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_context_managers(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect context managers (with statements)"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                pattern = CodePattern(
                    pattern_type="context_manager",
                    description="Context manager (with statement) detected",
                    complexity=MigrationComplexity.MODERATE,
                    risk_level=RiskLevel.MEDIUM,
                    line_numbers=[node.lineno],
                    code_snippets=["with statement"],
                    migration_notes="Context managers need RAII pattern in C++",
                    suggested_approach="Use RAII with constructors/destructors or smart pointers"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_async_patterns(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect async/await patterns"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncWith, ast.AsyncFor)):
                pattern = CodePattern(
                    pattern_type="async_pattern",
                    description="Async/await pattern detected",
                    complexity=MigrationComplexity.EXPERT,
                    risk_level=RiskLevel.CRITICAL,
                    line_numbers=[node.lineno],
                    code_snippets=["async/await"],
                    migration_notes="Async patterns require C++20 coroutines or async library",
                    suggested_approach="Use C++20 coroutines, std::async, or third-party async library"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_dynamic_typing(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect dynamic typing patterns"""
        patterns = []
        dynamic_operations = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['isinstance', 'type', 'hasattr', 'getattr', 'setattr']:
                        dynamic_operations += 1
        
        if dynamic_operations > 0:
            pattern = CodePattern(
                pattern_type="dynamic_typing",
                description=f"{dynamic_operations} dynamic typing operations detected",
                complexity=MigrationComplexity.COMPLEX,
                risk_level=RiskLevel.HIGH,
                line_numbers=[],
                code_snippets=["isinstance", "type", "hasattr", "getattr", "setattr"],
                migration_notes="Dynamic typing operations need careful C++ design",
                suggested_approach="Use templates, variants, or polymorphism for type flexibility"
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_string_formatting(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect string formatting patterns"""
        patterns = []
        
        if 'f"' in source_code or "f'" in source_code:
            pattern = CodePattern(
                pattern_type="f_string",
                description="F-string formatting detected",
                complexity=MigrationComplexity.SIMPLE,
                risk_level=RiskLevel.LOW,
                line_numbers=[],
                code_snippets=["f-string"],
                migration_notes="F-strings translate to string streams or fmt library",
                suggested_approach="Use std::stringstream or {fmt} library for formatting"
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_exception_handling(self, tree: ast.AST, source_code: str) -> List[CodePattern]:
        """Detect exception handling patterns"""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                pattern = CodePattern(
                    pattern_type="exception_handling",
                    description="Exception handling (try/except) detected",
                    complexity=MigrationComplexity.MODERATE,
                    risk_level=RiskLevel.MEDIUM,
                    line_numbers=[node.lineno],
                    code_snippets=["try/except"],
                    migration_notes="Exception handling translates to C++ try/catch",
                    suggested_approach="Use C++ exceptions with proper RAII for cleanup"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _initialize_complexity_weights(self) -> Dict[MigrationComplexity, float]:
        """Initialize complexity scoring weights"""
        return {
            MigrationComplexity.TRIVIAL: 0.5,
            MigrationComplexity.SIMPLE: 1.0,
            MigrationComplexity.MODERATE: 2.0,
            MigrationComplexity.COMPLEX: 3.0,
            MigrationComplexity.EXPERT: 4.0
        }
    
    def _load_dependency_mappings(self) -> Dict[str, Dict]:
        """Load dependency mapping information"""
        return {
            "numpy": {
                "alternatives": ["Eigen", "Armadillo", "Blaze"],
                "complexity": "complex",
                "patterns": ["array operations", "linear algebra"],
                "notes": "NumPy arrays map to Eigen matrices or std::vector"
            },
            "pandas": {
                "alternatives": ["custom data structures", "Apache Arrow"],
                "complexity": "expert",
                "patterns": ["DataFrame operations", "data analysis"],
                "notes": "No direct equivalent - consider designing custom data structures"
            },
            "requests": {
                "alternatives": ["libcurl", "cpprest", "httplib"],
                "complexity": "moderate",
                "patterns": ["HTTP requests"],
                "notes": "HTTP client libraries available for C++"
            },
            "json": {
                "alternatives": ["nlohmann/json", "RapidJSON"],
                "complexity": "simple",
                "patterns": ["JSON parsing"],
                "notes": "Multiple excellent C++ JSON libraries available"
            },
            "re": {
                "alternatives": ["std::regex", "PCRE", "RE2"],
                "complexity": "simple",
                "patterns": ["regular expressions"],
                "notes": "std::regex provides similar functionality"
            }
        }
    
    def _create_error_report(self, python_file: Path, error_message: str) -> MigrationReport:
        """Create error report for files that can't be analyzed"""
        return MigrationReport(
            source_file=python_file,
            total_lines=0,
            overall_complexity=MigrationComplexity.EXPERT,
            estimated_effort_hours=(0, 0),
            recommended_strategy=MigrationStrategy.MANUAL_REWRITE,
            risk_assessment=RiskLevel.CRITICAL,
            potential_issues=[error_message]
        )
    
    def _estimate_translation_accuracy(self, analysis: MigrationReport) -> float:
        """Estimate translation accuracy based on analysis"""
        base_accuracy = 0.8
        
        # Reduce accuracy for complex patterns
        complex_patterns = len([p for p in analysis.patterns 
                               if p.complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.EXPERT]])
        accuracy_reduction = complex_patterns * 0.1
        
        return max(0.3, base_accuracy - accuracy_reduction)
    
    def _get_direct_translation_risks(self, analysis: MigrationReport) -> List[str]:
        """Get risks for direct translation strategy"""
        risks = []
        
        if analysis.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            risks.append("High complexity may lead to incorrect translation")
        
        complex_deps = [d for d in analysis.dependencies 
                       if d.migration_effort in [MigrationComplexity.COMPLEX, MigrationComplexity.EXPERT]]
        if complex_deps:
            risks.append("Complex dependencies may not translate correctly")
        
        return risks
    
    def _get_refactor_risks(self, analysis: MigrationReport) -> List[str]:
        """Get risks for refactor-then-translate strategy"""
        return [
            "Refactoring may introduce bugs",
            "Additional time investment required",
            "May change original behavior"
        ]
    
    def _recommend_strategy(self, analysis: MigrationReport, strategies: Dict) -> str:
        """Recommend best migration strategy"""
        if analysis.overall_complexity in [MigrationComplexity.TRIVIAL, MigrationComplexity.SIMPLE]:
            return MigrationStrategy.DIRECT_TRANSLATION.value
        elif analysis.overall_complexity == MigrationComplexity.MODERATE:
            return MigrationStrategy.REFACTOR_THEN_TRANSLATE.value
        elif analysis.risk_assessment == RiskLevel.CRITICAL:
            return MigrationStrategy.MANUAL_REWRITE.value
        else:
            return MigrationStrategy.HYBRID_APPROACH.value
    
    def _create_comparison_matrix(self, strategies: Dict) -> Dict:
        """Create strategy comparison matrix"""
        matrix = {}
        
        criteria = ["effort", "accuracy", "maintenance", "performance"]
        
        for criterion in criteria:
            matrix[criterion] = {}
            for strategy_name, strategy_data in strategies.items():
                matrix[criterion][strategy_name.value] = strategy_data.get(criterion, "Unknown")
        
        return matrix
    
    def _sort_by_migration_order(self, analyses: List[MigrationReport]) -> List[MigrationReport]:
        """Sort files by recommended migration order"""
        def migration_priority(analysis: MigrationReport) -> Tuple[int, int, int]:
            # Sort by: complexity (ascending), dependencies (ascending), risk (ascending)
            complexity_order = {
                MigrationComplexity.TRIVIAL: 0,
                MigrationComplexity.SIMPLE: 1,
                MigrationComplexity.MODERATE: 2,
                MigrationComplexity.COMPLEX: 3,
                MigrationComplexity.EXPERT: 4
            }
            
            risk_order = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 1,
                RiskLevel.HIGH: 2,
                RiskLevel.CRITICAL: 3
            }
            
            return (
                complexity_order[analysis.overall_complexity],
                len(analysis.dependencies),
                risk_order[analysis.risk_assessment]
            )
        
        return sorted(analyses, key=migration_priority)
    
    def _calculate_total_effort(self, analyses: List[MigrationReport]) -> Tuple[int, int]:
        """Calculate total effort range for all files"""
        min_total = sum(a.estimated_effort_hours[0] for a in analyses)
        max_total = sum(a.estimated_effort_hours[1] for a in analyses)
        return (min_total, max_total)
    
    def _calculate_average_complexity(self, analyses: List[MigrationReport]) -> str:
        """Calculate average complexity across all files"""
        complexity_scores = {
            MigrationComplexity.TRIVIAL: 0,
            MigrationComplexity.SIMPLE: 1,
            MigrationComplexity.MODERATE: 2,
            MigrationComplexity.COMPLEX: 3,
            MigrationComplexity.EXPERT: 4
        }
        
        if not analyses:
            return "unknown"
        
        avg_score = sum(complexity_scores[a.overall_complexity] for a in analyses) / len(analyses)
        
        if avg_score <= 0.5:
            return "trivial"
        elif avg_score <= 1.5:
            return "simple"
        elif avg_score <= 2.5:
            return "moderate"
        elif avg_score <= 3.5:
            return "complex"
        else:
            return "expert"
    
    def _create_migration_phases(self, sorted_analyses: List[MigrationReport]) -> List[Dict]:
        """Create migration phases based on complexity"""
        phases = []
        
        # Group files by complexity
        simple_files = [a for a in sorted_analyses 
                       if a.overall_complexity in [MigrationComplexity.TRIVIAL, MigrationComplexity.SIMPLE]]
        moderate_files = [a for a in sorted_analyses 
                         if a.overall_complexity == MigrationComplexity.MODERATE]
        complex_files = [a for a in sorted_analyses 
                        if a.overall_complexity in [MigrationComplexity.COMPLEX, MigrationComplexity.EXPERT]]
        
        if simple_files:
            phases.append({
                "phase": 1,
                "name": "Foundation",
                "description": "Translate simple, low-risk files first",
                "files": [a.source_file.name for a in simple_files[:5]],  # Limit to first 5
                "estimated_weeks": max(1, len(simple_files) // 3)
            })
        
        if moderate_files:
            phases.append({
                "phase": 2,
                "name": "Core Implementation",
                "description": "Translate moderate complexity files",
                "files": [a.source_file.name for a in moderate_files],
                "estimated_weeks": max(2, len(moderate_files) * 2)
            })
        
        if complex_files:
            phases.append({
                "phase": 3,
                "name": "Advanced Features",
                "description": "Handle complex files requiring expert attention",
                "files": [a.source_file.name for a in complex_files],
                "estimated_weeks": max(3, len(complex_files) * 4)
            })
        
        return phases
    
    def _analyze_cross_file_dependencies(self, analyses: List[MigrationReport]) -> Dict:
        """Analyze dependencies between files"""
        # This is a simplified version - would need more sophisticated import analysis
        all_dependencies = set()
        for analysis in analyses:
            for dep in analysis.dependencies:
                all_dependencies.add(dep.name)
        
        return {
            "total_unique_dependencies": len(all_dependencies),
            "common_dependencies": list(all_dependencies)[:10],  # Top 10
            "complex_dependencies": [dep for analysis in analyses 
                                   for dep in analysis.dependencies 
                                   if dep.migration_effort in [MigrationComplexity.COMPLEX, MigrationComplexity.EXPERT]]
        }
    
    def _estimate_resource_requirements(self, analyses: List[MigrationReport]) -> Dict:
        """Estimate resource requirements for migration"""
        total_effort = self._calculate_total_effort(analyses)
        
        return {
            "estimated_developer_months": (total_effort[0] // 160, total_effort[1] // 160),
            "recommended_team_size": min(4, max(1, len(analyses) // 5)),
            "required_skills": [
                "C++ expertise (modern C++17/20)",
                "Python proficiency",
                "Software architecture",
                "Testing and validation"
            ],
            "tools_needed": [
                "C++ compiler and build system",
                "Profiling and debugging tools",
                "Static analysis tools",
                "Testing frameworks"
            ]
        }
    
    def _create_migration_timeline(self, sorted_analyses: List[MigrationReport]) -> Dict:
        """Create detailed migration timeline"""
        phases = self._create_migration_phases(sorted_analyses)
        
        timeline = {
            "total_duration_weeks": sum(phase.get("estimated_weeks", 1) for phase in phases),
            "phases": phases,
            "milestones": [
                {
                    "week": 2,
                    "milestone": "Project setup and first simple translations completed",
                    "deliverables": ["Build system", "CI/CD", "First translated modules"]
                },
                {
                    "week": sum(phase.get("estimated_weeks", 1) for phase in phases[:2]),
                    "milestone": "Core functionality translated and tested",
                    "deliverables": ["Core modules", "Test suites", "Performance benchmarks"]
                },
                {
                    "week": sum(phase.get("estimated_weeks", 1) for phase in phases),
                    "milestone": "Complete migration with optimization",
                    "deliverables": ["All modules", "Documentation", "Final optimization"]
                }
            ]
        }
        
        return timeline
    
    def _create_risk_mitigation_plan(self, analyses: List[MigrationReport]) -> Dict:
        """Create risk mitigation plan"""
        high_risk_files = [a for a in analyses if a.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        return {
            "high_risk_files": len(high_risk_files),
            "mitigation_strategies": [
                "Implement comprehensive test coverage before translation",
                "Use incremental translation with frequent validation",
                "Plan for manual code review of complex translations",
                "Maintain parallel Python version for comparison",
                "Consider hybrid approach for highest-risk components"
            ],
            "contingency_plans": [
                "Keep critical Python modules as fallback",
                "Plan additional time buffer for complex files",
                "Have C++ expert available for consultation",
                "Prepare for potential architecture changes"
            ]
        }


# CLI Integration
def analyze_migration_cli():
    """Command-line interface for migration analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Python-to-C++ Migration Analyzer")
    parser.add_argument("python_file", help="Python file to analyze")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--compare-strategies", action="store_true", 
                       help="Compare different migration strategies")
    
    args = parser.parse_args()
    
    analyzer = AdvancedMigrationAnalyzer()
    python_file = Path(args.python_file)
    
    if not python_file.exists():
        print(f"Error: File {python_file} not found")
        return
    
    if args.compare_strategies:
        # Strategy comparison
        comparison = analyzer.compare_migration_strategies(python_file)
        
        print("=== Migration Strategy Comparison ===")
        print(f"File: {python_file}")
        print(f"Overall Complexity: {comparison['analysis'].overall_complexity.value}")
        print(f"Risk Level: {comparison['analysis'].risk_assessment.value}")
        print(f"Recommended Strategy: {comparison['recommended']}")
        print()
        
        for strategy, details in comparison['strategies'].items():
            print(f"Strategy: {strategy.value.replace('_', ' ').title()}")
            print(f"  Description: {details['description']}")
            print(f"  Effort: {details['effort'][0]}-{details['effort'][1]} hours")
            print(f"  Accuracy: {details['accuracy']:.1%}")
            print(f"  Maintenance: {details['maintenance']}")
            print(f"  Performance: {details['performance']}")
            print()
    else:
        # Standard analysis
        report = analyzer.analyze_migration(python_file)
        
        print("=== Migration Analysis Report ===")
        print(f"File: {python_file}")
        print(f"Overall Complexity: {report.overall_complexity.value}")
        print(f"Estimated Effort: {report.estimated_effort_hours[0]}-{report.estimated_effort_hours[1]} hours")
        print(f"Risk Level: {report.risk_assessment.value}")
        print(f"Recommended Strategy: {report.recommended_strategy.value}")
        print()
        
        print(f"Functions: {len(report.functions)}")
        print(f"Classes: {len(report.classes)}")
        print(f"Dependencies: {len(report.dependencies)}")
        print(f"Detected Patterns: {len(report.patterns)}")
        
        if args.output:
            output_file = Path(args.output)
            analyzer.generate_detailed_report(report, output_file)
            print(f"\n✅ Detailed report saved to {output_file}")


if __name__ == "__main__":
    analyze_migration_cli()
