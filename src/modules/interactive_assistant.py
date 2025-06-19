"""
Interactive Translation Assistant for Python-to-C++ Translation
Provides real-time feedback, suggestions, and guided translation
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import difflib


class SuggestionType(Enum):
    TYPE_HINT = "type_hint"
    OPTIMIZATION = "optimization"
    LIBRARY_REPLACEMENT = "library_replacement"
    REFACTORING = "refactoring"
    ERROR_HANDLING = "error_handling"
    MEMORY_MANAGEMENT = "memory_management"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TranslationSuggestion:
    """Represents a translation improvement suggestion"""
    suggestion_type: SuggestionType
    severity: Severity
    message: str
    line_number: int
    code_snippet: str
    suggested_fix: str
    explanation: str
    auto_fixable: bool = False


@dataclass
class TranslationProgress:
    """Tracks translation progress and quality"""
    total_functions: int
    translated_functions: int
    total_classes: int
    translated_classes: int
    quality_score: float
    completeness: float
    suggestions_count: int
    errors_count: int


@dataclass
class InteractiveSession:
    """Represents an interactive translation session"""
    session_id: str
    source_file: Path
    target_file: Path
    suggestions: List[TranslationSuggestion]
    progress: TranslationProgress
    user_preferences: Dict[str, Any]
    history: List[Dict[str, Any]]


class InteractiveTranslationAssistant:
    """Provides interactive guidance for Python-to-C++ translation"""
    
    def __init__(self):
        self.sessions: Dict[str, InteractiveSession] = {}
        self.suggestion_rules = self._load_suggestion_rules()
        self.code_patterns = self._load_code_patterns()
        self.user_preferences = self._load_default_preferences()
    
    def start_interactive_session(self, python_file: Path, 
                                target_file: Optional[Path] = None) -> str:
        """Start a new interactive translation session"""
        import uuid
        
        session_id = str(uuid.uuid4())
        
        if target_file is None:
            target_file = python_file.with_suffix('.cpp')
        
        # Analyze source file
        progress = self._analyze_source_file(python_file)
        
        session = InteractiveSession(
            session_id=session_id,
            source_file=python_file,
            target_file=target_file,
            suggestions=[],
            progress=progress,
            user_preferences=self.user_preferences.copy(),
            history=[]
        )
        
        # Generate initial suggestions
        session.suggestions = self._analyze_and_suggest(python_file)
        
        self.sessions[session_id] = session
        
        return session_id
    
    def get_suggestions(self, session_id: str) -> List[TranslationSuggestion]:
        """Get current suggestions for a session"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        return session.suggestions
    
    def apply_suggestion(self, session_id: str, suggestion_index: int) -> Dict[str, Any]:
        """Apply a specific suggestion and return the result"""
        session = self.sessions.get(session_id)
        if not session or suggestion_index >= len(session.suggestions):
            return {"success": False, "error": "Invalid session or suggestion"}
        
        suggestion = session.suggestions[suggestion_index]
        
        if not suggestion.auto_fixable:
            return {"success": False, "error": "Suggestion requires manual intervention"}
        
        # Apply the fix
        try:
            updated_code = self._apply_suggestion_fix(session.source_file, suggestion)
            
            # Save the change
            with open(session.source_file, 'w', encoding='utf-8') as f:
                f.write(updated_code)
            
            # Log the change
            session.history.append({
                "action": "apply_suggestion",
                "suggestion_type": suggestion.suggestion_type.value,
                "line_number": suggestion.line_number,
                "timestamp": self._get_timestamp()
            })
            
            # Re-analyze and update suggestions
            session.suggestions = self._analyze_and_suggest(session.source_file)
            session.progress = self._analyze_source_file(session.source_file)
            
            return {
                "success": True,
                "updated_code": updated_code,
                "new_suggestions_count": len(session.suggestions)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_real_time_feedback(self, session_id: str, code_snippet: str, 
                             line_number: int) -> List[TranslationSuggestion]:
        """Get real-time feedback for a code snippet being edited"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        suggestions = []
        
        # Analyze the specific code snippet
        try:
            tree = ast.parse(code_snippet)
            
            for rule in self.suggestion_rules:
                matches = rule['pattern'](tree, code_snippet)
                if matches:
                    suggestion = TranslationSuggestion(
                        suggestion_type=SuggestionType(rule['type']),
                        severity=Severity(rule['severity']),
                        message=rule['message'],
                        line_number=line_number,
                        code_snippet=code_snippet,
                        suggested_fix=rule['fix'](code_snippet),
                        explanation=rule['explanation'],
                        auto_fixable=rule.get('auto_fixable', False)
                    )
                    suggestions.append(suggestion)
        
        except SyntaxError:
            # Handle incomplete code during typing
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.ERROR_HANDLING,
                severity=Severity.WARNING,
                message="Incomplete code detected",
                line_number=line_number,
                code_snippet=code_snippet,
                suggested_fix="Complete the current statement",
                explanation="The code appears to be incomplete. Finish the current statement before translation.",
                auto_fixable=False
            ))
        
        return suggestions
    
    def generate_translation_plan(self, session_id: str) -> Dict[str, Any]:
        """Generate a step-by-step translation plan"""
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        # Analyze complexity and dependencies
        analysis = self._comprehensive_analysis(session.source_file)
        
        plan = {
            "overview": {
                "estimated_time": self._estimate_translation_time(analysis),
                "difficulty": self._assess_difficulty(analysis),
                "key_challenges": self._identify_challenges(analysis)
            },
            "phases": [
                {
                    "phase": 1,
                    "name": "Preparation",
                    "tasks": [
                        "Add type hints to improve translation accuracy",
                        "Simplify complex expressions",
                        "Add docstrings for better C++ documentation"
                    ],
                    "estimated_time": "15-30 minutes"
                },
                {
                    "phase": 2,
                    "name": "Core Translation",
                    "tasks": [
                        "Translate basic functions and classes",
                        "Map Python data structures to C++",
                        "Handle control flow and logic"
                    ],
                    "estimated_time": "30-60 minutes"
                },
                {
                    "phase": 3,
                    "name": "Optimization",
                    "tasks": [
                        "Apply memory management optimizations",
                        "Add const correctness",
                        "Optimize performance-critical sections"
                    ],
                    "estimated_time": "20-40 minutes"
                },
                {
                    "phase": 4,
                    "name": "Testing & Validation",
                    "tasks": [
                        "Generate comprehensive test suite",
                        "Validate translation correctness",
                        "Performance benchmarking"
                    ],
                    "estimated_time": "30-45 minutes"
                }
            ],
            "dependencies": analysis.get("dependencies", []),
            "risks": analysis.get("risks", []),
            "recommendations": analysis.get("recommendations", [])
        }
        
        return plan
    
    def get_translation_status(self, session_id: str) -> Dict[str, Any]:
        """Get current translation status and progress"""
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        # Calculate detailed progress metrics
        progress = session.progress
        
        status = {
            "session_id": session_id,
            "source_file": str(session.source_file),
            "target_file": str(session.target_file),
            "progress": {
                "functions": {
                    "total": progress.total_functions,
                    "translated": progress.translated_functions,
                    "percentage": (progress.translated_functions / progress.total_functions * 100) 
                                 if progress.total_functions > 0 else 0
                },
                "classes": {
                    "total": progress.total_classes,
                    "translated": progress.translated_classes,
                    "percentage": (progress.translated_classes / progress.total_classes * 100) 
                                 if progress.total_classes > 0 else 0
                },
                "overall": {
                    "completeness": progress.completeness,
                    "quality_score": progress.quality_score
                }
            },
            "issues": {
                "suggestions": progress.suggestions_count,
                "errors": progress.errors_count,
                "critical_issues": len([s for s in session.suggestions 
                                      if s.severity == Severity.CRITICAL])
            },
            "recent_activity": session.history[-5:],  # Last 5 actions
            "next_suggestions": session.suggestions[:3]  # Top 3 suggestions
        }
        
        return status
    
    def suggest_code_improvements(self, session_id: str, 
                                code_snippet: str) -> List[TranslationSuggestion]:
        """Suggest improvements for specific code snippet"""
        suggestions = []
        
        # Check for common Python patterns that need special C++ handling
        improvements = [
            self._check_list_comprehensions(code_snippet),
            self._check_string_formatting(code_snippet),
            self._check_exception_handling(code_snippet),
            self._check_memory_management(code_snippet),
            self._check_performance_patterns(code_snippet)
        ]
        
        for improvement_list in improvements:
            suggestions.extend(improvement_list)
        
        return suggestions
    
    def generate_diff_preview(self, session_id: str, 
                            suggestion: TranslationSuggestion) -> str:
        """Generate a diff preview for applying a suggestion"""
        session = self.sessions.get(session_id)
        if not session:
            return ""
        
        try:
            with open(session.source_file, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # Apply the suggestion to get modified content
            modified_content = self._apply_suggestion_fix(session.source_file, suggestion)
            modified_lines = modified_content.splitlines(keepends=True)
            
            # Generate unified diff
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=str(session.source_file),
                tofile=f"{session.source_file} (with suggestion)",
                lineterm=""
            )
            
            return ''.join(diff)
            
        except Exception:
            return "Unable to generate diff preview"
    
    def export_session_report(self, session_id: str, output_file: Path) -> str:
        """Export a comprehensive session report"""
        session = self.sessions.get(session_id)
        if not session:
            return ""
        
        report_lines = [
            "# Interactive Translation Session Report",
            "",
            f"**Session ID**: {session_id}",
            f"**Source File**: {session.source_file}",
            f"**Target File**: {session.target_file}",
            f"**Session Duration**: {self._calculate_session_duration(session)}",
            "",
            "## Translation Progress",
            "",
            f"- **Functions**: {session.progress.translated_functions}/{session.progress.total_functions} "
            f"({session.progress.translated_functions/session.progress.total_functions*100:.1f}%)",
            f"- **Classes**: {session.progress.translated_classes}/{session.progress.total_classes} "
            f"({session.progress.translated_classes/session.progress.total_classes*100:.1f}%)",
            f"- **Quality Score**: {session.progress.quality_score:.2f}/10",
            f"- **Completeness**: {session.progress.completeness:.1f}%",
            "",
            "## Suggestions Summary",
            ""
        ]
        
        # Group suggestions by type
        suggestion_groups = {}
        for suggestion in session.suggestions:
            stype = suggestion.suggestion_type.value
            if stype not in suggestion_groups:
                suggestion_groups[stype] = []
            suggestion_groups[stype].append(suggestion)
        
        for stype, suggestions in suggestion_groups.items():
            report_lines.extend([
                f"### {stype.replace('_', ' ').title()}",
                f"**Count**: {len(suggestions)}",
                ""
            ])
            
            for suggestion in suggestions[:5]:  # Show top 5
                report_lines.extend([
                    f"- **Line {suggestion.line_number}**: {suggestion.message}",
                    f"  - *Severity*: {suggestion.severity.value}",
                    f"  - *Auto-fixable*: {'Yes' if suggestion.auto_fixable else 'No'}",
                    ""
                ])
        
        report_lines.extend([
            "## Activity History",
            ""
        ])
        
        for activity in session.history:
            report_lines.append(
                f"- **{activity['timestamp']}**: {activity['action']} "
                f"({activity.get('suggestion_type', 'N/A')})"
            )
        
        report_lines.extend([
            "",
            "## Recommendations",
            "",
            "1. **Review critical suggestions** before finalizing translation",
            "2. **Add comprehensive tests** to validate translation accuracy",
            "3. **Performance benchmark** the translated code",
            "4. **Code review** with C++ best practices in mind",
            ""
        ])
        
        report_content = '\n'.join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    # Helper methods
    def _load_suggestion_rules(self) -> List[Dict]:
        """Load suggestion rules for code analysis"""
        return [
            {
                'type': 'type_hint',
                'severity': 'warning',
                'pattern': lambda tree, code: not self._has_type_hints(tree),
                'message': 'Consider adding type hints for better C++ translation',
                'fix': lambda code: self._suggest_type_hints(code),
                'explanation': 'Type hints help generate more accurate C++ types',
                'auto_fixable': False
            },
            {
                'type': 'optimization',
                'severity': 'info',
                'pattern': lambda tree, code: 'for i in range(len(' in code,
                'message': 'Consider using enumerate() instead of range(len())',
                'fix': lambda code: self._suggest_enumerate_fix(code),
                'explanation': 'enumerate() is more Pythonic and translates better to C++',
                'auto_fixable': True
            },
            {
                'type': 'library_replacement',
                'severity': 'warning',
                'pattern': lambda tree, code: 'import numpy' in code,
                'message': 'NumPy operations may need manual C++ implementation',
                'fix': lambda code: '// Consider using Eigen library for matrix operations',
                'explanation': 'NumPy has no direct C++ equivalent, consider Eigen library',
                'auto_fixable': False
            }
        ]
    
    def _load_code_patterns(self) -> Dict:
        """Load code patterns for recognition and suggestion"""
        return {
            'list_comprehension': r'\[.*for.*in.*\]',
            'string_formatting': r'f["\'].*{.*}.*["\']',
            'exception_handling': r'try:|except.*:|finally:',
            'generator': r'yield\s+',
            'decorator': r'@\w+'
        }
    
    def _load_default_preferences(self) -> Dict:
        """Load default user preferences"""
        return {
            'auto_apply_safe_fixes': True,
            'suggestion_verbosity': 'medium',
            'cpp_standard': 'c++17',
            'optimization_level': 'O2',
            'include_comments': True,
            'generate_tests': True
        }
    
    def _analyze_source_file(self, python_file: Path) -> TranslationProgress:
        """Analyze source file to determine translation progress"""
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            total_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            total_classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            
            # For now, assume no functions/classes are translated yet
            # This would be updated as translation progresses
            
            return TranslationProgress(
                total_functions=total_functions,
                translated_functions=0,
                total_classes=total_classes,
                translated_classes=0,
                quality_score=0.0,
                completeness=0.0,
                suggestions_count=0,
                errors_count=0
            )
            
        except Exception:
            return TranslationProgress(0, 0, 0, 0, 0.0, 0.0, 0, 1)
    
    def _analyze_and_suggest(self, python_file: Path) -> List[TranslationSuggestion]:
        """Analyze file and generate suggestions"""
        suggestions = []
        
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            tree = ast.parse(content)
            
            # Apply suggestion rules
            for rule in self.suggestion_rules:
                if rule['pattern'](tree, content):
                    suggestion = TranslationSuggestion(
                        suggestion_type=SuggestionType(rule['type']),
                        severity=Severity(rule['severity']),
                        message=rule['message'],
                        line_number=1,  # Would need more sophisticated line detection
                        code_snippet=content[:100],  # First 100 chars as preview
                        suggested_fix=rule['fix'](content),
                        explanation=rule['explanation'],
                        auto_fixable=rule.get('auto_fixable', False)
                    )
                    suggestions.append(suggestion)
            
            # Check for specific patterns
            for i, line in enumerate(lines):
                line_suggestions = self._analyze_line(line, i + 1)
                suggestions.extend(line_suggestions)
        
        except Exception as e:
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.ERROR_HANDLING,
                severity=Severity.ERROR,
                message=f"Failed to analyze file: {str(e)}",
                line_number=1,
                code_snippet="",
                suggested_fix="Fix syntax errors in Python code",
                explanation="The file contains syntax errors that prevent analysis",
                auto_fixable=False
            ))
        
        return suggestions
    
    def _analyze_line(self, line: str, line_number: int) -> List[TranslationSuggestion]:
        """Analyze a single line for suggestions"""
        suggestions = []
        
        # Check for common patterns
        if re.search(r'print\s*\(', line):
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.LIBRARY_REPLACEMENT,
                severity=Severity.INFO,
                message="print() will be replaced with std::cout",
                line_number=line_number,
                code_snippet=line.strip(),
                suggested_fix=line.replace('print(', 'std::cout << ').replace(')', ' << std::endl;'),
                explanation="Python print() maps to C++ std::cout",
                auto_fixable=True
            ))
        
        if re.search(r'def\s+\w+\([^)]*\)\s*:', line) and '->' not in line:
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.TYPE_HINT,
                severity=Severity.WARNING,
                message="Function missing return type annotation",
                line_number=line_number,
                code_snippet=line.strip(),
                suggested_fix=line.rstrip(':') + ' -> ReturnType:',
                explanation="Return type annotations help generate better C++ function signatures",
                auto_fixable=False
            ))
        
        return suggestions
    
    def _apply_suggestion_fix(self, python_file: Path, suggestion: TranslationSuggestion) -> str:
        """Apply a suggestion fix to the source code"""
        with open(python_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Simple line-based replacement (would need more sophisticated logic for complex fixes)
        if suggestion.line_number <= len(lines):
            if suggestion.suggestion_type == SuggestionType.OPTIMIZATION:
                # Handle enumerate fix
                original_line = lines[suggestion.line_number - 1]
                if 'for i in range(len(' in original_line:
                    # Replace with enumerate
                    new_line = original_line.replace(
                        'for i in range(len(',
                        'for i, item in enumerate('
                    ).replace(')):', '))')
                    lines[suggestion.line_number - 1] = new_line
        
        return ''.join(lines)
    
    def _comprehensive_analysis(self, python_file: Path) -> Dict[str, Any]:
        """Perform comprehensive analysis for translation planning"""
        analysis = {
            "complexity": "medium",
            "dependencies": [],
            "risks": [],
            "recommendations": []
        }
        
        try:
            with open(python_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["dependencies"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["dependencies"].append(node.module)
            
            # Identify risks
            if 'numpy' in analysis["dependencies"]:
                analysis["risks"].append("NumPy dependency requires C++ matrix library")
            
            if any('eval' in line for line in content.split('\n')):
                analysis["risks"].append("Dynamic code execution detected")
            
            # Generate recommendations
            analysis["recommendations"].extend([
                "Add type hints to all function parameters and return values",
                "Consider breaking down complex functions",
                "Review exception handling for C++ translation"
            ])
        
        except Exception:
            analysis["complexity"] = "high"
            analysis["risks"].append("File contains syntax errors")
        
        return analysis
    
    def _estimate_translation_time(self, analysis: Dict) -> str:
        """Estimate translation time based on analysis"""
        base_time = 30  # minutes
        
        complexity_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0
        }
        
        multiplier = complexity_multiplier.get(analysis.get("complexity", "medium"), 1.0)
        estimated_minutes = int(base_time * multiplier)
        
        if estimated_minutes < 60:
            return f"{estimated_minutes} minutes"
        else:
            hours = estimated_minutes // 60
            minutes = estimated_minutes % 60
            return f"{hours}h {minutes}m"
    
    def _assess_difficulty(self, analysis: Dict) -> str:
        """Assess translation difficulty"""
        risk_count = len(analysis.get("risks", []))
        dependency_count = len(analysis.get("dependencies", []))
        
        if risk_count == 0 and dependency_count <= 2:
            return "Easy"
        elif risk_count <= 2 and dependency_count <= 5:
            return "Medium"
        else:
            return "Hard"
    
    def _identify_challenges(self, analysis: Dict) -> List[str]:
        """Identify key translation challenges"""
        challenges = []
        
        dependencies = analysis.get("dependencies", [])
        
        if 'numpy' in dependencies:
            challenges.append("NumPy array operations")
        if 'pandas' in dependencies:
            challenges.append("DataFrame operations")
        if 'asyncio' in dependencies:
            challenges.append("Async/await patterns")
        
        return challenges
    
    def _check_list_comprehensions(self, code: str) -> List[TranslationSuggestion]:
        """Check for list comprehensions that need special handling"""
        suggestions = []
        
        if re.search(self.code_patterns['list_comprehension'], code):
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.REFACTORING,
                severity=Severity.INFO,
                message="List comprehension detected - consider refactoring for C++",
                line_number=1,
                code_snippet=code,
                suggested_fix="// Convert to std::transform or traditional loop",
                explanation="List comprehensions don't have direct C++ equivalent",
                auto_fixable=False
            ))
        
        return suggestions
    
    def _check_string_formatting(self, code: str) -> List[TranslationSuggestion]:
        """Check for string formatting patterns"""
        suggestions = []
        
        if re.search(self.code_patterns['string_formatting'], code):
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.LIBRARY_REPLACEMENT,
                severity=Severity.INFO,
                message="f-string detected - will use string streams in C++",
                line_number=1,
                code_snippet=code,
                suggested_fix="// Use std::stringstream or fmt library",
                explanation="f-strings map to C++ string formatting",
                auto_fixable=False
            ))
        
        return suggestions
    
    def _check_exception_handling(self, code: str) -> List[TranslationSuggestion]:
        """Check for exception handling patterns"""
        suggestions = []
        
        if re.search(self.code_patterns['exception_handling'], code):
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.ERROR_HANDLING,
                severity=Severity.WARNING,
                message="Exception handling needs C++ translation",
                line_number=1,
                code_snippet=code,
                suggested_fix="// Use try-catch with std::exception",
                explanation="Python exceptions map to C++ exception classes",
                auto_fixable=False
            ))
        
        return suggestions
    
    def _check_memory_management(self, code: str) -> List[TranslationSuggestion]:
        """Check for memory management concerns"""
        suggestions = []
        
        if 'list(' in code or 'dict(' in code:
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.MEMORY_MANAGEMENT,
                severity=Severity.INFO,
                message="Dynamic containers detected - consider smart pointers",
                line_number=1,
                code_snippet=code,
                suggested_fix="// Use std::vector, std::map with proper RAII",
                explanation="Python containers map to C++ STL containers",
                auto_fixable=False
            ))
        
        return suggestions
    
    def _check_performance_patterns(self, code: str) -> List[TranslationSuggestion]:
        """Check for performance-related patterns"""
        suggestions = []
        
        if 'range(len(' in code:
            suggestions.append(TranslationSuggestion(
                suggestion_type=SuggestionType.OPTIMIZATION,
                severity=Severity.INFO,
                message="Prefer enumerate() over range(len())",
                line_number=1,
                code_snippet=code,
                suggested_fix=code.replace('range(len(', 'enumerate('),
                explanation="enumerate() is more efficient and Pythonic",
                auto_fixable=True
            ))
        
        return suggestions
    
    def _has_type_hints(self, tree: ast.AST) -> bool:
        """Check if AST contains type hints"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    return True
        return False
    
    def _suggest_type_hints(self, code: str) -> str:
        """Suggest type hints for code"""
        return "# Add type hints: def function(param: int) -> str:"
    
    def _suggest_enumerate_fix(self, code: str) -> str:
        """Suggest enumerate fix"""
        return code.replace('range(len(', 'enumerate(')
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _calculate_session_duration(self, session: InteractiveSession) -> str:
        """Calculate session duration"""
        if not session.history:
            return "0 minutes"
        
        # Would calculate based on first and last activity timestamps
        return "45 minutes"  # Placeholder
    
    def cleanup_session(self, session_id: str):
        """Clean up a completed session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# CLI Integration
def interactive_translation_cli():
    """Command-line interface for interactive translation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Python-to-C++ Translation Assistant")
    parser.add_argument("python_file", help="Python file to translate")
    parser.add_argument("--output", "-o", help="Output C++ file")
    parser.add_argument("--report", "-r", help="Generate session report")
    
    args = parser.parse_args()
    
    assistant = InteractiveTranslationAssistant()
    python_file = Path(args.python_file)
    
    if not python_file.exists():
        print(f"Error: File {python_file} not found")
        return
    
    # Start interactive session
    session_id = assistant.start_interactive_session(python_file)
    
    print(f"Started interactive translation session: {session_id}")
    print(f"Analyzing {python_file}...")
    
    # Get initial suggestions
    suggestions = assistant.get_suggestions(session_id)
    
    print(f"\nFound {len(suggestions)} suggestions:")
    for i, suggestion in enumerate(suggestions[:5]):  # Show first 5
        print(f"{i+1}. [{suggestion.severity.value.upper()}] {suggestion.message}")
        print(f"   Line {suggestion.line_number}: {suggestion.code_snippet[:50]}...")
        if suggestion.auto_fixable:
            print("   (Auto-fixable)")
        print()
    
    # Get translation plan
    plan = assistant.generate_translation_plan(session_id)
    print("Translation Plan:")
    print(f"- Estimated time: {plan['overview']['estimated_time']}")
    print(f"- Difficulty: {plan['overview']['difficulty']}")
    print(f"- Key challenges: {', '.join(plan['overview']['key_challenges'])}")
    
    # Generate report if requested
    if args.report:
        report_file = Path(args.report)
        assistant.export_session_report(session_id, report_file)
        print(f"\nSession report saved to {report_file}")
    
    assistant.cleanup_session(session_id)


if __name__ == "__main__":
    interactive_translation_cli()
