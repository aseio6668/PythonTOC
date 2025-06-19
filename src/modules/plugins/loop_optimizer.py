"""
Loop Optimization Plugin

Optimizes common loop patterns for better C++ performance.
"""

import re
from typing import Dict, Any
from plugin_system import OptimizationPlugin


class LoopOptimizer(OptimizationPlugin):
    """Optimization plugin for loop patterns"""
    
    def __init__(self):
        super().__init__()
        self.name = "LoopOptimizer"
        self.version = "1.0.0"
        self.description = "Loop optimization plugin"
        self.author = "Python to C++ Translator"
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def can_optimize(self, code: str, metadata: Dict[str, Any]) -> bool:
        # Look for range-based loops that can be optimized
        return "for" in code and "range(" in code
    
    def optimize(self, code: str, metadata: Dict[str, Any]) -> str:
        # Convert range-based loops to more efficient C++ patterns
        optimized = code
        
        # Pattern: for i in range(n)
        pattern1 = r'for\s+(\w+)\s+in\s+range\((\w+)\)'
        replacement1 = r'for (int \1 = 0; \1 < \2; ++\1)'
        optimized = re.sub(pattern1, replacement1, optimized)
        
        # Pattern: for i in range(start, end)
        pattern2 = r'for\s+(\w+)\s+in\s+range\((\w+),\s*(\w+)\)'
        replacement2 = r'for (int \1 = \2; \1 < \3; ++\1)'
        optimized = re.sub(pattern2, replacement2, optimized)
        
        return optimized
    
    def get_optimization_description(self) -> str:
        return "Range-based loop optimization"
