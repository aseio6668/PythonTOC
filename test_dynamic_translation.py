#!/usr/bin/env python3
"""Test dynamic module translation"""

from src.modules.dependency_manager import DynamicModuleAnalyzer
from pathlib import Path

def test_translation():
    analyzer = DynamicModuleAnalyzer()
    
    # Test analysis
    print("=== Module Analysis ===")
    analysis = analyzer.analyze_module('examples/simple_module_example.py')
    print(f"Module: {analysis.name}")
    print(f"Complexity: {analysis.complexity.value}")
    print(f"Functions: {len(analysis.translatable_functions)}")
    print(f"Classes: {len(analysis.translatable_classes)}")
    print(f"Effort: {analysis.estimated_effort}")
    print(f"Approach: {analysis.suggested_approach}")
    
    print("\n=== Auto Translation ===")
    # Test translation
    output_dir = Path('test_output')
    result = analyzer.translate_module('examples/simple_module_example.py', output_dir)
    
    print(f"Success: {result.success}")
    print(f"Module: {result.module_name}")
    
    if result.success:
        print(f"Generated files: {len(result.cpp_files) + len(result.header_files)}")
        if result.cpp_files:
            print("C++ files:")
            for cpp_file in result.cpp_files:
                print(f"  - {cpp_file}")
        if result.header_files:
            print("Header files:")
            for header_file in result.header_files:
                print(f"  - {header_file}")
    else:
        print(f"Error: {result.error_message}")

if __name__ == "__main__":
    test_translation()
