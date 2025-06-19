#!/usr/bin/env python3
"""Test script to verify ML migration guide integration"""

from src.modules.dependency_manager import ModuleDependencyManager

def test_ml_guide():
    """Test ML migration guide generation"""
    dm = ModuleDependencyManager()
    
    # Analyze PyTorch example dependencies
    print("Analyzing PyTorch example dependencies...")
    deps = dm.analyze_dependencies('examples/pytorch_example.py')
    print(f"Found {len(deps)} dependencies: {[dep.name for dep in deps]}")
    
    # Generate suggestions
    suggestions = dm.suggest_cpp_alternatives(deps)
    
    # Generate complete report with ML guide
    report = dm.generate_dependency_report(deps, suggestions)
    
    # Save to file
    with open('pytorch_ml_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Complete dependency report with ML migration guide saved to 'pytorch_ml_report.md'")
    
    # Test that ML guide is present
    if 'Machine Learning Framework Migration Guide' in report:
        print("✅ ML Migration Guide is included in the report")
    else:
        print("❌ ML Migration Guide is missing from the report")
    
    # Check for specific PyTorch guidance
    if 'LibTorch' in report:
        print("✅ PyTorch-specific guidance found")
    else:
        print("❌ PyTorch-specific guidance missing")

if __name__ == "__main__":
    test_ml_guide()
