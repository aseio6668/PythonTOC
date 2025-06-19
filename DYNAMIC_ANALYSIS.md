# Dynamic Module Analysis System

The Python-to-C++ translator now includes a **Dynamic Module Analysis System** that can analyze and translate unknown Python modules in real-time. This addresses the challenge of maintaining mappings for the vast PyPI ecosystem.

## Overview

Instead of manually maintaining mappings for every Python package, the system can:

1. **Download** module source code from PyPI
2. **Analyze** the module's complexity and translateability 
3. **Generate** C++ code automatically for suitable modules
4. **Provide guidance** for modules that require manual intervention

## Core Components

### DynamicModuleAnalyzer

The main analyzer class that performs AST-based analysis of Python modules:

```python
from src.modules.dependency_manager import DynamicModuleAnalyzer

analyzer = DynamicModuleAnalyzer()
analysis = analyzer.analyze_module('path/to/module.py')
```

### ModuleAnalysis Results

Each analysis produces a comprehensive report:

- **Complexity Level**: SIMPLE, MODERATE, COMPLEX, or NATIVE
- **Translatable Components**: Functions and classes that can be converted
- **Dependencies**: External modules required
- **Translation Notes**: Specific features that affect translation
- **Effort Estimation**: LOW, MEDIUM, or HIGH
- **Suggested Approach**: AUTOMATIC, SEMI_AUTOMATIC, MANUAL_REVIEW, or FIND_CPP_ALTERNATIVE

### Module Complexity Classification

#### SIMPLE (Automatic Translation)
- Pure Python with basic functions and classes
- Standard control flow and data structures
- Type hints present (preferred)
- Minimal use of advanced Python features

#### MODERATE (Semi-Automatic Translation) 
- Some decorators and advanced features
- List/dict comprehensions
- Context managers (`with` statements)
- Basic generators

#### COMPLEX (Manual Review Required)
- Heavy use of metaclasses
- Dynamic code generation
- Extensive use of `eval`/`exec`
- Complex inheritance hierarchies

#### NATIVE (Find C++ Alternative)
- Contains C extensions
- Uses compiled binary modules
- Cannot be translated automatically

## Usage Examples

### Basic Module Analysis

```python
from src.modules.dependency_manager import DynamicModuleAnalyzer

# Analyze a local module
analyzer = DynamicModuleAnalyzer()
analysis = analyzer.analyze_module('examples/my_module.py')

print(f"Complexity: {analysis.complexity.value}")
print(f"Functions: {len(analysis.translatable_functions)}")
print(f"Classes: {len(analysis.translatable_classes)}")
print(f"Suggested approach: {analysis.suggested_approach}")
```

### Automatic Translation

```python
from pathlib import Path

# Translate module to C++
result = analyzer.translate_module('my_module.py', output_dir=Path('cpp_output'))

if result.success:
    print(f"Generated {len(result.cpp_files)} C++ files")
    print(f"Generated {len(result.header_files)} header files")
else:
    print(f"Translation failed: {result.error_message}")
```

### Integration with Dependency Manager

```python
from src.modules.dependency_manager import ModuleDependencyManager

# Enable dynamic analysis in dependency manager
dm = ModuleDependencyManager()
dm.initialize_dynamic_analyzer()

# Analyze unknown module
analysis = dm.analyze_unknown_module('unknown_package')

# Auto-translate if suitable
if analysis.suggested_approach == 'automatic':
    result = dm.auto_translate_module('unknown_package')
```

## Command Line Integration

The dynamic analysis system integrates with the main CLI:

```bash
# Analyze dependencies with dynamic analysis for unknown modules
python translate.py my_script.py --deps-report --analyze-unknown

# Auto-translate discovered modules
python translate.py my_script.py --auto-translate-deps

# Create complete project with dynamic translation
python translate.py my_script.py --create-project --include-unknown
```

## Real-Time PyPI Integration

For unknown modules, the system can:

1. **Search PyPI** for the module
2. **Download source** using pip
3. **Extract and analyze** the main module files
4. **Generate translation report** with recommendations

### Example: Analyzing a PyPI Package

```python
# This will download 'requests' from PyPI and analyze it
analysis = dm.analyze_unknown_module('requests')

if analysis.complexity == ModuleComplexity.SIMPLE:
    # Attempt automatic translation
    result = dm.auto_translate_module('requests')
    if result.success:
        print("Successfully translated 'requests' to C++!")
```

## Translation Strategies by Complexity

### For SIMPLE Modules
- **Direct AST translation** to C++
- **Automatic type inference** from usage patterns
- **Standard library mapping** (list → std::vector, dict → std::map)

### For MODERATE Modules  
- **Selective translation** of simple parts
- **Manual stubs** for complex features
- **Hybrid approach** keeping some Python code

### For COMPLEX Modules
- **Analysis report only** with migration guidance
- **Suggestions for C++ alternatives**
- **Break-down recommendations** into simpler components

### For NATIVE Modules
- **No translation attempted**
- **C++ library recommendations** 
- **Integration guidance** for existing C++ solutions

## Quality Estimation

The system provides translation quality estimates:

- **HIGH**: Near-perfect automatic translation expected
- **MEDIUM**: Good translation with minor manual fixes needed  
- **LOW**: Basic structure only, significant manual work required
- **FAILED**: Cannot translate, alternative approach needed

## Best Practices

### When to Use Dynamic Analysis

1. **Unknown modules** not in the built-in mapping
2. **Custom/internal modules** specific to your project
3. **Simple utility modules** that are pure Python
4. **Proof-of-concept** translations before manual implementation

### When NOT to Use

1. **Complex ML frameworks** - use dedicated C++ APIs instead
2. **Web frameworks** - consider keeping in Python or using C++ web frameworks
3. **Database ORMs** - use native C++ database libraries
4. **GUI frameworks** - use Qt, GTK, or other C++ GUI libraries

### Optimization Tips

1. **Start with simple modules** to build confidence
2. **Review generated C++ code** before integration
3. **Use type hints** in Python for better C++ translation
4. **Profile performance** of translated vs. original code
5. **Maintain hybrid architecture** when full translation isn't practical

## Future Enhancements

The dynamic analysis system can be extended with:

1. **Machine Learning models** for better complexity prediction
2. **Community database** of translation patterns
3. **Interactive translation** with user feedback
4. **Performance benchmarking** of translated code
5. **Automatic test generation** for translated modules

## Limitations

Current limitations include:

- **No runtime code generation** support
- **Limited metaclass** handling
- **Basic error handling** in generated C++
- **No automatic memory management** optimization
- **Requires manual testing** of translated code

Despite these limitations, the dynamic analysis system significantly reduces the manual effort required to translate Python codebases to C++, making the translation process more scalable and maintainable.
