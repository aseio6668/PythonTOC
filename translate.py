#!/usr/bin/env python3
"""
Enhanced CLI for Python to C++ translator with comprehensive features
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from translator.translator import PythonToCppTranslator
from utils.helpers import generate_translation_report, analyze_complexity

# Import new advanced modules
from modules.test_generator import TestGenerator, test_translated_code
from modules.interactive_assistant import InteractiveTranslationAssistant
from modules.project_generator import ProjectTemplateGenerator
from modules.migration_analyzer import AdvancedMigrationAnalyzer
from modules.plugin_system import PluginManager


def main():
    """Enhanced main entry point with comprehensive features"""
    parser = argparse.ArgumentParser(
        description="Advanced Python to C++ Translator with Testing, Analysis, and Project Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Advanced Features:
  --analyze           Perform comprehensive migration analysis
  --interactive       Start interactive translation session
  --generate-tests    Generate comprehensive test suite
  --create-project    Generate modern C++ project template
  --benchmark         Run performance benchmarks
  --compare-strategies Compare different migration strategies

Examples:
  python translate.py input.py                           # Basic translation
  python translate.py input.py -o output.cpp            # Output to file
  python translate.py input.py --analyze                # Migration analysis
  python translate.py input.py --interactive            # Interactive mode
  python translate.py input.py --generate-tests         # Generate tests
  python translate.py input.py --create-project MyApp   # Create C++ project
  python translate.py input.py --benchmark              # Performance comparison
  python translate.py examples/ --batch                 # Batch translate directory
        """
    )    
    parser.add_argument(
        "input",
        type=str,
        nargs='?',
        help="Input Python file or directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output C++ file or directory"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed translation report"
    )
    
    parser.add_argument(
        "--cmake",
        action="store_true",
        help="Generate CMake build files"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory of Python files"
    )
    
    # New advanced features
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform comprehensive migration analysis"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive translation session"
    )
    
    parser.add_argument(
        "--generate-tests",
        action="store_true",
        help="Generate comprehensive test suite for translated code"
    )
    
    parser.add_argument(
        "--create-project",
        type=str,
        metavar="PROJECT_NAME",
        help="Generate modern C++ project template"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks comparing Python vs C++"
    )
    
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare different migration strategies"
    )
    
    parser.add_argument(
        "--template-type",
        choices=["minimal", "standard", "enterprise"],
        default="standard",
        help="Project template type (for --create-project)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated files"
    )
    
    # Plugin system arguments
    parser.add_argument(
        "--load-plugins",
        action="store_true",
        help="Load and use available plugins"
    )
    
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available plugins and exit"
    )
    
    parser.add_argument(
        "--plugin-dir",
        type=str,
        default="plugins",
        help="Directory containing plugins"
    )

    args = parser.parse_args()
    
    # Initialize plugin manager
    plugin_manager = None
    if args.load_plugins or args.list_plugins:
        print("Initializing plugin system...")
        plugin_manager = PluginManager(args.plugin_dir)
        plugin_manager.load_plugins()
    
    # Handle plugin listing
    if args.list_plugins:
        plugins = plugin_manager.list_plugins()
        if plugins:
            print(f"\nLoaded {len(plugins)} plugins:")
            for name, info in plugins.items():                print(f"  - {name} ({info['type']}): {info['description']}")
        else:
            print("No plugins found.")
        return 0
    
    # Check if input is provided for other operations
    if not args.input:
        print("Error: Input file or directory is required")
        parser.print_help()
        return 1
    
    input_path = Path(args.input)
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        return 1
    
    try:
        # Handle different modes of operation
        if args.create_project:
            return handle_project_generation(args)
        elif args.analyze:
            return handle_migration_analysis(args, input_path)
        elif args.interactive:
            return handle_interactive_mode(args, input_path)
        elif args.generate_tests:
            return handle_test_generation(args, input_path)
        elif args.benchmark:
            return handle_benchmarking(args, input_path)
        elif args.compare_strategies:
            return handle_strategy_comparison(args, input_path)
        elif args.batch:
            return handle_batch_translation(args, input_path)
        else:
            return handle_standard_translation(args, input_path)
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_project_generation(args) -> int:
    """Handle C++ project template generation"""
    print(f"🏗️ Generating C++ project: {args.create_project}")
    
    generator = ProjectTemplateGenerator()
    output_dir = Path(args.output_dir)
    
    try:
        structure = generator.create_modern_cpp_project(
            args.create_project, 
            output_dir, 
            args.template_type
        )
        
        print(f"✅ Project generated successfully!")
        print(f"📁 Location: {structure.root_dir}")
        print(f"🔨 Build files: {len(structure.build_files)} created")
        print(f"📚 Config files: {len(structure.config_files)} created")
        print()
        print("Next steps:")
        print(f"  cd {structure.root_dir}")
        print("  mkdir build && cd build")
        print("  cmake ..")
        print("  cmake --build .")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to generate project: {e}")
        return 1


def handle_migration_analysis(args, input_path: Path) -> int:
    """Handle comprehensive migration analysis"""
    print(f"🔍 Analyzing migration complexity: {input_path}")
    
    if input_path.is_dir():
        print("❌ Directory analysis not yet supported. Please specify a single Python file.")
        return 1
    
    analyzer = AdvancedMigrationAnalyzer()
    
    try:
        report = analyzer.analyze_migration(input_path)
        
        print(f"\n📊 Migration Analysis Results:")
        print(f"   Overall Complexity: {report.overall_complexity.value.title()}")
        print(f"   Estimated Effort: {report.estimated_effort_hours[0]}-{report.estimated_effort_hours[1]} hours")
        print(f"   Risk Level: {report.risk_assessment.value.title()}")
        print(f"   Recommended Strategy: {report.recommended_strategy.value.replace('_', ' ').title()}")
        print()
        print(f"📈 Code Metrics:")
        print(f"   Functions: {len(report.functions)}")
        print(f"   Classes: {len(report.classes)}")
        print(f"   Dependencies: {len(report.dependencies)}")
        print(f"   Detected Patterns: {len(report.patterns)}")
        
        # Generate detailed report
        output_dir = Path(args.output_dir)
        report_file = output_dir / f"{input_path.stem}_migration_analysis.md"
        analyzer.generate_detailed_report(report, report_file)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


def handle_interactive_mode(args, input_path: Path) -> int:
    """Handle interactive translation session"""
    print(f"🎯 Starting interactive translation session: {input_path}")
    
    if input_path.is_dir():
        print("❌ Interactive mode requires a single Python file.")
        return 1
    
    assistant = InteractiveTranslationAssistant()
    
    try:
        session_id = assistant.start_interactive_session(input_path)
        
        print(f"✅ Interactive session started: {session_id}")
        
        # Get initial suggestions
        suggestions = assistant.get_suggestions(session_id)
        print(f"\n💡 Found {len(suggestions)} suggestions:")
        
        for i, suggestion in enumerate(suggestions[:5], 1):
            icon = "🔧" if suggestion.auto_fixable else "⚠️"
            print(f"  {i}. {icon} [{suggestion.severity.value.upper()}] {suggestion.message}")
            print(f"     Line {suggestion.line_number}: {suggestion.explanation}")
        
        # Get translation plan
        plan = assistant.generate_translation_plan(session_id)
        print(f"\n📋 Translation Plan:")
        print(f"   Estimated Time: {plan['overview']['estimated_time']}")
        print(f"   Difficulty: {plan['overview']['difficulty']}")
        print(f"   Key Challenges: {', '.join(plan['overview']['key_challenges'])}")
        
        print(f"\n📊 Migration Phases:")
        for phase in plan['phases']:
            print(f"   Phase {phase['phase']}: {phase['name']} ({phase['estimated_time']})")
        
        # Generate session report
        output_dir = Path(args.output_dir)
        report_file = output_dir / f"{input_path.stem}_interactive_session.md"
        assistant.export_session_report(session_id, report_file)
        
        print(f"\n📄 Session report saved: {report_file}")
        
        assistant.cleanup_session(session_id)
        return 0
        
    except Exception as e:
        print(f"❌ Interactive session failed: {e}")
        return 1


def handle_test_generation(args, input_path: Path) -> int:
    """Handle comprehensive test generation"""
    print(f"🧪 Generating comprehensive test suite: {input_path}")
    
    if input_path.is_dir():
        print("❌ Test generation requires a single Python file.")
        return 1
    
    # First, we need to translate the Python code
    output_file = Path(args.output) if args.output else input_path.with_suffix('.cpp')
    
    print(f"🔄 Translating Python to C++...")
    translator = PythonToCppTranslator(verbose=args.verbose)
    
    try:
        cpp_code = translator.translate_file(str(input_path))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        
        print(f"✅ Translation completed: {output_file}")
        
        # Generate comprehensive tests
        output_dir = Path(args.output_dir) / "tests"
        output_dir.mkdir(exist_ok=True)
        
        print(f"🧪 Generating test suite...")
        test_report = test_translated_code(input_path, output_file, output_dir)
        
        print(f"✅ Test suite generated!")
        print(f"📁 Test files location: {output_dir}")
        print(f"📄 Test report: {output_dir / f'{input_path.stem}_test_report.md'}")
        print(f"🔧 Google Test file: {output_dir / f'{input_path.stem}_gtest.cpp'}")
        
        # Show test summary
        if "Passed:" in test_report:
            lines = test_report.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ["Passed:", "Failed:", "Average Speedup:", "Total Tests:"]):
                    print(f"   {line.strip()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
        return 1


def handle_benchmarking(args, input_path: Path) -> int:
    """Handle performance benchmarking"""
    print(f"🏃 Running performance benchmarks: {input_path}")
    
    if input_path.is_dir():
        print("❌ Benchmarking requires a single Python file.")
        return 1
    
    # Import benchmarking module
    from modules.performance_benchmarker import PerformanceBenchmarker, benchmark_translated_code
    
    # First translate the code
    output_file = Path(args.output) if args.output else input_path.with_suffix('.cpp')
    
    print(f"🔄 Translating Python to C++...")
    translator = PythonToCppTranslator(verbose=args.verbose)
    
    try:
        cpp_code = translator.translate_file(str(input_path))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        
        print(f"✅ Translation completed: {output_file}")
        
        # Run benchmarks
        print(f"⏱️ Running performance benchmarks...")
        benchmark_report = benchmark_translated_code(input_path, output_file)
        
        print(f"✅ Benchmarking completed!")
        
        # Parse and display key metrics
        lines = benchmark_report.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ["Speedup Factor:", "Memory Improvement:", "Overall Score:"]):
                print(f"   {line.strip()}")
        
        # Save benchmark report
        output_dir = Path(args.output_dir)
        benchmark_file = output_dir / f"{input_path.stem}_benchmark_report.md"
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            f.write(benchmark_report)
        
        print(f"📄 Benchmark report saved: {benchmark_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return 1


def handle_strategy_comparison(args, input_path: Path) -> int:
    """Handle migration strategy comparison"""
    print(f"🔄 Comparing migration strategies: {input_path}")
    
    if input_path.is_dir():
        print("❌ Strategy comparison requires a single Python file.")
        return 1
    
    analyzer = AdvancedMigrationAnalyzer()
    
    try:
        comparison = analyzer.compare_migration_strategies(input_path)
        
        print(f"\n📊 Migration Strategy Comparison:")
        print(f"   File: {input_path}")
        print(f"   Recommended: {comparison['recommended'].replace('_', ' ').title()}")
        print()
        
        # Show comparison table
        print("Strategy Comparison:")
        print("=" * 80)
        print(f"{'Strategy':<25} {'Effort (hrs)':<12} {'Accuracy':<10} {'Performance':<12}")
        print("-" * 80)
        
        for strategy, details in comparison['strategies'].items():
            strategy_name = strategy.value.replace('_', ' ').title()
            effort_range = f"{details['effort'][0]}-{details['effort'][1]}"
            accuracy = f"{details['accuracy']:.0%}"
            performance = details['performance']
            
            print(f"{strategy_name:<25} {effort_range:<12} {accuracy:<10} {performance:<12}")
        
        print()
        
        # Show analysis summary
        analysis = comparison['analysis']
        print(f"Analysis Summary:")
        print(f"   Overall Complexity: {analysis.overall_complexity.value.title()}")
        print(f"   Risk Level: {analysis.risk_assessment.value.title()}")
        print(f"   Functions: {len(analysis.functions)}")
        print(f"   Classes: {len(analysis.classes)}")
        print(f"   Dependencies: {len(analysis.dependencies)}")
        
        # Save comparison report
        output_dir = Path(args.output_dir)
        comparison_file = output_dir / f"{input_path.stem}_strategy_comparison.json"
        
        # Make comparison serializable
        serializable_comparison = {
            "recommended": comparison['recommended'],
            "analysis_summary": {
                "complexity": analysis.overall_complexity.value,
                "risk_level": analysis.risk_assessment.value,
                "estimated_effort": analysis.estimated_effort_hours,
                "functions": len(analysis.functions),
                "classes": len(analysis.classes),
                "dependencies": len(analysis.dependencies)
            },
            "strategies": {
                strategy.value: {
                    "description": details["description"],
                    "effort_hours": details["effort"],
                    "accuracy": details["accuracy"],
                    "maintenance": details["maintenance"],
                    "performance": details["performance"]
                }
                for strategy, details in comparison['strategies'].items()
            }
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_comparison, f, indent=2)
        
        print(f"\n📄 Strategy comparison saved: {comparison_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Strategy comparison failed: {e}")
        return 1


def handle_batch_translation(args, input_path: Path) -> int:
    """Handle batch translation of multiple files"""
    print(f"📦 Batch processing directory: {input_path}")
    
    if not input_path.is_dir():
        print("❌ Batch mode requires a directory input.")
        return 1
    
    # Find all Python files
    python_files = list(input_path.rglob("*.py"))
    
    if not python_files:
        print(f"❌ No Python files found in {input_path}")
        return 1
    
    print(f"📁 Found {len(python_files)} Python files")
    
    # Set up output directory
    output_dir = Path(args.output_dir) / "cpp_output"
    output_dir.mkdir(exist_ok=True)
    
    translator = PythonToCppTranslator(verbose=args.verbose)
    
    success_count = 0
    total_files = len(python_files)
    
    for i, py_file in enumerate(python_files, 1):
        print(f"🔄 [{i}/{total_files}] Processing {py_file.name}...")
        
        try:
            cpp_code = translator.translate_file(str(py_file))
            
            # Create output file
            output_file = output_dir / f"{py_file.stem}.cpp"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cpp_code)
            
            # Generate report if requested
            if args.report:
                report = generate_translation_report(py_file, output_file)
                report_file = output_dir / f"{py_file.stem}.translation_report.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
            
            success_count += 1
            print(f"✅ {py_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"❌ Failed to process {py_file.name}: {e}")
    
    print(f"\n📊 Batch processing completed:")
    print(f"   Successful: {success_count}/{total_files}")
    print(f"   Output directory: {output_dir}")
    
    # Generate CMake files if requested
    if args.cmake:
        print(f"🔨 Generating CMake files...")
        try:
            generate_cmake_files(output_dir, [f.stem for f in python_files])
            print(f"✅ CMake files generated")
        except Exception as e:
            print(f"❌ CMake generation failed: {e}")
    
    return 0 if success_count > 0 else 1


def handle_standard_translation(args, input_path: Path) -> int:
    """Handle standard single-file translation"""
    print(f"🔄 Translating Python to C++: {input_path}")
    
    if input_path.is_dir():
        print("❌ Standard translation requires a single Python file. Use --batch for directories.")
        return 1
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_path.with_suffix('.cpp')
    
    translator = PythonToCppTranslator(verbose=args.verbose)
    
    try:
        cpp_code = translator.translate_file(str(input_path))
        
        if args.output:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cpp_code)
            print(f"✅ Translation completed: {output_file}")
        else:
            print(cpp_code)
        
        # Generate report if requested
        if args.report:
            report = generate_translation_report(input_path, output_file)
            report_file = output_file.with_suffix('.translation_report.md')
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"📄 Translation report: {report_file}")
        
        # Generate CMake files if requested
        if args.cmake:
            output_dir = output_file.parent
            generate_cmake_files(output_dir, [output_file.stem])
            print(f"🔨 CMake files generated in {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return 1


def generate_cmake_files(output_dir: Path, file_stems: list):
    """Generate CMake build files"""
    cmake_content = f"""cmake_minimum_required(VERSION 3.12)
project(translated_python_code)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
{chr(10).join(f'    {stem}.cpp' for stem in file_stems)}
)

# Create executable for each source file
"""
    
    for stem in file_stems:
        cmake_content += f"""
add_executable({stem} {stem}.cpp)

# Compiler options
target_compile_options({stem} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
"""
    
    cmake_file = output_dir / "CMakeLists.txt"
    with open(cmake_file, 'w', encoding='utf-8') as f:
        f.write(cmake_content)


if __name__ == "__main__":
    sys.exit(main())
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate translation report"
    )
    
    parser.add_argument(
        "--cmake",
        action="store_true",
        help="Generate CMakeLists.txt file"
    )
    
    parser.add_argument(
        "--namespace",
        type=str,
        help="Wrap generated code in a namespace"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process directory of Python files"
    )
    
    parser.add_argument(
        "--no-headers",
        action="store_true",
        help="Don't include standard headers"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--dependencies",
        action="store_true",
        help="Enable automatic dependency analysis and management"
    )
    
    parser.add_argument(
        "--deps-report",
        action="store_true",
        help="Generate detailed dependency analysis report"
    )
    
    parser.add_argument(
        "--create-project",
        action="store_true",
        help="Create complete C++ project with dependency management"
    )
    
    parser.add_argument(
        "--download-deps",
        action="store_true",
        help="Download and convert pure Python dependencies"
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = load_config(args.config) if args.config else {}
    
    # Create translator with configuration
    translator = PythonToCppTranslator(
        include_headers=not args.no_headers,
        namespace=args.namespace or config.get('namespace'),
        verbose=args.verbose,
        manage_dependencies=args.dependencies or args.deps_report or args.create_project,
        output_dir=Path(args.output).parent if args.output else None
    )
    
    input_path = Path(args.input)
    
    try:
        if args.batch and input_path.is_dir():
            translate_directory(translator, input_path, args)
        elif input_path.is_file():
            translate_single_file(translator, input_path, args)
        else:
            print(f"Error: '{args.input}' is not a valid file or directory", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during translation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config file '{config_path}': {e}", file=sys.stderr)
        return {}


def translate_single_file(translator: PythonToCppTranslator, input_path: Path, args):
    """Translate a single Python file"""
    if args.verbose:
        print(f"Translating {input_path}...")
    
    # Translate the file
    cpp_code = translator.translate_file(str(input_path))
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.cpp')
    
    # Write output
    if args.output and args.output != '-':
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        
        if args.verbose:
            print(f"C++ code written to {output_path}")
    else:
        print(cpp_code)
    
    # Generate additional files
    if args.cmake:
        generate_cmake_file(input_path, output_path, args.namespace)
    
    if args.report:
        generate_report(translator, input_path, args)
    
    # Generate dependency report
    if args.deps_report:
        generate_dependency_report(translator, input_path, args)
    
    # Create complete project
    if args.create_project:
        create_complete_project(translator, input_path, args)


def translate_directory(translator: PythonToCppTranslator, input_dir: Path, args):
    """Translate all Python files in a directory"""
    python_files = list(input_dir.glob("**/*.py"))
    
    if not python_files:
        print(f"No Python files found in {input_dir}")
        return
    
    if args.verbose:
        print(f"Found {len(python_files)} Python files to translate")
    
    output_dir = Path(args.output) if args.output else input_dir / "cpp_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    translated_files = []
    
    for py_file in python_files:
        try:
            if args.verbose:
                print(f"Translating {py_file.relative_to(input_dir)}...")
            
            cpp_code = translator.translate_file(str(py_file))
            
            # Create corresponding C++ file path
            relative_path = py_file.relative_to(input_dir)
            cpp_file = output_dir / relative_path.with_suffix('.cpp')
            cpp_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cpp_file, 'w', encoding='utf-8') as f:
                f.write(cpp_code)
            
            translated_files.append(cpp_file)
            
        except Exception as e:
            print(f"Error translating {py_file}: {e}", file=sys.stderr)
    
    if args.verbose:
        print(f"Successfully translated {len(translated_files)} files")
    
    # Generate project-wide CMake file
    if args.cmake:
        generate_project_cmake_file(translated_files, output_dir, args.namespace)


def generate_cmake_file(input_file: Path, output_file: Path, namespace: Optional[str]):
    """Generate CMakeLists.txt for a single file"""
    project_name = input_file.stem
    
    cmake_content = f"""cmake_minimum_required(VERSION 3.12)
project({project_name})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable({project_name} {output_file.name})

# Compiler options
target_compile_options({project_name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
"""
    
    cmake_file = output_file.parent / "CMakeLists.txt"
    with open(cmake_file, 'w') as f:
        f.write(cmake_content)
    
    print(f"Generated CMakeLists.txt at {cmake_file}")


def generate_project_cmake_file(cpp_files: list, output_dir: Path, namespace: Optional[str]):
    """Generate CMakeLists.txt for multiple files"""
    project_name = output_dir.name
    
    # Find main file or create one
    main_file = None
    for cpp_file in cpp_files:
        with open(cpp_file, 'r') as f:
            content = f.read()
            if 'int main(' in content:
                main_file = cpp_file
                break
    
    if not main_file:
        main_file = cpp_files[0] if cpp_files else None
    
    if main_file:
        source_files = [f.name for f in cpp_files]
        source_files_str = ' '.join(source_files)
        
        cmake_content = f"""cmake_minimum_required(VERSION 3.12)
project({project_name})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES {source_files_str})

# Add executable
add_executable({project_name} ${{SOURCES}})

# Compiler options
target_compile_options({project_name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
"""
        
        cmake_file = output_dir / "CMakeLists.txt"
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        
        print(f"Generated project CMakeLists.txt at {cmake_file}")


def generate_report(translator: PythonToCppTranslator, input_file: Path, args):
    """Generate translation report"""
    # Get parser and type information
    parser_info = translator.get_parser_info()
    type_info = translator.get_type_info()
    
    # Analyze complexity
    with open(input_file, 'r') as f:
        code = f.read()
    
    import ast
    tree = ast.parse(code)
    complexity = analyze_complexity(tree)
    
    # Generate report
    report = generate_translation_report(parser_info, type_info, complexity)
    
    # Write report
    report_file = input_file.with_suffix('.translation_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Translation report written to {report_file}")


def generate_dependency_report(translator: PythonToCppTranslator, input_file: Path, args):
    """Generate dependency analysis report"""
    if not translator.dependency_manager:
        print("Dependency management not enabled")
        return
      # Generate dependency report
    dep_report = translator.generate_dependency_report(str(input_file))
    
    if dep_report:
        # Write report
        report_file = input_file.with_suffix('.dependencies.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(dep_report)
        
        print(f"Dependency report written to {report_file}")
    else:
        print("No dependencies found or dependency management not enabled")


def create_complete_project(translator: PythonToCppTranslator, input_file: Path, args):
    """Create complete C++ project with dependency management"""
    if not translator.dependency_manager:
        print("Dependency management not enabled")
        return
    
    project_name = args.output or input_file.stem
    if isinstance(project_name, str) and project_name.endswith('.cpp'):
        project_name = Path(project_name).stem
    
    # Create project directory
    project_dir = translator.create_module_project(str(input_file), str(project_name))
    
    if project_dir:
        # Translate main file to the project directory
        cpp_code = translator.translate_file(str(input_file))
        main_cpp = project_dir / "main.cpp"
        
        with open(main_cpp, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        
        # Generate README for the project
        readme_content = f"""# {project_name}

Generated C++ project from Python source: {input_file.name}

## Building

### With CMake:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### With vcpkg (if vcpkg.json exists):
```bash
vcpkg install
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
cmake --build .
```

### With Conan (if conanfile.txt exists):
```bash
mkdir build
cd build
conan install ..
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

## Dependencies

See the dependency report and CMakeLists.txt for required libraries.
"""
        
        with open(project_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print(f"Complete C++ project created at {project_dir}")
        print(f"Main file: {main_cpp}")
    else:
        print("Failed to create project")


if __name__ == "__main__":
    main()
