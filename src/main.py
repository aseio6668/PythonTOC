#!/usr/bin/env python3
"""
Main CLI interface for Python to C++ translator
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from translator.translator import PythonToCppTranslator


def main():
    """Main entry point for the CLI application"""
    parser = argparse.ArgumentParser(
        description="Translate Python source code to C++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.py                    # Output to stdout
  python main.py input.py output.cpp        # Output to file
  python main.py input.py -o output.cpp     # Output to file (explicit)
  python main.py input.py --verbose         # Verbose output
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input Python file to translate"
    )
    
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        help="Output C++ file (optional, defaults to stdout)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        dest="output_file_alt",
        help="Output C++ file (alternative syntax)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-headers",
        action="store_true",
        help="Don't include standard C++ headers"
    )
    
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Wrap generated code in a namespace"
    )
    
    args = parser.parse_args()
    
    # Determine output file
    output_file = args.output_file or args.output_file_alt
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.suffix == ".py":
        print(f"Warning: Input file '{args.input_file}' does not have .py extension", file=sys.stderr)
    
    try:
        # Create translator instance
        translator = PythonToCppTranslator(
            include_headers=not args.no_headers,
            namespace=args.namespace,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"Translating {args.input_file}...", file=sys.stderr)
        
        # Translate the file
        cpp_code = translator.translate_file(str(input_path))
        
        # Output the result
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cpp_code)
            
            if args.verbose:
                print(f"C++ code written to {output_file}", file=sys.stderr)
        else:
            print(cpp_code)
            
    except Exception as e:
        print(f"Error during translation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
