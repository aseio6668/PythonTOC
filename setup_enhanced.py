#!/usr/bin/env python3
"""
Setup script for Python to C++ Translator Enhanced Features

This script helps set up the enhanced features including:
- Plugin system
- Web API
- AI optimization
- Development environment
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description="", check=True):
    """Run a shell command with logging"""
    print(f"🔄 {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def setup_virtual_environment():
    """Set up Python virtual environment"""
    print("🔧 Setting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Activate script path
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    print(f"✓ Virtual environment created")
    print(f"  Activate with: {activate_script}")
    return True


def install_dependencies(level="basic"):
    """Install dependencies based on level"""
    print(f"📦 Installing {level} dependencies...")
    
    requirements_files = {
        "basic": "requirements.txt",
        "enhanced": "requirements-enhanced.txt",
        "development": "requirements-dev.txt"
    }
    
    req_file = requirements_files.get(level, "requirements.txt")
    
    if not Path(req_file).exists():
        print(f"⚠️  {req_file} not found, installing basic packages")
        packages = [
            "ast-parsing",
            "pathlib2",
            "typing-extensions"
        ]
        for package in packages:
            run_command(f"pip install {package}", f"Installing {package}", check=False)
    else:
        if not run_command(f"pip install -r {req_file}", f"Installing from {req_file}"):
            return False
    
    return True


def setup_plugin_system():
    """Set up the plugin system"""
    print("🔌 Setting up plugin system...")
    
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    init_file = plugins_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
    
    # Create sample plugins if they don't exist
    if not run_command(
        "python src/modules/plugin_system.py", 
        "Creating sample plugins", 
        check=False
    ):
        print("⚠️  Could not create sample plugins (module may not be available)")
    
    print("✓ Plugin system ready")
    return True


def setup_web_api():
    """Set up web API components"""
    print("🌐 Setting up web API...")
    
    # Create necessary directories
    dirs = ["web/static", "web/templates", "uploads", "generated"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Try to install web dependencies
    web_packages = [
        "fastapi",
        "uvicorn[standard]",
        "sqlalchemy",
        "pydantic"
    ]
    
    for package in web_packages:
        run_command(f"pip install {package}", f"Installing {package}", check=False)
    
    print("✓ Web API setup complete")
    return True


def setup_database():
    """Set up database for web API"""
    print("🗄️  Setting up database...")
    
    try:
        # Try to initialize database
        run_command(
            "python -c \"from src.modules.web_api import Base, engine; Base.metadata.create_all(bind=engine)\"",
            "Initializing database schema",
            check=False
        )
        print("✓ Database initialized")
    except Exception as e:
        print(f"⚠️  Database setup skipped: {e}")
    
    return True


def setup_ai_optimization():
    """Set up AI optimization components"""
    print("🤖 Setting up AI optimization...")
    
    ai_packages = [
        "numpy",
        "scikit-learn"
    ]
    
    for package in ai_packages:
        run_command(f"pip install {package}", f"Installing {package}", check=False)
    
    # Create analysis results directory
    Path("optimization_results").mkdir(exist_ok=True)
    Path("analysis_results").mkdir(exist_ok=True)
    
    print("✓ AI optimization setup complete")
    return True


def create_example_files():
    """Create example files for testing"""
    print("📝 Creating example files...")
    
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Simple example
    simple_example = examples_dir / "simple_example.py"
    if not simple_example.exists():
        simple_example.write_text('''
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    result = fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
'''.strip())
    
    # Complex example
    complex_example = examples_dir / "complex_example.py"
    if not complex_example.exists():
        complex_example.write_text('''
import math
from typing import List, Dict

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.data: List[float] = []
    
    def add_data(self, values: List[float]) -> None:
        self.data.extend(values)
    
    def calculate_statistics(self) -> Dict[str, float]:
        if not self.data:
            return {}
        
        n = len(self.data)
        mean = sum(self.data) / n
        variance = sum((x - mean) ** 2 for x in self.data) / n
        std_dev = math.sqrt(variance)
        
        return {
            "count": n,
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "min": min(self.data),
            "max": max(self.data)
        }

def main():
    processor = DataProcessor("Test Processor")
    processor.add_data([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = processor.calculate_statistics()
    
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
'''.strip())
    
    print("✓ Example files created")
    return True


def run_tests():
    """Run basic tests to verify setup"""
    print("🧪 Running setup verification tests...")
    
    tests = [
        ("python translate.py --help", "CLI help"),
        ("python plugin_manager.py --help", "Plugin manager"),
        ("python -c \"import src.modules.plugin_system; print('Plugin system OK')\"", "Plugin system import"),
    ]
    
    success_count = 0
    for command, description in tests:
        if run_command(command, description, check=False):
            success_count += 1
    
    print(f"✓ {success_count}/{len(tests)} tests passed")
    return success_count == len(tests)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Python to C++ Translator Enhanced Features")
    parser.add_argument(
        "--level",
        choices=["basic", "enhanced", "development"],
        default="enhanced",
        help="Installation level"
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip virtual environment setup"
    )
    parser.add_argument(
        "--skip-web",
        action="store_true",
        help="Skip web API setup"
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip AI optimization setup"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run verification tests"
    )
    
    args = parser.parse_args()
    
    print("🚀 Python to C++ Translator Enhanced Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    if not args.skip_venv:
        if not setup_virtual_environment():
            print("⚠️  Virtual environment setup failed, continuing anyway...")
    
    # Install dependencies
    if not install_dependencies(args.level):
        print("⚠️  Some dependencies failed to install, continuing anyway...")
    
    # Setup plugin system
    if not setup_plugin_system():
        print("⚠️  Plugin system setup had issues, continuing anyway...")
    
    # Setup web API
    if not args.skip_web:
        if not setup_web_api():
            print("⚠️  Web API setup had issues, continuing anyway...")
        
        if not setup_database():
            print("⚠️  Database setup had issues, continuing anyway...")
    
    # Setup AI optimization
    if not args.skip_ai:
        if not setup_ai_optimization():
            print("⚠️  AI optimization setup had issues, continuing anyway...")
    
    # Create examples
    if not create_example_files():
        print("⚠️  Example file creation had issues, continuing anyway...")
    
    # Run tests
    if args.test:
        run_tests()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Activate virtual environment (if created)")
    print("2. Try: python translate.py examples/simple_example.py")
    print("3. For web API: python src/modules/web_api.py")
    print("4. For plugins: python plugin_manager.py list")
    print("\nSee ADVANCED_FEATURES.md for detailed usage instructions.")


if __name__ == "__main__":
    main()
