"""
Plugin Management CLI Extension

Provides command-line interface for managing plugins in the Python to C++ translator.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.plugin_system import PluginManager


def main():
    """Main entry point for plugin management CLI"""
    parser = argparse.ArgumentParser(
        description="Plugin Management for Python to C++ Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plugin_manager.py list                           # List all plugins
  python plugin_manager.py create translation MyPlugin   # Create plugin template
  python plugin_manager.py test                           # Test plugin system
  python plugin_manager.py info numpy_plugin             # Get plugin info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List plugins command
    list_parser = subparsers.add_parser('list', help='List all loaded plugins')
    list_parser.add_argument(
        '--type',
        choices=['translation', 'library', 'optimization', 'analysis'],
        help='Filter by plugin type'
    )
    
    # Create plugin template command
    create_parser = subparsers.add_parser('create', help='Create a new plugin template')
    create_parser.add_argument(
        'plugin_type',
        choices=['translation', 'library', 'optimization', 'analysis'],
        help='Type of plugin to create'
    )
    create_parser.add_argument('name', help='Name of the plugin')
    create_parser.add_argument(
        '-o', '--output',
        help='Output file path (default: plugins/{name}_plugin.py)'
    )
    
    # Plugin info command
    info_parser = subparsers.add_parser('info', help='Get detailed plugin information')
    info_parser.add_argument('plugin_name', help='Name of the plugin')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the plugin system')
    test_parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample plugins for testing'
    )
    
    # Install plugin command
    install_parser = subparsers.add_parser('install', help='Install a plugin from file')
    install_parser.add_argument('plugin_file', help='Path to plugin file')
    
    # Remove plugin command
    remove_parser = subparsers.add_parser('remove', help='Remove a plugin')
    remove_parser.add_argument('plugin_name', help='Name of the plugin to remove')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize plugin manager
    manager = PluginManager()
    
    try:
        if args.command == 'list':
            list_plugins(manager, args.type)
        elif args.command == 'create':
            create_plugin_template(manager, args.plugin_type, args.name, args.output)
        elif args.command == 'info':
            show_plugin_info(manager, args.plugin_name)
        elif args.command == 'test':
            test_plugin_system(manager, args.create_samples)
        elif args.command == 'install':
            install_plugin(manager, args.plugin_file)
        elif args.command == 'remove':
            remove_plugin(args.plugin_name)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        manager.cleanup()


def list_plugins(manager: PluginManager, plugin_type: Optional[str] = None):
    """List all loaded plugins"""
    print("Loading plugins...")
    manager.load_plugins()
    
    plugins = manager.list_plugins()
    
    if not plugins:
        print("No plugins found.")
        return
    
    # Filter by type if specified
    if plugin_type:
        plugins = {
            name: info for name, info in plugins.items()
            if plugin_type in info['type'].lower()
        }
    
    print(f"\nFound {len(plugins)} plugin(s):")
    print("-" * 60)
    
    for name, info in plugins.items():
        print(f"Name: {name}")
        print(f"Type: {info['type']}")
        print(f"Version: {info['version']}")
        print(f"Description: {info['description']}")
        print(f"Author: {info['author']}")
        if info['dependencies']:
            print(f"Dependencies: {', '.join(info['dependencies'])}")
        print("-" * 60)


def create_plugin_template(manager: PluginManager, plugin_type: str, name: str, output: Optional[str] = None):
    """Create a new plugin template"""
    if not output:
        output = f"plugins/{name.lower()}_plugin.py"
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    template = manager.create_plugin_template(plugin_type, name)
    
    output_path.write_text(template, encoding='utf-8')
    
    print(f"Created {plugin_type} plugin template: {output_path}")
    print(f"\nTo use this plugin:")
    print(f"1. Edit {output_path} to implement your plugin logic")
    print(f"2. Place it in the 'plugins' directory")
    print(f"3. Run 'python plugin_manager.py list' to verify it loads")


def show_plugin_info(manager: PluginManager, plugin_name: str):
    """Show detailed information about a specific plugin"""
    print("Loading plugins...")
    manager.load_plugins()
    
    plugins = manager.list_plugins()
    
    if plugin_name not in plugins:
        print(f"Plugin '{plugin_name}' not found.")
        print(f"Available plugins: {', '.join(plugins.keys())}")
        return
    
    info = plugins[plugin_name]
    plugin_instance = manager.plugin_registry[plugin_name]
    
    print(f"\nPlugin Information: {plugin_name}")
    print("=" * 50)
    print(f"Type: {info['type']}")
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    print(f"Author: {info['author']}")
    print(f"Dependencies: {info['dependencies'] or 'None'}")
    
    # Type-specific information
    if hasattr(plugin_instance, 'get_supported_modules'):
        modules = plugin_instance.get_supported_modules()
        print(f"Supported Modules: {', '.join(modules)}")
    
    if hasattr(plugin_instance, 'get_cpp_dependencies'):
        cpp_deps = plugin_instance.get_cpp_dependencies()
        if cpp_deps:
            print("C++ Dependencies:")
            for dep, version in cpp_deps.items():
                print(f"  - {dep}: {version}")
    
    if hasattr(plugin_instance, 'get_priority'):
        priority = plugin_instance.get_priority()
        print(f"Priority: {priority}")


def test_plugin_system(manager: PluginManager, create_samples: bool = False):
    """Test the plugin system functionality"""
    if create_samples:
        print("Creating sample plugins...")
        from modules.plugin_system import create_sample_plugins
        create_sample_plugins()
        print("Sample plugins created in 'plugins' directory")
    
    print("\nTesting plugin system...")
    manager.load_plugins()
    
    plugins = manager.list_plugins()
    print(f"Loaded {len(plugins)} plugins")
    
    # Test library support
    print("\nTesting library support:")
    test_modules = ['numpy', 'pandas', 'requests', 'unknown_module']
    for module in test_modules:
        plugin = manager.get_library_support(module)
        if plugin:
            print(f"  ✓ {module}: Supported by {plugin.name}")
        else:
            print(f"  ✗ {module}: No support found")
    
    # Test optimization
    print("\nTesting optimization:")
    test_code = """
for i in range(10):
    print(i)
    
for j in range(5, 15):
    result = j * 2
"""
    
    optimized = manager.optimize_code(test_code, {})
    if optimized != test_code:
        print("  ✓ Code optimization applied")
        print(f"  Original lines: {len(test_code.splitlines())}")
        print(f"  Optimized lines: {len(optimized.splitlines())}")
    else:
        print("  ✗ No optimizations applied")
    
    # Test analysis
    print("\nTesting analysis:")
    import ast
    try:
        ast_tree = ast.parse(test_code)
        analysis = manager.run_analysis(test_code, ast_tree)
        if analysis:
            print(f"  ✓ Analysis completed: {len(analysis)} analysis results")
            for name, result in analysis.items():
                print(f"    - {name}: {len(result)} metrics")
        else:
            print("  ✗ No analysis plugins available")
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
    
    print("\nPlugin system test completed!")


def install_plugin(manager: PluginManager, plugin_file: str):
    """Install a plugin from a file"""
    plugin_path = Path(plugin_file)
    
    if not plugin_path.exists():
        print(f"Plugin file not found: {plugin_file}")
        return
    
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    target_path = plugins_dir / plugin_path.name
    
    # Copy the plugin file
    import shutil
    shutil.copy2(plugin_path, target_path)
    
    print(f"Plugin installed: {target_path}")
    
    # Test loading the plugin
    manager.load_plugins()
    plugins = manager.list_plugins()
    
    # Find newly installed plugin
    plugin_name = plugin_path.stem
    matching_plugins = [name for name in plugins.keys() if plugin_name.lower() in name.lower()]
    
    if matching_plugins:
        print(f"Successfully loaded plugin: {matching_plugins[0]}")
    else:
        print("Warning: Plugin file copied but may not have loaded correctly")


def remove_plugin(plugin_name: str):
    """Remove a plugin"""
    plugins_dir = Path("plugins")
    
    # Find plugin file
    plugin_files = list(plugins_dir.glob(f"*{plugin_name.lower()}*.py"))
    
    if not plugin_files:
        print(f"Plugin file not found for: {plugin_name}")
        return
    
    for plugin_file in plugin_files:
        plugin_file.unlink()
        print(f"Removed plugin file: {plugin_file}")
    
    print(f"Plugin '{plugin_name}' removed successfully")


if __name__ == "__main__":
    main()
