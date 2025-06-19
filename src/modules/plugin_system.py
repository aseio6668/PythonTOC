"""
Plugin Architecture System for Python to C++ Translator

This module provides a flexible plugin system that allows users to:
- Create custom translation patterns
- Add support for new Python libraries
- Implement domain-specific optimizations
- Extend the translator with new features

Plugin Types:
1. Translation Plugins - Custom AST node translation rules
2. Library Plugins - Support for specific Python libraries
3. Optimization Plugins - Code optimization patterns
4. Analysis Plugins - Custom code analysis features
"""

import os
import sys
import json
import importlib.util
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PluginBase(ABC):
    """Base class for all plugins"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "Base plugin"
        self.author = "Unknown"
        self.dependencies = []
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass


class TranslationPlugin(PluginBase):
    """Plugin for custom AST node translation"""
    
    @abstractmethod
    def can_translate(self, node: ast.AST) -> bool:
        """Check if this plugin can translate the given AST node"""
        pass
    
    @abstractmethod
    def translate(self, node: ast.AST, context: Dict[str, Any]) -> str:
        """Translate the AST node to C++ code"""
        pass
    
    def get_priority(self) -> int:
        """Return plugin priority (higher = more priority)"""
        return 0


class LibraryPlugin(PluginBase):
    """Plugin for Python library support"""
    
    @abstractmethod
    def get_supported_modules(self) -> List[str]:
        """Return list of Python modules this plugin supports"""
        pass
    
    @abstractmethod
    def get_cpp_dependencies(self) -> Dict[str, str]:
        """Return C++ dependencies and their versions"""
        pass
    
    @abstractmethod
    def translate_import(self, module_name: str, alias: Optional[str] = None) -> str:
        """Translate Python import to C++ includes"""
        pass
    
    @abstractmethod
    def translate_function_call(self, func_name: str, args: List[str], kwargs: Dict[str, str]) -> str:
        """Translate function call to C++ equivalent"""
        pass


class OptimizationPlugin(PluginBase):
    """Plugin for code optimization"""
    
    @abstractmethod
    def can_optimize(self, code: str, metadata: Dict[str, Any]) -> bool:
        """Check if this plugin can optimize the given code"""
        pass
    
    @abstractmethod
    def optimize(self, code: str, metadata: Dict[str, Any]) -> str:
        """Apply optimization to the code"""
        pass
    
    def get_optimization_description(self) -> str:
        """Return description of the optimization"""
        return "Generic optimization"


class AnalysisPlugin(PluginBase):
    """Plugin for custom code analysis"""
    
    @abstractmethod
    def analyze(self, source_code: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """Analyze the source code and return results"""
        pass
    
    def get_analysis_name(self) -> str:
        """Return name of the analysis"""
        return "Generic Analysis"


class PluginManager:
    """Manager for loading and executing plugins"""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = plugins_dir or "plugins"
        self.translation_plugins: List[TranslationPlugin] = []
        self.library_plugins: List[LibraryPlugin] = []
        self.optimization_plugins: List[OptimizationPlugin] = []
        self.analysis_plugins: List[AnalysisPlugin] = []
        self.plugin_registry: Dict[str, PluginBase] = {}
        
        # Create plugins directory if it doesn't exist
        Path(self.plugins_dir).mkdir(exist_ok=True)
          # Create __init__.py if it doesn't exist
        init_file = Path(self.plugins_dir) / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
    
    def load_plugins(self) -> None:
        """Load all plugins from the plugins directory"""
        try:
            # Add plugins directory to Python path
            plugins_path = Path(self.plugins_dir).absolute()
            if str(plugins_path) not in sys.path:
                sys.path.insert(0, str(plugins_path))
            
            # Find all Python files in plugins directory
            plugin_files = list(Path(self.plugins_dir).glob("*.py"))
            plugin_files = [f for f in plugin_files if f.name != "__init__.py"]
            
            for plugin_file in plugin_files:
                self._load_plugin_file(plugin_file)
                
            logger.info(f"Loaded {len(self.plugin_registry)} plugins")
            
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
    
    def _load_plugin_file(self, plugin_file: Path) -> None:
        """Load a single plugin file"""
        try:
            module_name = plugin_file.stem
            
            # Add the plugin system module to sys.modules so plugins can import it
            current_module = sys.modules[__name__]
            sys.modules['plugin_system'] = current_module
            
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, PluginBase) and 
                        obj != PluginBase and 
                        not inspect.isabstract(obj)):
                        
                        plugin_instance = obj()
                        if plugin_instance.initialize():
                            self._register_plugin(plugin_instance)
                            logger.info(f"Loaded plugin: {plugin_instance.name}")
                        else:
                            logger.warning(f"Failed to initialize plugin: {plugin_instance.name}")
                            
        except Exception as e:
            logger.error(f"Error loading plugin file {plugin_file}: {e}")
    
    def _register_plugin(self, plugin: PluginBase) -> None:
        """Register a plugin instance"""
        self.plugin_registry[plugin.name] = plugin
        
        # Add to appropriate category
        if isinstance(plugin, TranslationPlugin):
            self.translation_plugins.append(plugin)
            # Sort by priority
            self.translation_plugins.sort(key=lambda p: p.get_priority(), reverse=True)
        elif isinstance(plugin, LibraryPlugin):
            self.library_plugins.append(plugin)
        elif isinstance(plugin, OptimizationPlugin):
            self.optimization_plugins.append(plugin)
        elif isinstance(plugin, AnalysisPlugin):
            self.analysis_plugins.append(plugin)
    
    def translate_node(self, node: ast.AST, context: Dict[str, Any]) -> Optional[str]:
        """Try to translate an AST node using registered plugins"""
        for plugin in self.translation_plugins:
            try:
                if plugin.can_translate(node):
                    return plugin.translate(node, context)
            except Exception as e:
                logger.error(f"Error in translation plugin {plugin.name}: {e}")
        return None
    
    def get_library_support(self, module_name: str) -> Optional[LibraryPlugin]:
        """Get library plugin that supports the given module"""
        for plugin in self.library_plugins:
            try:
                if module_name in plugin.get_supported_modules():
                    return plugin
            except Exception as e:
                logger.error(f"Error checking library plugin {plugin.name}: {e}")
        return None
    
    def optimize_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """Apply available optimizations to the code"""
        optimized_code = code
        applied_optimizations = []
        
        for plugin in self.optimization_plugins:
            try:
                if plugin.can_optimize(optimized_code, metadata):
                    optimized_code = plugin.optimize(optimized_code, metadata)
                    applied_optimizations.append(plugin.get_optimization_description())
            except Exception as e:
                logger.error(f"Error in optimization plugin {plugin.name}: {e}")
        
        if applied_optimizations:
            logger.info(f"Applied optimizations: {', '.join(applied_optimizations)}")
        
        return optimized_code
    
    def run_analysis(self, source_code: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """Run all analysis plugins on the code"""
        analysis_results = {}
        
        for plugin in self.analysis_plugins:
            try:
                result = plugin.analyze(source_code, ast_tree)
                analysis_results[plugin.get_analysis_name()] = result
            except Exception as e:
                logger.error(f"Error in analysis plugin {plugin.name}: {e}")
        
        return analysis_results
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with their information"""
        plugin_info = {}
        
        for name, plugin in self.plugin_registry.items():
            plugin_info[name] = {
                'type': type(plugin).__bases__[0].__name__,
                'version': plugin.version,
                'description': plugin.description,
                'author': plugin.author,
                'dependencies': plugin.dependencies
            }
        
        return plugin_info
    
    def create_plugin_template(self, plugin_type: str, plugin_name: str) -> str:
        """Create a template for a new plugin"""
        templates = {
            'translation': self._get_translation_plugin_template(plugin_name),
            'library': self._get_library_plugin_template(plugin_name),
            'optimization': self._get_optimization_plugin_template(plugin_name),
            'analysis': self._get_analysis_plugin_template(plugin_name)
        }
        
        return templates.get(plugin_type, "# Unknown plugin type")
    
    def _get_translation_plugin_template(self, name: str) -> str:
        return f'''"""
{name} Translation Plugin

Custom translation plugin for specific Python constructs.
"""

import ast
from typing import Dict, Any
from plugin_system import TranslationPlugin


class {name}(TranslationPlugin):
    """Custom translation plugin for {name.lower()} constructs"""
    
    def __init__(self):
        super().__init__()
        self.name = "{name}"
        self.version = "1.0.0"
        self.description = "{name} translation plugin"
        self.author = "Your Name"
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def can_translate(self, node: ast.AST) -> bool:
        """Check if this plugin can translate the given AST node"""
        # Example: Handle specific AST node types
        return isinstance(node, ast.FunctionDef) and node.name.startswith("special_")
    
    def translate(self, node: ast.AST, context: Dict[str, Any]) -> str:
        """Translate the AST node to C++ code"""
        if isinstance(node, ast.FunctionDef):
            # Custom translation logic here
            func_name = node.name
            return f"// Custom translation for {{func_name}}\\nvoid {{func_name}}() {{\\n    // TODO: Implement\\n}}"
        return ""
    
    def get_priority(self) -> int:
        """Return plugin priority"""
        return 10  # Higher priority than default
'''
    
    def _get_library_plugin_template(self, name: str) -> str:
        return f'''"""
{name} Library Plugin

Support for {name.lower()} Python library.
"""

from typing import Dict, List, Optional
from plugin_system import LibraryPlugin


class {name}Library(LibraryPlugin):
    """Library plugin for {name.lower()}"""
    
    def __init__(self):
        super().__init__()
        self.name = "{name}Library"
        self.version = "1.0.0"
        self.description = "Support for {name.lower()} library"
        self.author = "Your Name"
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def get_supported_modules(self) -> List[str]:
        """Return list of Python modules this plugin supports"""
        return ["{name.lower()}", "{name.lower()}.submodule"]
    
    def get_cpp_dependencies(self) -> Dict[str, str]:
        """Return C++ dependencies and their versions"""
        return {{
            "lib{name.lower()}": "latest",
            "fmt": "^9.0.0"
        }}
    
    def translate_import(self, module_name: str, alias: Optional[str] = None) -> str:
        """Translate Python import to C++ includes"""
        if module_name == "{name.lower()}":
            return f"#include <{name.lower()}/{name.lower()}.hpp>"        return f"#include <{name.lower()}/{{module_name.split('.')[-1]}}.hpp>"
      def translate_function_call(self, func_name: str, args: List[str], kwargs: Dict[str, str]) -> str:
        """Translate function call to C++ equivalent"""
        args_str = ", ".join(args)
        lib_name = "{name.lower()}"
        return f"{{lib_name}}::{{func_name}}({{args_str}})"
'''
    
    def _get_optimization_plugin_template(self, name: str) -> str:
        return f'''"""
{name} Optimization Plugin

Custom optimization for specific code patterns.
"""

from typing import Dict, Any
from plugin_system import OptimizationPlugin


class {name}Optimizer(OptimizationPlugin):
    """Optimization plugin for {name.lower()} patterns"""
    
    def __init__(self):
        super().__init__()
        self.name = "{name}Optimizer"
        self.version = "1.0.0"
        self.description = "{name} optimization plugin"
        self.author = "Your Name"
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def can_optimize(self, code: str, metadata: Dict[str, Any]) -> bool:
        """Check if this plugin can optimize the given code"""
        # Example: Look for specific patterns
        return "inefficient_pattern" in code
    
    def optimize(self, code: str, metadata: Dict[str, Any]) -> str:
        """Apply optimization to the code"""
        # Example optimization
        optimized = code.replace("inefficient_pattern", "optimized_pattern")
        return optimized
    
    def get_optimization_description(self) -> str:
        """Return description of the optimization"""
        return "{name} pattern optimization"
'''
    
    def _get_analysis_plugin_template(self, name: str) -> str:
        return f'''"""
{name} Analysis Plugin

Custom code analysis for specific metrics or patterns.
"""

import ast
from typing import Dict, Any
from plugin_system import AnalysisPlugin


class {name}Analyzer(AnalysisPlugin):
    """Analysis plugin for {name.lower()} metrics"""
    
    def __init__(self):
        super().__init__()
        self.name = "{name}Analyzer"
        self.version = "1.0.0"
        self.description = "{name} analysis plugin"
        self.author = "Your Name"
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def analyze(self, source_code: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """Analyze the source code and return results"""
        results = {{
            "lines_of_code": len(source_code.splitlines()),
            "function_count": len([n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)]),
            "class_count": len([n for n in ast.walk(ast_tree) if isinstance(n, ast.ClassDef)]),
            # Add custom analysis here
        }}
        
        return results
    
    def get_analysis_name(self) -> str:
        """Return name of the analysis"""
        return "{name} Analysis"
'''
    
    def cleanup(self) -> None:
        """Cleanup all plugins"""
        for plugin in self.plugin_registry.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.name}: {e}")


def create_sample_plugins():
    """Create sample plugins for demonstration"""
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    # Sample NumPy plugin
    numpy_plugin = '''"""
NumPy Library Plugin

Provides translation support for NumPy arrays and operations.
"""

import ast
from typing import Dict, List, Optional
from plugin_system import LibraryPlugin


class NumpyLibrary(LibraryPlugin):
    """Library plugin for NumPy"""
    
    def __init__(self):
        super().__init__()
        self.name = "NumpyLibrary"
        self.version = "1.0.0"
        self.description = "Support for NumPy library"
        self.author = "Python to C++ Translator"
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def get_supported_modules(self) -> List[str]:
        return ["numpy", "np"]
    
    def get_cpp_dependencies(self) -> Dict[str, str]:
        return {
            "eigen3": "^3.4.0",
            "xtensor": "^0.24.0"
        }
    
    def translate_import(self, module_name: str, alias: Optional[str] = None) -> str:
        if alias == "np" or module_name == "numpy":
            return "#include <xtensor/xarray.hpp>\\n#include <xtensor/xio.hpp>"
        return "#include <xtensor/xarray.hpp>"
    
    def translate_function_call(self, func_name: str, args: List[str], kwargs: Dict[str, str]) -> str:
        # Map common NumPy functions
        numpy_map = {
            "array": f"xt::adapt({args[0]})" if args else "xt::xarray<double>{}",
            "zeros": f"xt::zeros<double>({{{args[0] if args else '0'}}})",
            "ones": f"xt::ones<double>({{{args[0] if args else '0'}}})",
            "sum": f"xt::sum({args[0]})" if args else "xt::sum()",
            "mean": f"xt::mean({args[0]})" if args else "xt::mean()",
        }
        
        return numpy_map.get(func_name, f"numpy::{func_name}({', '.join(args)})")
'''
    
    # Sample loop optimization plugin
    loop_optimizer = '''"""
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
        pattern1 = r'for\\s+(\\w+)\\s+in\\s+range\\((\\w+)\\)'
        replacement1 = r'for (int \\1 = 0; \\1 < \\2; ++\\1)'
        optimized = re.sub(pattern1, replacement1, optimized)
        
        # Pattern: for i in range(start, end)
        pattern2 = r'for\\s+(\\w+)\\s+in\\s+range\\((\\w+),\\s*(\\w+)\\)'
        replacement2 = r'for (int \\1 = \\2; \\1 < \\3; ++\\1)'
        optimized = re.sub(pattern2, replacement2, optimized)
        
        return optimized
    
    def get_optimization_description(self) -> str:
        return "Range-based loop optimization"
'''
    
    # Write sample plugins
    (plugins_dir / "numpy_plugin.py").write_text(numpy_plugin)
    (plugins_dir / "loop_optimizer.py").write_text(loop_optimizer)
    (plugins_dir / "__init__.py").write_text("")
    
    print("Created sample plugins:")
    print("- plugins/numpy_plugin.py")
    print("- plugins/loop_optimizer.py")


if __name__ == "__main__":
    # Demo the plugin system
    print("Python to C++ Translator - Plugin System Demo")
    print("=" * 50)
    
    # Create sample plugins
    create_sample_plugins()
    
    # Initialize plugin manager
    manager = PluginManager()
    manager.load_plugins()
    
    # List loaded plugins
    plugins = manager.list_plugins()
    print(f"\\nLoaded {len(plugins)} plugins:")
    for name, info in plugins.items():
        print(f"  - {name} ({info['type']}): {info['description']}")
    
    # Test library support
    numpy_plugin = manager.get_library_support("numpy")
    if numpy_plugin:
        print(f"\\nNumPy support available: {numpy_plugin.name}")
        print(f"Import translation: {numpy_plugin.translate_import('numpy', 'np')}")
        print(f"Function call: {numpy_plugin.translate_function_call('zeros', ['10'], {})}")
    
    # Test optimization
    test_code = "for i in range(10):\\n    print(i)"
    optimized = manager.optimize_code(test_code, {})
    print(f"\\nOriginal: {test_code}")
    print(f"Optimized: {optimized}")
    
    # Cleanup
    manager.cleanup()
    print("\\nPlugin system demo completed!")
