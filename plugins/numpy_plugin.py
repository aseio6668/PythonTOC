"""
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
            return "#include <xtensor/xarray.hpp>\n#include <xtensor/xio.hpp>"
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
