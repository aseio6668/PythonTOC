"""
C++ code templates for common patterns
"""

from typing import Dict, List, Optional


class CppTemplates:
    """Collection of C++ code templates"""
    
    def __init__(self):
        self.templates = {
            'main_function': self._main_function_template(),
            'class_declaration': self._class_declaration_template(),
            'function_declaration': self._function_declaration_template(),
            'constructor': self._constructor_template(),
            'destructor': self._destructor_template(),
            'header_file': self._header_file_template(),
            'source_file': self._source_file_template(),
            'cmake_file': self._cmake_file_template(),
            'makefile': self._makefile_template(),
        }
    
    def get_template(self, name: str) -> str:
        """Get a template by name"""
        return self.templates.get(name, "")
    
    def _main_function_template(self) -> str:
        """Template for main function"""
        return """int main(int argc, char* argv[]) {
    // Generated from Python code
    
    {body}
    
    return 0;
}"""
    
    def _class_declaration_template(self) -> str:
        """Template for class declaration"""
        return """class {class_name}{inheritance} {
private:
{private_members}

public:
{public_members}
};"""
    
    def _function_declaration_template(self) -> str:
        """Template for function declaration"""
        return """{return_type} {function_name}({parameters}){const_qualifier};"""
    
    def _constructor_template(self) -> str:
        """Template for constructor"""
        return """{class_name}({parameters}){initializer_list} {
{body}
}"""
    
    def _destructor_template(self) -> str:
        """Template for destructor"""
        return """~{class_name}() {
{body}
}"""
    
    def _header_file_template(self) -> str:
        """Template for header file"""
        return """#ifndef {header_guard}
#define {header_guard}

{includes}

{forward_declarations}

{namespace_open}

{class_declarations}

{function_declarations}

{namespace_close}

#endif // {header_guard}"""
    
    def _source_file_template(self) -> str:
        """Template for source file"""
        return """{includes}

{namespace_open}

{implementations}

{namespace_close}"""
    
    def _cmake_file_template(self) -> str:
        """Template for CMakeLists.txt"""
        return """cmake_minimum_required(VERSION 3.12)
project({project_name})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
{find_packages}

# Add executable
add_executable({executable_name} {source_files})

# Link libraries
{link_libraries}

# Compiler options
target_compile_options({executable_name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Include directories
{include_directories}"""
    
    def _makefile_template(self) -> str:
        """Template for Makefile"""
        return """CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -O2
TARGET = {target}
SOURCES = {sources}
OBJECTS = $(SOURCES:.cpp=.o)

{includes}
{libraries}

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean"""

    def format_main_function(self, body: str) -> str:
        """Format main function with body"""
        return self.templates['main_function'].format(body=body)
    
    def format_class_declaration(self, 
                                class_name: str,
                                inheritance: str = "",
                                private_members: str = "",
                                public_members: str = "") -> str:
        """Format class declaration"""
        inheritance_str = f" : {inheritance}" if inheritance else ""
        return self.templates['class_declaration'].format(
            class_name=class_name,
            inheritance=inheritance_str,
            private_members=private_members,
            public_members=public_members
        )
    
    def format_function_declaration(self,
                                   return_type: str,
                                   function_name: str,
                                   parameters: str = "",
                                   const_qualifier: str = "") -> str:
        """Format function declaration"""
        const_str = f" {const_qualifier}" if const_qualifier else ""
        return self.templates['function_declaration'].format(
            return_type=return_type,
            function_name=function_name,
            parameters=parameters,
            const_qualifier=const_str
        )
    
    def format_constructor(self,
                          class_name: str,
                          parameters: str = "",
                          initializer_list: str = "",
                          body: str = "") -> str:
        """Format constructor"""
        init_str = f" : {initializer_list}" if initializer_list else ""
        return self.templates['constructor'].format(
            class_name=class_name,
            parameters=parameters,
            initializer_list=init_str,
            body=body
        )
    
    def format_header_file(self,
                          header_guard: str,
                          includes: str = "",
                          forward_declarations: str = "",
                          namespace_open: str = "",
                          namespace_close: str = "",
                          class_declarations: str = "",
                          function_declarations: str = "") -> str:
        """Format header file"""
        return self.templates['header_file'].format(
            header_guard=header_guard,
            includes=includes,
            forward_declarations=forward_declarations,
            namespace_open=namespace_open,
            namespace_close=namespace_close,
            class_declarations=class_declarations,
            function_declarations=function_declarations
        )
    
    def format_cmake_file(self,
                         project_name: str,
                         executable_name: str,
                         source_files: str,
                         find_packages: str = "",
                         link_libraries: str = "",
                         include_directories: str = "") -> str:
        """Format CMakeLists.txt file"""
        return self.templates['cmake_file'].format(
            project_name=project_name,
            executable_name=executable_name,
            source_files=source_files,
            find_packages=find_packages,
            link_libraries=link_libraries,
            include_directories=include_directories
        )

    @staticmethod
    def generate_standard_includes() -> List[str]:
        """Generate commonly used standard includes"""
        return [
            "#include <iostream>",
            "#include <string>",
            "#include <vector>",
            "#include <memory>",
            "#include <algorithm>",
            "#include <functional>",
            "#include <utility>",
            "#include <stdexcept>"
        ]
    
    @staticmethod
    def generate_namespace_wrapper(namespace: str, content: str) -> str:
        """Wrap content in a namespace"""
        if not namespace:
            return content
        
        return f"""namespace {namespace} {{

{content}

}} // namespace {namespace}"""
    
    @staticmethod
    def generate_include_guard(header_name: str) -> tuple[str, str]:
        """Generate include guard for header file"""
        guard_name = header_name.upper().replace('.', '_').replace('/', '_')
        if not guard_name.endswith('_H'):
            guard_name += '_H'
        
        opening = f"#ifndef {guard_name}\n#define {guard_name}"
        closing = f"#endif // {guard_name}"
        
        return opening, closing
    
    @staticmethod
    def generate_file_header_comment(original_file: str, 
                                   generator_info: str = "Python to C++ Translator") -> str:
        """Generate file header comment"""
        import datetime
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""/*
 * Generated by {generator_info}
 * Original Python file: {original_file}
 * Generated on: {current_time}
 * 
 * Note: This is auto-generated code. Manual modifications may be lost
 * when the code is regenerated. Please modify the original Python source
 * instead.
 */"""
    
    @staticmethod
    def generate_todo_comment(message: str) -> str:
        """Generate a TODO comment"""
        return f"// TODO: {message}"
    
    @staticmethod
    def generate_warning_comment(message: str) -> str:
        """Generate a warning comment"""
        return f"// WARNING: {message}"
    
    @staticmethod
    def generate_note_comment(message: str) -> str:
        """Generate a note comment"""
        return f"// NOTE: {message}"
