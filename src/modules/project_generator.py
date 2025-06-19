"""
C++ Project Template Generator for Python-to-C++ Translation
Generates complete C++ projects with modern build systems, testing, and CI/CD
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import tempfile
import subprocess


class BuildSystem(Enum):
    CMAKE = "cmake"
    MESON = "meson"
    BAZEL = "bazel"
    MAKE = "make"


class PackageManager(Enum):
    VCPKG = "vcpkg"
    CONAN = "conan"
    HUNTER = "hunter"
    MANUAL = "manual"


class TestFramework(Enum):
    GOOGLETEST = "googletest"
    CATCH2 = "catch2"
    DOCTEST = "doctest"
    BOOST_TEST = "boost_test"


class CppStandard(Enum):
    CPP11 = "11"
    CPP14 = "14"
    CPP17 = "17"
    CPP20 = "20"
    CPP23 = "23"


@dataclass
class ProjectConfig:
    """Configuration for C++ project generation"""
    project_name: str
    cpp_standard: CppStandard
    build_system: BuildSystem
    package_manager: PackageManager
    test_framework: TestFramework
    enable_ci: bool
    enable_docs: bool
    enable_benchmarks: bool
    dependencies: List[str]
    include_examples: bool
    target_platform: str  # "windows", "linux", "macos", "all"


@dataclass
class ProjectStructure:
    """Represents the generated project structure"""
    root_dir: Path
    source_dirs: List[Path]
    include_dirs: List[Path]
    test_dirs: List[Path]
    build_files: List[Path]
    config_files: List[Path]


class ProjectTemplateGenerator:
    """Generates comprehensive C++ project templates"""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.predefined_configs = self._load_predefined_configs()
    
    def generate_project(self, config: ProjectConfig, output_dir: Path) -> ProjectStructure:
        """Generate a complete C++ project based on configuration"""
        project_root = output_dir / config.project_name
        project_root.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        structure = self._create_directory_structure(project_root, config)
        
        # Generate build system files
        self._generate_build_files(structure, config)
        
        # Generate source template files
        self._generate_source_templates(structure, config)
        
        # Generate test files
        if config.test_framework != TestFramework.DOCTEST:  # Placeholder check
            self._generate_test_files(structure, config)
        
        # Generate configuration files
        self._generate_config_files(structure, config)
        
        # Generate documentation
        if config.enable_docs:
            self._generate_documentation(structure, config)
        
        # Generate CI/CD files
        if config.enable_ci:
            self._generate_ci_files(structure, config)
        
        # Generate README and other project files
        self._generate_project_files(structure, config)
        
        return structure
    
    def create_modern_cpp_project(self, project_name: str, output_dir: Path,
                                template_type: str = "standard") -> ProjectStructure:
        """Create a modern C++ project with best practices"""
        
        templates = {
            "standard": ProjectConfig(
                project_name=project_name,
                cpp_standard=CppStandard.CPP17,
                build_system=BuildSystem.CMAKE,
                package_manager=PackageManager.VCPKG,
                test_framework=TestFramework.GOOGLETEST,
                enable_ci=True,
                enable_docs=True,
                enable_benchmarks=True,
                dependencies=["fmt", "spdlog", "nlohmann-json"],
                include_examples=True,
                target_platform="all"
            ),
            "lightweight": ProjectConfig(
                project_name=project_name,
                cpp_standard=CppStandard.CPP17,
                build_system=BuildSystem.CMAKE,
                package_manager=PackageManager.MANUAL,
                test_framework=TestFramework.CATCH2,
                enable_ci=False,
                enable_docs=False,
                enable_benchmarks=False,
                dependencies=[],
                include_examples=False,
                target_platform="all"
            ),
            "enterprise": ProjectConfig(
                project_name=project_name,
                cpp_standard=CppStandard.CPP20,
                build_system=BuildSystem.CMAKE,
                package_manager=PackageManager.CONAN,
                test_framework=TestFramework.GOOGLETEST,
                enable_ci=True,
                enable_docs=True,
                enable_benchmarks=True,
                dependencies=["fmt", "spdlog", "nlohmann-json", "boost", "eigen3"],
                include_examples=True,
                target_platform="all"
            )
        }
        
        config = templates.get(template_type, templates["standard"])
        return self.generate_project(config, output_dir)
    
    def _create_directory_structure(self, project_root: Path, 
                                  config: ProjectConfig) -> ProjectStructure:
        """Create the directory structure for the project"""
        
        # Standard C++ project structure
        dirs_to_create = [
            "src",
            "include/" + config.project_name,
            "tests",
            "examples",
            "docs",
            "scripts",
            "third_party",
            "cmake",
            "tools"
        ]
        
        if config.enable_benchmarks:
            dirs_to_create.append("benchmarks")
        
        created_dirs = []
        for dir_path in dirs_to_create:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(full_path)
        
        return ProjectStructure(
            root_dir=project_root,
            source_dirs=[project_root / "src"],
            include_dirs=[project_root / "include"],
            test_dirs=[project_root / "tests"],
            build_files=[],
            config_files=[]
        )
    
    def _generate_build_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate build system files"""
        
        if config.build_system == BuildSystem.CMAKE:
            self._generate_cmake_files(structure, config)
        elif config.build_system == BuildSystem.MESON:
            self._generate_meson_files(structure, config)
        elif config.build_system == BuildSystem.BAZEL:
            self._generate_bazel_files(structure, config)
        else:  # Makefile
            self._generate_makefile(structure, config)
    
    def _generate_cmake_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate CMake build files"""
        
        # Main CMakeLists.txt
        cmake_content = self._generate_main_cmake(config)
        main_cmake = structure.root_dir / "CMakeLists.txt"
        with open(main_cmake, 'w', encoding='utf-8') as f:
            f.write(cmake_content)
        structure.build_files.append(main_cmake)
        
        # Source CMakeLists.txt
        src_cmake_content = self._generate_src_cmake(config)
        src_cmake = structure.root_dir / "src" / "CMakeLists.txt"
        with open(src_cmake, 'w', encoding='utf-8') as f:
            f.write(src_cmake_content)
        structure.build_files.append(src_cmake)
        
        # Tests CMakeLists.txt
        if config.test_framework != TestFramework.DOCTEST:  # Placeholder
            test_cmake_content = self._generate_test_cmake(config)
            test_cmake = structure.root_dir / "tests" / "CMakeLists.txt"
            with open(test_cmake, 'w', encoding='utf-8') as f:
                f.write(test_cmake_content)
            structure.build_files.append(test_cmake)
        
        # CMake modules
        self._generate_cmake_modules(structure, config)
        
        # Package manager integration
        if config.package_manager == PackageManager.VCPKG:
            self._generate_vcpkg_files(structure, config)
        elif config.package_manager == PackageManager.CONAN:
            self._generate_conan_files(structure, config)
    
    def _generate_main_cmake(self, config: ProjectConfig) -> str:
        """Generate main CMakeLists.txt content"""
        
        lines = [
            f"cmake_minimum_required(VERSION 3.15)",
            f"",
            f"project({config.project_name}",
            f"    VERSION 1.0.0",
            f"    DESCRIPTION \"Generated C++ project from Python translation\"",
            f"    LANGUAGES CXX)",
            f"",
            f"# Set C++ standard",
            f"set(CMAKE_CXX_STANDARD {config.cpp_standard.value})",
            f"set(CMAKE_CXX_STANDARD_REQUIRED ON)",
            f"set(CMAKE_CXX_EXTENSIONS OFF)",
            f"",
            f"# Add project options",
            f"option(BUILD_TESTS \"Build tests\" ON)",
            f"option(BUILD_EXAMPLES \"Build examples\" ON)",
            f"option(BUILD_DOCS \"Build documentation\" OFF)",
        ]
        
        if config.enable_benchmarks:
            lines.append("option(BUILD_BENCHMARKS \"Build benchmarks\" OFF)")
        
        lines.extend([
            "",
            "# Set default build type",
            "if(NOT CMAKE_BUILD_TYPE)",
            "    set(CMAKE_BUILD_TYPE Release)",
            "endif()",
            "",
            "# Compiler-specific options",
            "if(MSVC)",
            "    add_compile_options(/W4)",
            "    add_definitions(-D_CRT_SECURE_NO_WARNINGS)",
            "else()",
            "    add_compile_options(-Wall -Wextra -Wpedantic)",
            "endif()",
            "",
            "# Find packages"
        ])
        
        # Add package dependencies
        package_mappings = {
            "fmt": "find_package(fmt REQUIRED)",
            "spdlog": "find_package(spdlog REQUIRED)", 
            "nlohmann-json": "find_package(nlohmann_json REQUIRED)",
            "boost": "find_package(Boost REQUIRED)",
            "eigen3": "find_package(Eigen3 REQUIRED)"
        }
        
        for dep in config.dependencies:
            if dep in package_mappings:
                lines.append(package_mappings[dep])
        
        lines.extend([
            "",
            "# Include directories",
            "include_directories(include)",
            "",
            "# Add subdirectories",
            "add_subdirectory(src)",
            "",
            "if(BUILD_TESTS)",
            "    enable_testing()",
            "    add_subdirectory(tests)",
            "endif()",
            "",
            "if(BUILD_EXAMPLES)",
            "    add_subdirectory(examples)",
            "endif()",
        ])
        
        if config.enable_benchmarks:
            lines.extend([
                "",
                "if(BUILD_BENCHMARKS)",
                "    add_subdirectory(benchmarks)",
                "endif()"
            ])
        
        if config.enable_docs:
            lines.extend([
                "",
                "if(BUILD_DOCS)",
                "    add_subdirectory(docs)",
                "endif()"
            ])
        
        return '\n'.join(lines)
    
    def _generate_src_cmake(self, config: ProjectConfig) -> str:
        """Generate src/CMakeLists.txt content"""
        
        lines = [
            f"# Source files",
            f"set(SOURCES",
            f"    main.cpp",
            f"    {config.project_name}.cpp",
            f")",
            f"",
            f"# Create library",
            f"add_library({config.project_name}_lib ${{SOURCES}})",
            f"",
            f"# Link dependencies",
            f"target_include_directories({config.project_name}_lib PUBLIC",
            f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/../include)",
            f"",
        ]
        
        # Add dependency linking
        dependency_links = {
            "fmt": f"target_link_libraries({config.project_name}_lib PUBLIC fmt::fmt)",
            "spdlog": f"target_link_libraries({config.project_name}_lib PUBLIC spdlog::spdlog)",
            "nlohmann-json": f"target_link_libraries({config.project_name}_lib PUBLIC nlohmann_json::nlohmann_json)",
            "boost": f"target_link_libraries({config.project_name}_lib PUBLIC Boost::boost)",
            "eigen3": f"target_link_libraries({config.project_name}_lib PUBLIC Eigen3::Eigen)"
        }
        
        for dep in config.dependencies:
            if dep in dependency_links:
                lines.append(dependency_links[dep])
        
        lines.extend([
            "",
            f"# Create executable",
            f"add_executable({config.project_name} main.cpp)",
            f"target_link_libraries({config.project_name} PRIVATE {config.project_name}_lib)",
            "",
            f"# Installation",
            f"install(TARGETS {config.project_name}",
            f"    RUNTIME DESTINATION bin)",
            f"",
            f"install(TARGETS {config.project_name}_lib",
            f"    LIBRARY DESTINATION lib",
            f"    ARCHIVE DESTINATION lib)",
            f"",
            f"install(DIRECTORY ../include/{config.project_name}",
            f"    DESTINATION include)"
        ])
        
        return '\n'.join(lines)
    
    def _generate_test_cmake(self, config: ProjectConfig) -> str:
        """Generate tests/CMakeLists.txt content"""
        
        if config.test_framework == TestFramework.GOOGLETEST:
            return self._generate_gtest_cmake(config)
        elif config.test_framework == TestFramework.CATCH2:
            return self._generate_catch2_cmake(config)
        else:
            return "# Test configuration not implemented"
    
    def _generate_gtest_cmake(self, config: ProjectConfig) -> str:
        """Generate Google Test CMake configuration"""
        
        lines = [
            "# Google Test configuration",
            "find_package(GTest REQUIRED)",
            "",
            "# Test sources",
            "set(TEST_SOURCES",
            f"    test_{config.project_name}.cpp",
            "    test_main.cpp",
            ")",
            "",
            f"# Create test executable",
            f"add_executable({config.project_name}_tests ${{TEST_SOURCES}})",
            "",
            f"# Link libraries",
            f"target_link_libraries({config.project_name}_tests",
            f"    PRIVATE",
            f"    {config.project_name}_lib",
            f"    GTest::gtest",
            f"    GTest::gtest_main",
            f")",
            "",
            f"# Include directories",
            f"target_include_directories({config.project_name}_tests PRIVATE",
            f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/../include)",
            "",
            "# Register tests",
            "include(GoogleTest)",
            f"gtest_discover_tests({config.project_name}_tests)"
        ]
        
        return '\n'.join(lines)
    
    def _generate_catch2_cmake(self, config: ProjectConfig) -> str:
        """Generate Catch2 CMake configuration"""
        
        lines = [
            "# Catch2 configuration",
            "find_package(Catch2 REQUIRED)",
            "",
            "# Test sources",
            "set(TEST_SOURCES",
            f"    test_{config.project_name}.cpp",
            ")",
            "",
            f"# Create test executable",
            f"add_executable({config.project_name}_tests ${{TEST_SOURCES}})",
            "",
            f"# Link libraries",
            f"target_link_libraries({config.project_name}_tests",
            f"    PRIVATE",
            f"    {config.project_name}_lib",
            f"    Catch2::Catch2WithMain",
            f")",
            "",
            f"# Include directories",
            f"target_include_directories({config.project_name}_tests PRIVATE",
            f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/../include)",
            "",
            "# Register tests",
            "include(CTest)",
            "include(Catch)",
            f"catch_discover_tests({config.project_name}_tests)"
        ]
        
        return '\n'.join(lines)
    
    def _generate_vcpkg_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate vcpkg configuration files"""
        
        # vcpkg.json
        vcpkg_config = {
            "name": config.project_name.lower(),
            "version": "1.0.0",
            "description": f"Generated C++ project: {config.project_name}",
            "dependencies": []
        }
        
        # Map dependencies to vcpkg names
        vcpkg_mappings = {
            "fmt": "fmt",
            "spdlog": "spdlog",
            "nlohmann-json": "nlohmann-json",
            "boost": "boost",
            "eigen3": "eigen3"
        }
        
        if config.test_framework == TestFramework.GOOGLETEST:
            vcpkg_mappings["gtest"] = "gtest"
        elif config.test_framework == TestFramework.CATCH2:
            vcpkg_mappings["catch2"] = "catch2"
        
        for dep in config.dependencies:
            if dep in vcpkg_mappings:
                vcpkg_config["dependencies"].append(vcpkg_mappings[dep])
        
        # Add test framework dependency
        if config.test_framework.value in vcpkg_mappings:
            vcpkg_config["dependencies"].append(vcpkg_mappings[config.test_framework.value])
        
        vcpkg_file = structure.root_dir / "vcpkg.json"
        with open(vcpkg_file, 'w', encoding='utf-8') as f:
            json.dump(vcpkg_config, f, indent=2)
        structure.config_files.append(vcpkg_file)
        
        # CMake toolchain reference
        cmake_presets = {
            "version": 3,
            "configurePresets": [
                {
                    "name": "vcpkg",
                    "displayName": "Default with vcpkg",
                    "description": "Default configuration with vcpkg",
                    "binaryDir": "build",
                    "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
                }
            ]
        }
        
        presets_file = structure.root_dir / "CMakePresets.json"
        with open(presets_file, 'w', encoding='utf-8') as f:
            json.dump(cmake_presets, f, indent=2)
        structure.config_files.append(presets_file)
    
    def _generate_conan_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate Conan configuration files"""
        
        # conanfile.txt
        conan_content = [
            "[requires]"
        ]
        
        # Map dependencies to Conan names
        conan_mappings = {
            "fmt": "fmt/8.1.1",
            "spdlog": "spdlog/1.11.0",
            "nlohmann-json": "nlohmann_json/3.11.2",
            "boost": "boost/1.81.0",
            "eigen3": "eigen/3.4.0"
        }
        
        for dep in config.dependencies:
            if dep in conan_mappings:
                conan_content.append(conan_mappings[dep])
        
        # Add test framework
        if config.test_framework == TestFramework.GOOGLETEST:
            conan_content.append("gtest/1.12.1")
        elif config.test_framework == TestFramework.CATCH2:
            conan_content.append("catch2/3.2.1")
        
        conan_content.extend([
            "",
            "[generators]",
            "CMakeDeps",
            "CMakeToolchain",
            "",
            "[options]",
            "",
            "[imports]"
        ])
        
        conanfile = structure.root_dir / "conanfile.txt"
        with open(conanfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conan_content))
        structure.config_files.append(conanfile)
    
    def _generate_source_templates(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate source code templates"""
        
        # Main header file
        header_content = self._generate_main_header(config)
        header_file = structure.root_dir / "include" / config.project_name / f"{config.project_name}.h"
        with open(header_file, 'w', encoding='utf-8') as f:
            f.write(header_content)
        
        # Main source file
        source_content = self._generate_main_source(config)
        source_file = structure.root_dir / "src" / f"{config.project_name}.cpp"
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(source_content)
        
        # Main executable
        main_content = self._generate_main_executable(config)
        main_file = structure.root_dir / "src" / "main.cpp"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(main_content)
    
    def _generate_main_header(self, config: ProjectConfig) -> str:
        """Generate main header file content"""
        
        guard_name = f"{config.project_name.upper()}_H"
        
        lines = [
            f"#ifndef {guard_name}",
            f"#define {guard_name}",
            "",
            "#include <string>",
            "#include <vector>",
            "#include <memory>",
            ""
        ]
        
        # Add dependency includes
        if "fmt" in config.dependencies:
            lines.append("#include <fmt/format.h>")
        if "spdlog" in config.dependencies:
            lines.append("#include <spdlog/spdlog.h>")
        if "nlohmann-json" in config.dependencies:
            lines.append("#include <nlohmann/json.hpp>")
        
        lines.extend([
            "",
            f"namespace {config.project_name.lower()} {{",
            "",
            f"class {config.project_name} {{",
            "public:",
            f"    {config.project_name}();",
            f"    ~{config.project_name}() = default;",
            "",
            "    // Core functionality",
            "    void initialize();",
            "    void process();",
            "    void finalize();",
            "",
            "    // Getters/Setters",
            "    const std::string& get_name() const { return name_; }",
            "    void set_name(const std::string& name) { name_ = name; }",
            "",
            "private:",
            "    std::string name_;",
            "    bool initialized_;",
            "};",
            "",
            "// Utility functions",
            "std::string get_version();",
            "void log_info(const std::string& message);",
            "",
            f"}} // namespace {config.project_name.lower()}",
            "",
            f"#endif // {guard_name}"
        ])
        
        return '\n'.join(lines)
    
    def _generate_main_source(self, config: ProjectConfig) -> str:
        """Generate main source file content"""
        
        lines = [
            f"#include \"{config.project_name}/{config.project_name}.h\"",
            "#include <iostream>",
            ""
        ]
        
        # Add dependency-specific code
        if "spdlog" in config.dependencies:
            lines.extend([
                "#include <spdlog/spdlog.h>",
                ""
            ])
        
        lines.extend([
            f"namespace {config.project_name.lower()} {{",
            "",
            f"{config.project_name}::{config.project_name}()",
            f"    : name_(\"{config.project_name}\"), initialized_(false) {{",
            "    // Constructor implementation",
            "}",
            "",
            f"void {config.project_name}::initialize() {{",
            "    if (initialized_) {",
            "        return;",
            "    }",
            "",
            "    log_info(\"Initializing \" + name_);",
            "    initialized_ = true;",
            "}",
            "",
            f"void {config.project_name}::process() {{",
            "    if (!initialized_) {",
            "        initialize();",
            "    }",
            "",
            "    log_info(\"Processing in \" + name_);",
            "    // Main processing logic here",
            "}",
            "",
            f"void {config.project_name}::finalize() {{",
            "    if (!initialized_) {",
            "        return;",
            "    }",
            "",
            "    log_info(\"Finalizing \" + name_);",
            "    initialized_ = false;",
            "}",
            "",
            "std::string get_version() {",
            "    return \"1.0.0\";",
            "}",
            "",
            "void log_info(const std::string& message) {"
        ])
        
        if "spdlog" in config.dependencies:
            lines.append("    spdlog::info(message);")
        else:
            lines.append("    std::cout << \"[INFO] \" << message << std::endl;")
        
        lines.extend([
            "}",
            "",
            f"}} // namespace {config.project_name.lower()}"        ])
        
        return '\n'.join(lines)
    
    def _generate_main_executable(self, config: ProjectConfig) -> str:
        """Generate main executable content"""
        
        lines = [
            f"#include \"{config.project_name}/{config.project_name}.h\"",
            "#include <iostream>",
            "",
            "int main(int argc, char* argv[]) {",
            f"    {config.project_name.lower()}::{config.project_name} app;",
            "",
            "    try {",
            "        app.initialize();",
            "        app.process();",
            "        app.finalize();",
            "",
            f"        std::cout << \"Application completed successfully!\" << std::endl;",
            "        return 0;",
            "",
            "    } catch (const std::exception& e) {",
            "        std::cerr << \"Error: \" << e.what() << std::endl;",
            "        return 1;",
            "    }",
            "}"
        ]
        
        return '\n'.join(lines)
    
    def _generate_test_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate test files"""
        
        if config.test_framework == TestFramework.GOOGLETEST:
            self._generate_gtest_files(structure, config)
        elif config.test_framework == TestFramework.CATCH2:
            self._generate_catch2_files(structure, config)
    
    def _generate_gtest_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate Google Test files"""
        
        # Main test file
        test_content = [
            "#include <gtest/gtest.h>",
            f"#include \"{config.project_name}/{config.project_name}.h\"",
            "",
            f"class {config.project_name}Test : public ::testing::Test {{",
            "protected:",
            "    void SetUp() override {",
            f"        app_ = std::make_unique<{config.project_name.lower()}::{config.project_name}>();",
            "    }",
            "",
            "    void TearDown() override {",
            "        app_.reset();",
            "    }",
            "",
            f"    std::unique_ptr<{config.project_name.lower()}::{config.project_name}> app_;",
            "};",
            "",
            f"TEST_F({config.project_name}Test, Initialization) {{",
            "    EXPECT_NO_THROW(app_->initialize());",
            "}",
            "",
            f"TEST_F({config.project_name}Test, Processing) {{",
            "    app_->initialize();",
            "    EXPECT_NO_THROW(app_->process());",
            "}",
            "",
            f"TEST_F({config.project_name}Test, NameGetterSetter) {{",
            "    const std::string test_name = \"TestApp\";",
            "    app_->set_name(test_name);",
            "    EXPECT_EQ(app_->get_name(), test_name);",
            "}",
            "",
            "TEST(UtilityTests, Version) {",
            f"    std::string version = {config.project_name.lower()}::get_version();",
            "    EXPECT_FALSE(version.empty());",
            "}"
        ]
        
        test_file = structure.root_dir / "tests" / f"test_{config.project_name}.cpp"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_content))
        
        # Test main file
        test_main_content = [
            "#include <gtest/gtest.h>",
            "",
            "int main(int argc, char **argv) {",
            "    ::testing::InitGoogleTest(&argc, argv);",
            "    return RUN_ALL_TESTS();",
            "}"
        ]
        
        test_main_file = structure.root_dir / "tests" / "test_main.cpp"
        with open(test_main_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_main_content))
    
    def _generate_catch2_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate Catch2 test files"""
        
        test_content = [
            "#include <catch2/catch_test_macros.hpp>",
            f"#include \"{config.project_name}/{config.project_name}.h\"",
            "",
            f"TEST_CASE(\"{config.project_name} functionality\", \"[{config.project_name.lower()}]\") {{",
            f"    {config.project_name.lower()}::{config.project_name} app;",
            "",
            "    SECTION(\"Initialization\") {",
            "        REQUIRE_NOTHROW(app.initialize());",
            "    }",
            "",
            "    SECTION(\"Name handling\") {",
            "        const std::string test_name = \"TestApp\";",
            "        app.set_name(test_name);",
            "        REQUIRE(app.get_name() == test_name);",
            "    }",
            "}",
            "",
            "TEST_CASE(\"Utility functions\", \"[utilities]\") {",
            f"    std::string version = {config.project_name.lower()}::get_version();",
            "    REQUIRE_FALSE(version.empty());",
            "}"
        ]
        
        test_file = structure.root_dir / "tests" / f"test_{config.project_name}.cpp"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_content))
    
    def _generate_config_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate configuration files"""
        
        # .clang-format
        clang_format = {
            "Language": "Cpp",
            "BasedOnStyle": "Google",
            "IndentWidth": 4,
            "ColumnLimit": 100,
            "AllowShortFunctionsOnASingleLine": "Empty",
            "AllowShortIfStatementsOnASingleLine": "false",
            "BreakBeforeBraces": "Attach"
        }
        
        # Convert to YAML-like format for .clang-format
        clang_format_content = []
        for key, value in clang_format.items():
            if isinstance(value, str) and value != "false":
                clang_format_content.append(f"{key}: {value}")
            elif isinstance(value, bool):
                clang_format_content.append(f"{key}: {'true' if value else 'false'}")
            else:
                clang_format_content.append(f"{key}: {value}")
        
        clang_format_file = structure.root_dir / ".clang-format"
        with open(clang_format_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(clang_format_content))
        structure.config_files.append(clang_format_file)
        
        # .gitignore
        gitignore_content = [
            "# Build directories",
            "build/",
            "cmake-build-*/",
            "",
            "# IDE files",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "",
            "# Compiled files",
            "*.exe",
            "*.dll",
            "*.so",
            "*.a",
            "*.lib",
            "",
            "# Cache and temporary files",
            ".cache/",
            "*.tmp",
            "",
            "# Package manager files",
            "conan.lock",
            "conanbuildinfo.*",
            "vcpkg_installed/",
            "",
            "# OS specific",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        gitignore_file = structure.root_dir / ".gitignore"
        with open(gitignore_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(gitignore_content))
        structure.config_files.append(gitignore_file)
    
    def _generate_ci_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate CI/CD configuration files"""
        
        # GitHub Actions
        github_actions_dir = structure.root_dir / ".github" / "workflows"
        github_actions_dir.mkdir(parents=True, exist_ok=True)
        
        ci_content = self._generate_github_actions_ci(config)
        ci_file = github_actions_dir / "ci.yml"
        with open(ci_file, 'w', encoding='utf-8') as f:
            f.write(ci_content)
        structure.config_files.append(ci_file)
    
    def _generate_github_actions_ci(self, config: ProjectConfig) -> str:
        """Generate GitHub Actions CI configuration"""
        
        ci_config = {
            "name": "CI",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]}
            },
            "jobs": {
                "build": {
                    "runs-on": "${{ matrix.os }}",
                    "strategy": {
                        "matrix": {
                            "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
                            "build_type": ["Debug", "Release"]
                        }
                    },
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up CMake",
                            "uses": "lukka/get-cmake@latest"
                        }
                    ]
                }
            }
        }
          # Add package manager specific steps
        if config.package_manager == PackageManager.VCPKG:
            ci_config["jobs"]["build"]["steps"].extend([
                {
                    "name": "Setup vcpkg",
                    "uses": "lukka/run-vcpkg@v11",
                    "with": {
                        "vcpkgGitCommitId": "${{ env.VCPKG_COMMIT_ID }}"
                    }
                }
            ])
        
        ci_config["jobs"]["build"]["steps"].extend([
            {
                "name": "Configure CMake",
                "run": "cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}"
            },
            {
                "name": "Build",
                "run": "cmake --build build --config ${{ matrix.build_type }}"
            },
            {
                "name": "Test",
                "run": "ctest --test-dir build --config ${{ matrix.build_type }}"
            }
        ])
          # Convert to YAML format (simplified)
        return self._dict_to_yaml(ci_config)
    
    def _dict_to_yaml(self, data: Dict, indent: int = 0) -> str:
        """Convert dictionary to YAML format (simplified)"""
        lines = []
        spaces = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{spaces}{key}:")
                    lines.append(self._dict_to_yaml(value, indent + 1))
                elif isinstance(value, list):
                    lines.append(f"{spaces}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"{spaces}  -")
                            for subkey, subvalue in item.items():
                                lines.append(f"{spaces}    {subkey}: {subvalue}")
                        else:
                            lines.append(f"{spaces}  - {item}")
                else:
                    lines.append(f"{spaces}{key}: {value}")
        
        return '\n'.join(lines)
    
    def _generate_documentation(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate documentation files"""
        
        # Doxygen configuration
        doxyfile_content = self._generate_doxyfile(config)
        doxyfile = structure.root_dir / "docs" / "Doxyfile"
        with open(doxyfile, 'w', encoding='utf-8') as f:
            f.write(doxyfile_content)
        
        # Documentation CMakeLists.txt
        docs_cmake = [
            "find_package(Doxygen)",
            "",
            "if(DOXYGEN_FOUND)",
            "    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)",
            "    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)",
            "",
            "    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)",
            "",
            "    add_custom_target(docs ALL",
            "        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}",
            "        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}",
            "        COMMENT \"Generating API documentation with Doxygen\"",
            "        VERBATIM)",
            "else()",
            "    message(\"Doxygen not found, documentation will not be built\")",
            "endif()"
        ]
        
        docs_cmake_file = structure.root_dir / "docs" / "CMakeLists.txt"
        with open(docs_cmake_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(docs_cmake))
    
    def _generate_doxyfile(self, config: ProjectConfig) -> str:
        """Generate Doxygen configuration"""
        
        return f"""
PROJECT_NAME           = "{config.project_name}"
PROJECT_NUMBER         = 1.0.0
PROJECT_BRIEF          = "Generated C++ project from Python translation"

OUTPUT_DIRECTORY       = ./
CREATE_SUBDIRS         = NO

INPUT                  = ../include ../src
INPUT_ENCODING         = UTF-8
FILE_PATTERNS          = *.cpp *.h *.hpp
RECURSIVE              = YES

EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES

GENERATE_HTML          = YES
HTML_OUTPUT            = html
HTML_FILE_EXTENSION    = .html

GENERATE_LATEX         = NO

HAVE_DOT               = NO
UML_LOOK               = NO
"""
    
    def _generate_project_files(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate README and other project files"""
        
        # README.md
        readme_content = self._generate_readme(config)
        readme_file = structure.root_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # BUILD.md
        build_content = self._generate_build_instructions(config)
        build_file = structure.root_dir / "BUILD.md"
        with open(build_file, 'w', encoding='utf-8') as f:
            f.write(build_content)
        
        # CONTRIBUTING.md
        contributing_content = self._generate_contributing_guide(config)
        contributing_file = structure.root_dir / "CONTRIBUTING.md"
        with open(contributing_file, 'w', encoding='utf-8') as f:
            f.write(contributing_content)
    
    def _generate_readme(self, config: ProjectConfig) -> str:
        """Generate README.md content"""
        
        lines = [
            f"# {config.project_name}",
            "",
            f"A C++ project generated from Python-to-C++ translation.",
            "",
            "## Features",
            "",
            "- Modern C++ design patterns",
            f"- C++{config.cpp_standard.value} standard compliance",
            f"- {config.build_system.value.upper()} build system",
            f"- {config.test_framework.value} testing framework",
        ]
        
        if config.package_manager != PackageManager.MANUAL:
            lines.append(f"- {config.package_manager.value} package management")
        
        if config.enable_ci:
            lines.append("- Continuous Integration setup")
        
        if config.enable_docs:
            lines.append("- Automated documentation generation")
        
        lines.extend([
            "",
            "## Building",
            "",
            "### Prerequisites",
            "",
            f"- CMake 3.15 or higher",
            f"- C++{config.cpp_standard.value} compatible compiler",
        ])
        
        if config.package_manager == PackageManager.VCPKG:
            lines.extend([
                "- vcpkg package manager",
                "",
                "### Quick Start with vcpkg",
                "",
                "```bash",
                "# Clone the repository",
                f"git clone <repository-url> {config.project_name}",
                f"cd {config.project_name}",
                "",
                "# Configure with vcpkg",
                "cmake --preset vcpkg",
                "",
                "# Build",
                "cmake --build build",
                "",
                "# Run tests",
                "ctest --test-dir build",
                "```"
            ])
        elif config.package_manager == PackageManager.CONAN:
            lines.extend([
                "- Conan package manager",
                "",
                "### Quick Start with Conan",
                "",
                "```bash",
                "# Clone the repository",
                f"git clone <repository-url> {config.project_name}",
                f"cd {config.project_name}",
                "",
                "# Install dependencies",
                "mkdir build && cd build",
                "conan install ..",
                "",
                "# Configure and build",
                "cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake",
                "cmake --build .",
                "",
                "# Run tests",
                "ctest",
                "```"
            ])
        else:
            lines.extend([
                "",
                "### Manual Build",
                "",
                "```bash",
                "# Clone the repository",
                f"git clone <repository-url> {config.project_name}",
                f"cd {config.project_name}",
                "",
                "# Configure and build",
                "mkdir build && cd build",
                "cmake ..",
                "cmake --build .",
                "",
                "# Run tests",
                "ctest",
                "```"
            ])
        
        lines.extend([
            "",
            "## Usage",
            "",
            f"```cpp",
            f"#include \"{config.project_name}/{config.project_name}.h\"",
            "",
            "int main() {",
            f"    {config.project_name.lower()}::{config.project_name} app;",
            "    app.initialize();",
            "    app.process();",
            "    app.finalize();",
            "    return 0;",
            "}",
            "```",
            "",
            "## Testing",
            "",
            f"This project uses {config.test_framework.value} for testing.",
            "",
            "```bash",
            "# Run all tests",
            "ctest --test-dir build",
            "",
            "# Run tests with verbose output",
            "ctest --test-dir build --verbose",
            "```"
        ])
        
        if config.enable_docs:
            lines.extend([
                "",
                "## Documentation",
                "",
                "Generate documentation using Doxygen:",
                "",
                "```bash",
                "cmake --build build --target docs",
                "```",
                "",
                "Open `build/docs/html/index.html` in your browser."
            ])
        
        lines.extend([
            "",
            "## Contributing",
            "",
            "See [CONTRIBUTING.md](CONTRIBUTING.md) for details.",
            "",
            "## License",
            "",
            "This project is licensed under the MIT License."
        ])
        
        return '\n'.join(lines)
    
    def _generate_build_instructions(self, config: ProjectConfig) -> str:
        """Generate detailed build instructions"""
        
        lines = [
            f"# Building {config.project_name}",
            "",
            "This document provides detailed build instructions for all supported platforms.",
            "",
            "## Prerequisites",
            "",
            "### All Platforms",
            "",
            f"- CMake 3.15 or higher",
            f"- C++{config.cpp_standard.value} compatible compiler",
        ]
        
        if config.package_manager == PackageManager.VCPKG:
            lines.extend([
                "- vcpkg package manager",
                "",
                "### Installing vcpkg",
                "",
                "```bash",
                "git clone https://github.com/Microsoft/vcpkg.git",
                "cd vcpkg",
                "./bootstrap-vcpkg.sh  # Linux/macOS",
                ".\\bootstrap-vcpkg.bat  # Windows",
                "```"
            ])
        
        lines.extend([
            "",
            "## Platform-Specific Instructions",
            "",
            "### Windows (Visual Studio)",
            "",
            "```cmd",
            "mkdir build",
            "cd build",
        ])
        
        if config.package_manager == PackageManager.VCPKG:
            lines.append('cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake')
        else:
            lines.append("cmake ..")
        
        lines.extend([
            "cmake --build . --config Release",
            "```",
            "",
            "### Linux (GCC/Clang)",
            "",
            "```bash",
            "mkdir build && cd build",
        ])
        
        if config.package_manager == PackageManager.VCPKG:
            lines.append('cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake')
        else:
            lines.append("cmake ..")
        
        lines.extend([
            "make -j$(nproc)",
            "```",
            "",
            "### macOS (Xcode/Clang)",
            "",
            "```bash",
            "mkdir build && cd build",
        ])
        
        if config.package_manager == PackageManager.VCPKG:
            lines.append('cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake')
        else:
            lines.append("cmake ..")
        
        lines.extend([
            "make -j$(sysctl -n hw.ncpu)",
            "```",
            "",
            "## Build Options",
            "",
            "```bash",
            "# Debug build",
            "cmake .. -DCMAKE_BUILD_TYPE=Debug",
            "",
            "# Release build with debugging info",
            "cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "",
            "# Disable tests",
            "cmake .. -DBUILD_TESTS=OFF",
        ])
        
        if config.enable_benchmarks:
            lines.extend([
                "",
                "# Enable benchmarks",
                "cmake .. -DBUILD_BENCHMARKS=ON"
            ])
        
        lines.extend([
            "```",
            "",
            "## Troubleshooting",
            "",
            "### Common Issues",
            "",
            "1. **Missing dependencies**: Ensure all required packages are installed",
            "2. **Compiler version**: Verify C++ compiler supports the required standard",
            "3. **CMake version**: Update CMake if build fails with version errors",
            "",
            "### Getting Help",
            "",
            "If you encounter issues, please:",
            "",
            "1. Check the error messages carefully",
            "2. Verify all prerequisites are installed",
            "3. Search existing issues in the project repository",
            "4. Create a new issue with detailed error information"
        ])
        
        return '\n'.join(lines)
    
    def _generate_contributing_guide(self, config: ProjectConfig) -> str:
        """Generate contributing guide"""
        
        lines = [
            f"# Contributing to {config.project_name}",
            "",
            "Thank you for your interest in contributing! This guide will help you get started.",
            "",
            "## Development Setup",
            "",
            "1. Fork the repository",
            "2. Clone your fork locally",
            "3. Install dependencies",
            "4. Build the project",
            "5. Run tests to ensure everything works",
            "",
            "## Code Style",
            "",
            "This project follows Google C++ Style Guide with some modifications:",
            "",
            "- Use 4 spaces for indentation",
            "- Line length limit: 100 characters",
            "- Use `snake_case` for variables and functions",
            "- Use `PascalCase` for classes and namespaces",
            "",
            "### Formatting",
            "",
            "Use clang-format to ensure consistent formatting:",
            "",
            "```bash",
            "clang-format -i src/**/*.cpp include/**/*.h",
            "```",
            "",
            "## Testing",
            "",
            "- Write tests for all new functionality",
            f"- Use {config.test_framework.value} for unit tests",
            "- Ensure all tests pass before submitting",
            "- Aim for high test coverage",
            "",
            "## Pull Request Process",
            "",
            "1. Create a feature branch from `main`",
            "2. Make your changes with clear, descriptive commits",
            "3. Add or update tests as needed",
            "4. Update documentation if necessary",
            "5. Ensure all tests pass and code is formatted",
            "6. Submit a pull request with a clear description",
            "",
            "## Commit Messages",
            "",
            "Use clear, descriptive commit messages:",
            "",
            "```",
            "feat: add new feature X",
            "fix: resolve issue with Y",
            "docs: update README with new instructions",
            "test: add tests for Z functionality",
            "refactor: improve code structure in module A",
            "```",
            "",
            "## Review Process",
            "",
            "All contributions go through code review:",
            "",
            "1. Automated CI checks must pass",
            "2. At least one maintainer review required",
            "3. Address any feedback promptly",
            "4. Maintainer will merge when ready",
            "",
            "## Reporting Issues",
            "",
            "When reporting bugs or requesting features:",
            "",
            "1. Check existing issues first",
            "2. Use the issue templates provided",
            "3. Include relevant system information",
            "4. Provide minimal reproduction steps for bugs",
            "",
            "## Questions?",
            "",
            "Feel free to ask questions by creating an issue or reaching out to maintainers."
        ]
        
        return '\n'.join(lines)
    
    def _generate_cmake_modules(self, structure: ProjectStructure, config: ProjectConfig):
        """Generate CMake module files"""
        
        cmake_dir = structure.root_dir / "cmake"
        
        # Compiler options module
        compiler_options = [
            "# Compiler-specific options and warnings",
            "",
            "function(set_project_warnings target)",
            "    if(MSVC)",
            "        target_compile_options(${target} PRIVATE",
            "            /W4        # Warning level 4",
            "            /WX        # Treat warnings as errors",
            "            /w14242    # Possible loss of data",
            "            /w14254    # Operator with different types",
            "            /w14263    # Function not virtual",
            "            /w14265    # Class has virtual functions but no virtual destructor",
            "            /w14287    # Unsigned/negative constant mismatch",
            "            /we4289    # Loop variable used outside loop",
            "            /w14296    # Expression always false",
            "            /w14311    # Pointer truncation",
            "            /w14545    # Expression before comma has no effect",
            "            /w14546    # Function call before comma missing argument list",
            "            /w14547    # Operator before comma has no effect",
            "            /w14549    # Operator before comma has no effect",
            "            /w14555    # Expression has no effect",
            "            /w14619    # Unknown pragma",
            "            /w14640    # Construction of local static object not thread-safe",
            "            /w14826    # Conversion is sign-extended",
            "            /w14905    # Wide string literal cast to LPSTR",
            "            /w14906    # String literal cast to LPWSTR",
            "            /w14928    # Illegal copy-initialization",
            "        )",
            "    else()",
            "        target_compile_options(${target} PRIVATE",
            "            -Wall      # All warnings",
            "            -Wextra    # Extra warnings",
            "            -Wpedantic # Pedantic warnings",
            "            -Werror    # Treat warnings as errors",
            "            -Wshadow   # Shadow warnings",
            "            -Wnon-virtual-dtor # Non-virtual destructor",
            "            -Wold-style-cast   # C-style casts",
            "            -Wcast-align       # Cast alignment",
            "            -Wunused           # Unused variables",
            "            -Woverloaded-virtual # Overloaded virtual",
            "            -Wconversion       # Type conversions",
            "            -Wsign-conversion  # Sign conversions",
            "            -Wmisleading-indentation # Misleading indentation",
            "            -Wduplicated-cond  # Duplicated conditions",
            "            -Wduplicated-branches # Duplicated branches",
            "            -Wlogical-op       # Logical operations",
            "            -Wnull-dereference # Null dereference",
            "            -Wuseless-cast     # Useless casts",
            "            -Wdouble-promotion # Double promotion",
            "            -Wformat=2         # Format string security",
            "        )",
            "    endif()",
            "endfunction()"
        ]
        
        compiler_file = cmake_dir / "CompilerOptions.cmake"
        with open(compiler_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(compiler_options))
    
    def _load_predefined_configs(self) -> Dict[str, ProjectConfig]:
        """Load predefined project configurations"""
        
        return {
            "minimal": ProjectConfig(
                project_name="MinimalProject",
                cpp_standard=CppStandard.CPP17,
                build_system=BuildSystem.CMAKE,
                package_manager=PackageManager.MANUAL,
                test_framework=TestFramework.CATCH2,
                enable_ci=False,
                enable_docs=False,
                enable_benchmarks=False,
                dependencies=[],
                include_examples=False,
                target_platform="all"
            ),
            "standard": ProjectConfig(
                project_name="StandardProject",
                cpp_standard=CppStandard.CPP17,
                build_system=BuildSystem.CMAKE,
                package_manager=PackageManager.VCPKG,
                test_framework=TestFramework.GOOGLETEST,
                enable_ci=True,
                enable_docs=True,
                enable_benchmarks=True,
                dependencies=["fmt", "spdlog"],
                include_examples=True,
                target_platform="all"
            ),
            "enterprise": ProjectConfig(
                project_name="EnterpriseProject",
                cpp_standard=CppStandard.CPP20,
                build_system=BuildSystem.CMAKE,
                package_manager=PackageManager.CONAN,
                test_framework=TestFramework.GOOGLETEST,
                enable_ci=True,
                enable_docs=True,
                enable_benchmarks=True,
                dependencies=["fmt", "spdlog", "nlohmann-json", "boost"],
                include_examples=True,
                target_platform="all"
            )
        }


# CLI Integration
def generate_project_cli():
    """Command-line interface for project generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate modern C++ project templates")
    parser.add_argument("project_name", help="Name of the project to generate")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--template", "-t", default="standard", 
                       choices=["minimal", "standard", "enterprise"],
                       help="Project template type")
    parser.add_argument("--cpp-standard", default="17", choices=["11", "14", "17", "20", "23"],
                       help="C++ standard version")
    parser.add_argument("--build-system", default="cmake", choices=["cmake", "meson", "bazel"],
                       help="Build system to use")
    parser.add_argument("--package-manager", default="vcpkg", 
                       choices=["vcpkg", "conan", "hunter", "manual"],
                       help="Package manager to use")
    parser.add_argument("--test-framework", default="googletest",
                       choices=["googletest", "catch2", "doctest", "boost_test"],
                       help="Test framework to use")
    parser.add_argument("--no-ci", action="store_true", help="Disable CI/CD generation")
    parser.add_argument("--no-docs", action="store_true", help="Disable documentation generation")
    parser.add_argument("--no-benchmarks", action="store_true", help="Disable benchmark generation")
    
    args = parser.parse_args()
    
    generator = ProjectTemplateGenerator()
    output_dir = Path(args.output)
    
    if args.template in ["minimal", "standard", "enterprise"]:
        structure = generator.create_modern_cpp_project(args.project_name, output_dir, args.template)
    else:
        # Custom configuration
        config = ProjectConfig(
            project_name=args.project_name,
            cpp_standard=CppStandard(args.cpp_standard),
            build_system=BuildSystem(args.build_system),
            package_manager=PackageManager(args.package_manager),
            test_framework=TestFramework(args.test_framework),
            enable_ci=not args.no_ci,
            enable_docs=not args.no_docs,
            enable_benchmarks=not args.no_benchmarks,
            dependencies=["fmt", "spdlog"],  # Default
            include_examples=True,
            target_platform="all"
        )
        
        structure = generator.generate_project(config, output_dir)
    
    print(f"✅ Generated C++ project: {structure.root_dir}")
    print(f"📁 Project structure created with {len(structure.build_files)} build files")
    print(f"🔧 Ready to build with: cd {structure.root_dir} && mkdir build && cd build && cmake ..")


if __name__ == "__main__":
    generate_project_cli()
