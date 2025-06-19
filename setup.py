from setuptools import setup, find_packages

setup(
    name="python-to-cpp-translator",
    version="0.1.0",
    description="A tool for translating Python source code to C++",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "ast-inspector>=0.8.0",
        "click>=8.1.7",
    ],
    entry_points={
        "console_scripts": [
            "py2cpp=src.main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
