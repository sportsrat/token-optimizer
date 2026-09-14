from setuptools import setup, find_packages

setup(
    name="contextflow",
    version="0.1.0",
    description="Intelligent caching and context optimization layer for LLMs",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "groq>=0.9.0",
        "sentence-transformers>=3.0.0",
        "numpy>=1.24.0",
        "tiktoken>=0.7.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)