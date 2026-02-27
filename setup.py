from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tokenr",
    version="0.1.2",
    author="Tokenr",
    author_email="support@tokenr.co",
    description="Automatic LLM cost tracking for OpenAI, Anthropic, and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tokenr-co/tokenr-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="llm openai anthropic cost tracking monitoring finops",
    project_urls={
        "Bug Reports": "https://github.com/tokenr-co/tokenr-python/issues",
        "Documentation": "https://tokenr.co/docs/api",
        "Source": "https://github.com/tokenr-co/tokenr-python",
    },
)
