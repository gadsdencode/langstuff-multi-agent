from setuptools import setup, find_packages

setup(
    name="langstuff-multi-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.0.20",
        "langchain-anthropic>=0.0.10",
        "langchain-core>=0.1.20",
        "langchain-openai>=0.0.5",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
) 