"""Setup for workflow-manager — AI CAD OS central orchestrator."""
from setuptools import setup, find_packages

setup(
    name="workflow-manager",
    version="0.1.0",
    description="Central orchestrator for the AI CAD OS pipeline",
    author="AI CAD OS Team",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0",
        "structlog>=23.2.0",
        "httpx>=0.25.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "httpx>=0.25.0",
        ],
    },
)
