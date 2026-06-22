"""
setup.py – makes `geometry-engine` an installable Python package.
Run:  pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="geometry-engine",
    version="1.0.0",
    description="Geometry Understanding Head for AI CAD OS",
    packages=find_packages(exclude=["tests*", "scripts*", "experiments*", "logs*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "Pillow>=10.0.0",
    ],
)
