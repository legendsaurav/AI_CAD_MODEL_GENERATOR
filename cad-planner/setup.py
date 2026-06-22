from setuptools import setup, find_packages

setup(
    name="cad-planner",
    version="1.0.0",
    description="CAD Planning Engine: Transforms Geometry Graph Language (GGL) into CAD Action Language (CAL).",
    author="Antigravity",
    packages=find_packages(exclude=["tests*", "scripts*", "experiments*", "logs*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "PyYAML>=6.0",
        "numpy>=1.24.0",
        "networkx>=3.0.0" # useful for Dependency Graph and Construction Graph
    ],
)
