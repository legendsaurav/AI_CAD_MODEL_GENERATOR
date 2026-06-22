from setuptools import setup, find_packages

setup(
    name="shared-schemas",
    version="1.0.0",
    description="Authoritative interface contracts for the AI CAD Operating System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["pydantic>=2.0"],
)
