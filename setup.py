"""Package setup for neuro_edge_reram."""
from setuptools import setup, find_packages

setup(
    name="neuro-edge-reram-simulator",
    version="0.1.0",
    description="Neuro-Edge ReRAM Simulator: in-memory computing and SNN simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "PyYAML>=6.0",
        "streamlit>=1.20.0",
        "matplotlib>=3.5.0",
        "jupyter>=1.0.0",
        "pytest>=7.0.0",
    ],
)
