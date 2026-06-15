from setuptools import setup, find_packages

setup(
    name="dfednexus",
    version="1.0.0",
    description=(
        "DFedNexus: Reliable Decentralised Federated Learning for Wireless Edge Networks"
    ),
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
        ]
    },
)