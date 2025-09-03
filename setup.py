from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mobo-kit",
    version="0.1.0",
    description="Multi-objective Bayesian optimization toolkit for functional thin-film fabrication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ethan Schwartz, Daniel Abdoue, Nicky Evans, Tonio Buonassisi",
    author_email="ebuddy23@uw.edu",
    url="https://github.com/PV-Lab/MOBO-FOM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    keywords="bayesian-optimization, multi-objective, materials-science, thin-films, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/PV-Lab/MOBO-FOM/issues",
        "Source": "https://github.com/PV-Lab/MOBO-FOM",
        "Documentation": "https://github.com/PV-Lab/MOBO-FOM#readme",
    },
)
