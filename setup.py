from setuptools import setup, find_packages

setup(
    name="MOBO-FOM",
    version="0.1.0",
    description="For driving multiobjective optmization campaigns.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    authors="Ethan Schwartz, Daniel Abdoue, Nicky Evans, Tonio Buonassisi",
    author_emails="ebuddy23@uw.edu, dabdoue@ucsd.edu, nickye17@mit.edu, buonassi@mit.edu",
    url="https://github.com/PV-Lab/MOBO-FOM",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)