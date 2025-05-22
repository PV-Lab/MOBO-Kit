import os

repo_path = "/MOBO-FOM"
os.makedirs(repo_path, exist_ok=True)

# README content
readme_content = """
<h1>
<p align="center">
    <img src="assets/mobo-fom-logo.jpg" alt="Slot-die optimization logo" width="600"/>
</p>
</h1>

<h4 align="center">

[![DOI](https://img.shields.io/badge/DOI-TBD-blue)](https://doi.org/TBD)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/TBD)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

</h4>

**Multiobjective Bayesian Optimization for FOM Slot-Die Coating** is an open-source framework for optimizing functional thin-film fabrication via **multi-objective Bayesian optimization**. Developed collaboratively across University of Washington, UC San Diego, and MIT, this toolkit enables the rapid design of high-performance films by balancing multiple Figures of Merit (FOMs) in slot-die coating experiments, for example efficiency, repeatability, and stability. Funding provided by the ADDEPT Center, sponsored by the US Department of Energy's Solar Energy Technology Office.

---

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Data Format](#data-format)
- [Deciding Which Input/Output Variables to Study](#decide)
- [Initial Sampling of Input Parameter Space](#data-format)
- [Training a Surrogate Model](#training-surrogate-models)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Performing Multi-Objective Optimization](#evaluation-and-visualization)
- [Citation](#citation)
- [License](#license)
- [Get in Touch](#get-in-touch)

---

## Installation

We recommend creating a clean Python environment using `uv` or `venv`:

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .