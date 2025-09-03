# README: MOBO-Kit
by Ethan Schwartz, Daniel Abdoue, Nicky Evans, and Tonio Buonassisi
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

**MOBO-Kit** is an open-source toolkit for optimizing functional thin-film fabrication via **multi-objective Bayesian optimization**. Developed collaboratively across University of Washington, UC San Diego, and MIT, this toolkit enables the rapid design of high-performance films by balancing multiple Figures of Merit (FOMs) in slot-die coating experiments, for example efficiency, repeatability, and stability. Funding provided by the ADDEPT Center, sponsored by the US Department of Energy's Solar Energy Technology Office.

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

### Option 1: Install from Source (Recommended)

We recommend creating a clean Python environment using `conda` or `venv`:

```bash
# Create and activate environment
conda create -n mobo-fom python=3.10
conda activate mobo-fom

# Or using venv
python -m venv mobo-env
source mobo-env/bin/activate  # On Windows: mobo-env\Scripts\activate

# Install MOBO-Kit
git clone https://github.com/PV-Lab/MOBO-FOM.git
cd MOBO-FOM
pip install -e .
```

### GPU Support (CUDA)

MOBO-Kit uses PyTorch for machine learning models. By default, the installation includes the CPU-only version of PyTorch. For GPU acceleration, you'll need to install the CUDA version of PyTorch.

**Check your CUDA version:**
```bash
nvidia-smi
```

**Install PyTorch with CUDA support:**
```bash
# For CUDA 12.1 (recommended for most systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (more compatible with older systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.8 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verify GPU support:**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
```

### Option 2: Install with pip

```bash
pip install mobo-kit
```

### Option 3: Google Colab

```python
# Install in Google Colab
!pip install git+https://github.com/PV-Lab/MOBO-FOM.git

# Import and use
import mobo_kit
```

### Dependencies

MOBO-Kit requires:
- Python 3.10+
- PyTorch 1.12+
- BoTorch 0.8+
- GPyTorch 1.8+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

See `requirements.txt` for the complete list of dependencies.

## Quick Start

### 1. Command Line Interface

```bash
# Run with default configuration
mobo-kit --csv data/processed/configCSV_example.csv

# Run with custom output directory
mobo-kit --csv data/my_data.csv --out results/my_experiment

# Run with verbose output
mobo-kit --csv data/my_data.csv --verbose
```

### 2. Python API

```python
import mobo_kit
from mobo_kit.main import main

# Run the main workflow
main()
```

### 3. Jupyter Notebooks

See the `notebooks/` directory for interactive examples:
- `MOBO_demo_annotated.ipynb` - Complete workflow demonstration

## Configuration

MOBO-Kit uses YAML configuration files. See `configs/` directory for examples:

- `demo_config.yaml` - Basic configuration
- `configCSV_example_config.yaml` - Configuration from CSV metadata

### Configuration Structure

```yaml
inputs:
  - name: "parameter1"
    unit: "unit"
    start: 0.0
    stop: 1.0
    step: 0.01

objectives:
  names: ["objective1", "objective2", "objective3"]

constraints:
  - clausius_clapeyron: true
    ah_col: "absolute_humidity"
    temp_c_col: "temperature_c"

files:
  save_dir: "results/demo"