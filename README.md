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

**MOBO-Kit** is an open-source toolkit for accelerating design of experiments via **multi-objective Bayesian optimization**. Developed collaboratively across University of Washington, UC San Diego, and MIT, this toolkit enables rapid optimization of complex systems by balancing multiple objectives across any number of inputs and outputs (>2). While demonstrated for slot-die coating experiments (e.g., optimizing efficiency, repeatability, and stability), MOBO-Kit is generalizable to any multi-objective optimization problem.

## Key Features

MOBO-Kit provides a complete package for multi-objective Bayesian optimization with:
- **Latin Hypercube Sampling** for initial experiment design
- **Gaussian Process models** with BoTorch
- **Multi-objective acquisition functions** (qNEHVI)
- **Batch candidate proposal** for efficient parallel experimentation
- **Comprehensive plotting and analysis tools**
- **Constraint handling** for complex design spaces
- **Command-line interface** and Python API

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Package Structure](#package-structure)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)
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

This will automatically install all required dependencies including:
- Core scientific computing: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn
- Machine learning: torch, gpytorch, botorch, emukit
- Additional tools: shap, pyDOE, pyyaml

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

### Option 2: Install with pip (once software license received)

```bash
pip install mobo-kit
```

### Option 3: Google Colab

**Option 3a: Direct Notebook Link**
- [Open MOBO-Kit notebook in Google Colab](https://colab.research.google.com/drive/1VzlCSTDw42kWxlI2xNUAOfmCZpLeKpN0?usp=sharing)

**Option 3b: Install in your own Colab notebook**
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

### 3. Advanced Usage Examples

```python
# Generate initial experiments
from mobo_kit.main import generate_initial_experiments

results = generate_initial_experiments(
    config_path="configs/demo_config.yaml",
    n_samples=20,
    save_path="initial_experiments.csv"
)

# Run MOBO optimization with custom parameters
from mobo_kit.main import run_mobo_experiment

results = run_mobo_experiment(
    csv_path="data/processed/configCSV_example.csv",
    save_dir="results/experiment",
    verbose=True
)
```

### 4. Jupyter Notebooks

See the `notebooks/` directory for interactive examples:
- `MOBO_demo_annotated.ipynb` - Complete workflow demonstration
  - *Note*: The LOOCV function may have trouble converging on small noisy datasets and is still in development.

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
  names:
  - objective 1
  - objective 2
  - objective 3

constraints:
  - clausius_clapeyron: true
    ah_col: "absolute_humidity"
    temp_c_col: "temperature_c"

## Package Structure

```
src/mobo_kit/
├── main.py          # Main API functions
├── cli.py           # Command-line interface
├── design.py        # Design space construction
├── data.py          # Data loading and preprocessing
├── models.py        # Gaussian Process models
├── acquisition.py   # Acquisition functions and batch proposal
├── lhs.py           # Latin Hypercube Sampling
├── plotting.py      # Visualization tools
├── metrics.py       # Performance metrics
├── constraints.py   # Constraint handling
└── utils.py         # Utility functions
```

## Troubleshooting

### Common Installation Issues

1. **Import errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU support**: Install PyTorch with CUDA (example, please use matching nvidia-smi):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Python version compatibility**: Use Python 3.10 or 3.11:
   ```bash
   conda create -n mobo-kit python=3.10
   conda activate mobo-kit
   pip install -e .
   ```

4. **Jupyter notebook support**:
   ```bash
   pip install jupyter ipykernel
   python -m ipykernel install --user --name=mobo-kit --display-name "Python (mobo-kit)"
   ```

### Runtime Issues

- **Memory issues**: For large datasets, consider using CPU instead of GPU or reducing batch sizes
- **Convergence issues**: The LOOCV function may have trouble converging on small noisy datasets
- **CUDA out of memory**: Reduce batch size or use CPU mode

## Next Steps

1. **Try the demo**: `mobo-kit --csv data/processed/configCSV_example.csv --verbose`
2. **Generate initial experiments**: `mobo-kit generate --config configs/demo_config.yaml --n-samples 20 --out my_experiments.csv`
3. **Explore Jupyter notebooks** in the `notebooks/` directory
4. **Check configuration examples** in the `configs/` directory

## Citation

*Citation information will be added upon publication.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Get in Touch

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation in the `notebooks/` directory