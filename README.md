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
[![Requires Python 3.11-3.12](https://img.shields.io/badge/Python-3.11--3.12-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

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
conda create -n mobo-kit python=3.11
conda activate mobo-kit

# Or using venv
python -m venv mobo-env
source mobo-env/bin/activate  # On Windows: mobo-env\Scripts\activate

# Install MOBO-Kit
git clone https://github.com/PV-Lab/MOBO-Kit.git
cd MOBO-Kit
python -m pip install -r requirements/dev.txt
```

This installs the core scientific stack, the tested Torch/GPyTorch/BoTorch
combination, workbook auditing support, and the development test tools.

### Optional GPU support

The Step 1 campaign baseline is tested on CPU with PyTorch 2.8.0. GPU wheels
are platform- and CUDA-specific and are not part of that tested baseline. If a
later workflow requires a GPU, install a PyTorch **2.8.0** build using the
[official PyTorch previous-versions instructions](https://pytorch.org/get-started/previous-versions/),
then rerun the import and test smoke checks. Installing an arbitrary newer
Torch release invalidates the pinned BoTorch/GPyTorch compatibility claim.

**Check your CUDA version:**
```bash
nvidia-smi
```

**Verify GPU support:**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
```

### Distribution status

The source/editable installation above is the only installation path validated
for the Step 1 campaign baseline. A PyPI release and the historical Colab link
are not treated as production campaign environments until they have their own
versioned release and smoke-test workflow.

### Dependencies

MOBO-Kit requires:
- Python 3.11 or 3.12
- PyTorch 2.8.x
- BoTorch 0.15.1
- GPyTorch 1.14
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- openpyxl for read-only campaign-workbook auditing
- Pillow for image-artifact validation

The tested probabilistic stack is pinned in `requirements/constraints.txt`.
See `requirements.txt` for the runtime install and `requirements/dev.txt` for
the editable test environment.

## Quick Start

### 1. Command Line Interface

```bash
# The runner accepts the repository's metadata-style campaign CSV format.
# Run model fitting and diagnostics without proposing candidates:
mobo-kit run --csv data/processed/configCSV_example.csv

# Run with custom output directory
mobo-kit run --csv data/my_data.csv --out local_outputs/my_experiment

# Run with verbose output
mobo-kit run --csv data/my_data.csv --verbose
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
    save_dir="local_outputs/experiment",
    verbose=True,
    propose_candidates=False,
)
```

`generate` writes an input-only R0 worklist. It is intentionally not accepted
directly by the legacy `run` proposal path. Campaign-specific workbook adapters
own that boundary and must validate their objective and state contracts first.

### 4. Jupyter Notebooks

See the `notebooks/` directory for interactive examples:
- `MOBO_demo_annotated.ipynb` - Complete workflow demonstration
- `D2D_MOBO_TEST Global Distance Candidate generation.ipynb` - D2D Step 2B
  score validation and guarded debug-adapter interface
  - *Note*: The LOOCV function may have trouble converging on small noisy datasets and is still in development.

### 5. D2D Step 2B algorithm debugging

The D2D adapter requires explicit paths to an ignored private workbook and its
matching ignored private configuration. The tracked configuration is a
non-runnable public template with no workbook identity. The adapter reads the
workbook without saving it, uses the supplied Z/AA/AB final scores directly,
and writes only ignored, watermarked debug artifacts:

```bash
python examples/d2d_step2b_debug.py \
  --workbook local_inputs/<private-workbook>.xlsx \
  --config local_inputs/d2d_step2b_private.yaml
```

This command is deliberately **not** experimental approval. Its five-condition
proposal and 15-row replicate file are labelled `DEBUG ONLY - NOT APPROVED FOR
EXPERIMENT`. The legacy `run --propose-candidates` path remains blocked.

### 6. D2D Step 2C robustness audit

Step 2C validates the small-data GP models, measures all-observation influence,
checks nested Sobol search convergence, refines candidates on the exact discrete
grid, and summarizes persistent candidate regions. It remains read-only and
debug-only:

```bash
# Generated sanitized workbook, small pools, and atomic artifact/ZIP CI coverage
python examples/d2d_step2c_synthetic_ci.py --overwrite

# Quick integration smoke; never eligible to emit a consensus batch
python examples/d2d_step2c_robustness.py --mode fast

# Declared 16,384/32,768/65,536/131,072 robustness study
python examples/d2d_step2c_robustness.py --mode full
```

Artifacts are written below the Git-ignored
`local_outputs/d2d_step2c_robustness/` directory. Every table and plot is
watermarked `DEBUG ONLY - NOT APPROVED FOR EXPERIMENT`; the source workbook is
hash/mtime checked before and after; no workbook writeback or real R2 proposal
is performed. A five-row consensus debug file is possible only in full mode and
only when every declared stability gate passes. Otherwise, the bundle records
the failed gates in `r1_no_stable_batch_reason.json`.

The synthetic CI command creates no private fixture and cannot make the guarded
campaign command accept another workbook. It covers the real v3 read-only
adapter, GP/search orchestration, strict artifact validator, atomic replacement,
and an aggregate-only public summary ZIP on generated sanitized data. Private
campaign runs keep their complete evidence in a separately labelled
`*_PRIVATE_EVIDENCE_DO_NOT_SHARE.zip` archive below the ignored output root.

## Configuration

MOBO-Kit uses YAML configuration files. See `configs/` directory for examples:

- `demo_config.yaml` - Basic configuration
- `configCSV_example_config.yaml` - Configuration from CSV metadata
- `d2d_step2b_debug.yaml` - Non-runnable public template; campaign runs require
  a matching ignored private config and workbook
- `d2d_step2c_debug.yaml` - Public synthetic-CI template; private robustness
  runs require explicit ignored campaign inputs

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
```

Constraints are opt-in. A generic campaign configuration should use
`constraints: []`; the example above is only for a design that actually
contains the two named humidity/temperature inputs.

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

2. **CUDA/GPU support**: Keep Torch at 2.8.0 and follow the official
   platform-specific installation instructions linked above. GPU behavior was
   not validated in the Step 1 baseline.

3. **Python version compatibility**: Python 3.11-3.12 is supported; the Step 1
   CPU baseline was tested with Python 3.12.10:
   ```bash
   conda create -n mobo-kit python=3.11
   conda activate mobo-kit
   python -m pip install -r requirements/dev.txt
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

1. **Try the demo**: `mobo-kit run --csv data/processed/configCSV_example.csv --verbose`
2. **Generate initial experiments**: `mobo-kit generate --config configs/demo_config.yaml --n-samples 20 --out local_outputs/my_experiments.csv`
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
