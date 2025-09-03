MOBO-Kit: Multi-objective Bayesian Optimization Toolkit
======================================================

Environment setup for MOBO-Kit (Python 3.10+ target)
====================================================

MOBO-Kit is a complete package for multi-objective Bayesian optimization with:
- Latin Hypercube Sampling for initial experiment design
- Gaussian Process models with BoTorch
- Multi-objective acquisition functions (qNEHVI)
- Batch candidate proposal
- Comprehensive plotting and analysis tools

Quick Installation
------------------
1) Create a fresh virtual environment:
   python -m venv mobo-env
   # Windows: mobo-env\Scripts\activate
   # macOS/Linux: source mobo-env/bin/activate

2) Install MOBO-Kit in development mode:
   pip install -e .

This will automatically install all required dependencies including:
- Core scientific computing: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn
- Machine learning: torch, gpytorch, botorch, emukit
- Additional tools: shap, pyDOE, pyyaml

GPU Support (CUDA)
------------------
By default, PyTorch is installed in CPU-only mode. For GPU acceleration:

1) Check your CUDA version:
   nvidia-smi

2) Install PyTorch with CUDA support:
   # For CUDA 12.1 (recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8 (more compatible)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.8 (latest)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

3) Verify GPU support:
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Alternative Installation Methods
--------------------------------
1) From source (development):
   git clone <repository-url>
   cd MOBO-FOM
   pip install -e .

2) For Google Colab:
   See COLAB_INSTALL.md for specific instructions

3) Using conda (if preferred):
   conda create -n mobo-kit python=3.10
   conda activate mobo-kit
   pip install -e .

Usage Examples
--------------
1) Generate initial experiments:
   mobo-kit generate --config configs/demo_config.yaml --n-samples 20 --out initial_experiments.csv

2) Run MOBO optimization:
   mobo-kit run --csv data/processed/configCSV_example.csv --verbose

3) Python API:
   from mobo_kit.main import run_mobo_experiment, generate_initial_experiments
   
   # Generate initial experiments
   results = generate_initial_experiments(
       config_path="configs/demo_config.yaml",
       n_samples=20,
       save_path="initial_experiments.csv"
   )
   
   # Run MOBO optimization
   results = run_mobo_experiment(
       csv_path="data/processed/configCSV_example.csv",
       save_dir="results/experiment"
   )

Package Structure
-----------------
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

Configuration Files
-------------------
- configs/demo_config.yaml: Example configuration
- configs/auto_config.yaml: Auto-generated from CSV metadata
- configs/configCSV_example_config.yaml: Example with CSV metadata

Output Files
------------
When running MOBO experiments, the following files are generated:
- predictions.csv: Model predictions for all data points
- parity_plots.png: Visual parity plots showing model performance
- model_metrics.csv: R² and RMSE metrics for each objective
- next_batch.csv: Proposed new experiments to run

Troubleshooting
---------------
1) If you encounter import errors, ensure all dependencies are installed:
   pip install -r requirements.txt

2) For CUDA/GPU support, install PyTorch with CUDA:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3) If you have Python 3.13+ compatibility issues, use Python 3.10 or 3.11:
   conda create -n mobo-kit python=3.10
   conda activate mobo-kit
   pip install -e .

4) For Jupyter notebook support:
   pip install jupyter ipykernel
   python -m ipykernel install --user --name=mobo-kit --display-name "Python (mobo-kit)"

Next Steps
----------
1) Try the demo: mobo-kit run --csv data/processed/configCSV_example.csv --verbose
2) Generate your own initial experiments: mobo-kit generate --config configs/demo_config.yaml --n-samples 20 --out my_experiments.csv
3) Explore the Jupyter notebooks in the notebooks/ directory
4) Check the README.md for detailed usage instructions

For more information, see:
- README.md: Complete usage guide
- COLAB_INSTALL.md: Google Colab setup
- notebooks/MOBO_demo_annotated.ipynb: Interactive tutorial