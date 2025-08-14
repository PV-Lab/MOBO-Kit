Environment setup for the uploaded notebooks (Python 3.13.5 target)
=================================================================

Detected top-level libraries:
- numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, emukit
- torch, gpytorch, botorch

Recommended approach
--------------------
1) Create a fresh venv:
   python -m venv mobo-env
   # Windows: mobo-env\Scripts\activate
   # macOS/Linux: source mobo-env/bin/activate

2) Upgrade pip and install core packages:
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements-core.txt

3) Install PyTorch stack (may depend on OS / CUDA / CPU):
   # Start with the generic command; if it fails, follow PyTorch's selector for your platform.
   pip install -r requirements-gpbo.txt

4) Jupyter kernel:
   pip install notebook ipykernel
   python -m ipykernel install --user --name=mobo-env --display-name "Python (mobo-env)"

Notes on Python 3.13.5
----------------------
- Some scientific/ML libraries lag official wheels for brand-new Python versions.
- If torch/gpytorch/botorch fail to install on 3.13 right now, create a *compat* env using Python 3.11 (known to be widely supported):
    conda create -n mobo-py311 python=3.11
    conda activate mobo-py311
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements-core.txt
    # Then install the PyTorch stack using the commands from the official PyTorch website for your OS/accelerator.
- Emukit is pure-Python and typically works across Python versions; if pip reports incompatibility, pin to an earlier release, e.g.:
    pip install "emukit<=0.4.10"

Next steps
----------
- Launch Jupyter and select the 'Python (mobo-env)' kernel.
- Run the notebooks; if a ModuleNotFoundError appears, let me know and I will update the requirements.
