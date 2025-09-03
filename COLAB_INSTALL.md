# Google Colab Installation Guide

This guide shows how to install and use MOBO-Kit in Google Colab.

## Quick Installation

```python
# Install MOBO-Kit and dependencies
!pip install git+https://github.com/PV-Lab/MOBO-FOM.git

# Check GPU availability (Colab provides free GPU access)
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Import the package
import mobo_kit
```

**Note:** Google Colab provides free GPU access. The default PyTorch installation in Colab already includes CUDA support, so you should see `CUDA available: True` when running the above code.

## Complete Example

```python
# Install MOBO-Kit
!pip install git+https://github.com/PV-Lab/MOBO-FOM.git

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Import MOBO-Kit modules
from mobo_kit.main import main
from mobo_kit.utils import load_csv, split_XY, csv_to_config
from mobo_kit.design import build_input_spec_list, build_space
from mobo_kit.data import x_normalizer_np
from mobo_kit.models import fit_gp_models, posterior_report

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# Run the main workflow
main()
```

## Using with Your Own Data

```python
# Upload your CSV file to Colab
from google.colab import files
uploaded = files.upload()

# Use your data file
import mobo_kit
from mobo_kit.main import run_mobo_experiment

# Create a simple config for your data
config = {
    "inputs": [
        {"name": "param1", "start": 0.0, "stop": 1.0, "step": 0.01},
        {"name": "param2", "start": 0.0, "stop": 1.0, "step": 0.01},
        # Add more parameters as needed
    ],
    "objectives": {
        "names": ["objective1", "objective2", "objective3"]
    },
    "data": {
        "csv_path": "your_data.csv"  # Replace with your uploaded file
    },
    "files": {
        "save_dir": "results/colab_run"
    }
}

# Run optimization
results = run_mobo_experiment(
    csv_path="your_data.csv",
    save_dir="results/colab_run",
    verbose=True
)
print(f"Results saved to: {results['save_dir']}")
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter PyTorch installation issues, try:

```python
# Install PyTorch first
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install MOBO-Kit
!pip install git+https://github.com/PV-Lab/MOBO-FOM.git
```

### Memory Issues

For large datasets, you may need to:

```python
# Use CPU instead of GPU to save memory
device = torch.device("cpu")

# Or reduce batch sizes in your configuration
```

### File Upload Issues

If file upload doesn't work:

```python
# Alternative: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Then use files from your Drive
config["data"]["csv_path"] = "/content/drive/MyDrive/your_data.csv"
```

## Example Notebook

See the `notebooks/MOBO_demo_annotated.ipynb` file for a complete example that can be run in Colab.
