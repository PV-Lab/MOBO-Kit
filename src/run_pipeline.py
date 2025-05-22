#delete Tonio's comments before releasing publicly: this structure has the following advantages:
#Reusability: Each function can be tested or reused independently. Daniel has a vision…
#Debuggability: Easier to trace failures or unexpected outputs, especially when working as a team.
#Extensibility: You can now easily insert things like logging, validation, hyperparameter tuning, etc.

import torch
import pandas as pd
import numpy as np
from src.design import generate_initial_design, get_variable_space
from src.utils import get_closest_array, x_normalizer, x_denormalizer, get_closest_array
from src.model import fit_gp_models

# Optional: add paths if your raw data is in data/raw/ and outputs in data/processed/
RAW_DATA_PATH = "data/raw/input.csv"
OUTPUT_PATH = "data/processed/output.csv"

def check_environment():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)

def load_or_generate_data():
    try:
        df = pd.read_csv("data/raw/batch0_results.csv")
        X = df.iloc[:, 0:8].values
        y = df[['PCE', 'Stability', 'Repeatability']].values
        return X, y
    except FileNotFoundError:
        print("No real data found. Generating initial design...")
        design = generate_initial_design(n_samples=10)
        var_array = get_variable_space()
        snapped_design = get_closest_array(design, var_array)
        return snapped_design, None

def load_data():
    # Replace with your actual data structure
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded data with shape: {df.shape}")
    return df

def preprocess_data(df):
    # Example placeholder: replace with your actual preprocessing logic
    print("Preprocessing data...")
    data = df.values
    normalized = x_normalizer(data, var_array=[data.T for _ in range(data.shape[1])])
    return np.array(normalized)

def run_model(X, y):
    print("Running GP model...")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Optionally scale outputs (replicates original notebook behavior)
    Y_max = y_tensor.max(dim=0).values
    Y_min = y_tensor.min(dim=0).values
    Y_scaled = (y_tensor - Y_min) / (Y_max - Y_min)

    model = fit_gp_models(X_tensor, Y_scaled)

    return Y_scaled.detach().numpy(), model

def postprocess_and_save(results):
    # Example: convert back to DataFrame and save
    print("Saving results...")
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

#def main():
#    check_environment()
#    df = load_data()
#    X = preprocess_data(df)
#    results = run_model(X)
#    postprocess_and_save(results)

def main():
    check_environment()
    X, y = load_or_generate_data()
    X_norm = preprocess_data(X)
    if y is not None:
        Y_scaled, model = run_model(X_norm, y)
        postprocess_and_save(Y_scaled)
    else:
        print("Initial design generated. Run experiments and save results to `data/raw/batch0_results.csv`.")

if __name__ == '__main__':
    main()