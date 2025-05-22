def load_data():
    df = pd.read_csv("data/raw/batch0_results.csv")
    X = df.iloc[:, 0:8].values
    y = df[['PCE', 'Stability', 'Repeatability']].values
    return X, y

def main():
    check_environment()
    X, y = load_data()
    X_norm = preprocess_data(X)
    Y_scaled, model = run_model(X_norm, y)
    postprocess_and_save(Y_scaled)