import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.design import build_design_from_config
from src.lhs import lhs_dataframe_optimized, lhs_dataframe
from src.constraints import constraints_from_config
from src.plotting import plot_parity_np, plot_correlation_heatmap, plot_distribution, plot_PCA, plot_pairplot
import yaml
from src.utils import load_csv, split_XY

CFG_PATH = "configs/configCSV_example_config.yaml"
CSV_PATH = "data/processed/configCSV_example.csv"

def test_plots():
    print("Starting test...")
    config = yaml.load(open(CFG_PATH), Loader=yaml.FullLoader)
    print("Config loaded successfully")
    design = build_design_from_config(config)
    print("Design built successfully")
    lhs_df = lhs_dataframe_optimized(design, n=10, max_abs_corr=0.3)
    print("LHS data generated successfully")
    df = load_csv(CSV_PATH)
    print("CSV loaded successfully")
    X, Y = split_XY(df, design, config)
    print(f"Data split successfully - X type: {type(X)}, Y type: {type(Y)}")
    print(f"X columns: {X.columns.tolist()}")
    print(f"X dtypes: {X.dtypes}")
    print(f"X shape: {X.shape}")
    print(X)
    
    # Test correlation heatmap
    print("\nTesting correlation heatmap...")
    fig1, corr_mat = plot_correlation_heatmap(X)
    print("Correlation heatmap created successfully")
    
    # Test distribution plots
    print("\nTesting distribution plots...")
    fig2 = plot_distribution(X, title="Experimental Data Distributions")
    print("Distribution plots created successfully")
    
    # Test PCA plots
    print("\nTesting PCA plots...")
    fig3, pca_result, pca_obj = plot_PCA(X, title="Experimental Data PCA")
    print("PCA plots created successfully")
    print(f"Explained variance: {pca_obj.explained_variance_ratio_[:2].sum():.3f}")
    
    # Test pairplot
    print("\nTesting pairplot...")
    fig4 = plot_pairplot(X, title="Experimental Data Pairwise Relationships")
    print("Pairplot created successfully")
    
    #fig1.savefig("tests/test_correlation.png")
    #fig2.savefig("tests/test_distributions.png")
    #fig3.savefig("tests/test_pca.png")
    #fig4.savefig("tests/test_pairplot.png")
    
    return fig1, fig2, fig3, fig4

def main():
    test_plots()

if __name__ == "__main__":
    main()