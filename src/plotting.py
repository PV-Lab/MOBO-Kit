# src/plotting.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error
import shap
import torch

def plot_parity_np(
    true_Y: np.ndarray,                 # (N, M) in ORIGINAL units
    pred_mean: np.ndarray,              # (N, M) in ORIGINAL units
    pred_std: Optional[np.ndarray] = None,  # (N, M) in ORIGINAL units (optional)
    objective_names: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,   # default: (5*M, 5)
    equal_axes: bool = True,
    save: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Parity plots (Predicted vs True) with optional vertical error bars (predictive std).
    Returns (fig, metrics_df) where metrics_df has R² and RMSE per objective.
    
    Args:
        true_Y: True values array of shape (N, M)
        pred_mean: Predicted mean values array of shape (N, M)
        pred_std: Optional predicted standard deviation array of shape (N, M)
        objective_names: Optional sequence of objective names
        figsize: Optional figure size tuple, defaults to (5*M, 5)
        equal_axes: Whether to set equal axis limits and aspect ratio
        save: Optional path to save the figure
        show_plot: Whether to display the plot using plt.show()
    """
    # Basic shape checks
    true = np.asarray(true_Y, dtype=float)
    pred = np.asarray(pred_mean, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: true {true.shape} vs pred {pred.shape}")
    if pred_std is not None:
        std = np.asarray(pred_std, dtype=float)
        if std.shape != true.shape:
            raise ValueError(f"Shape mismatch: std {std.shape} vs true {true.shape}")
    else:
        std = None

    N, M = true.shape
    names = list(objective_names) if objective_names is not None else [f"obj{j}" for j in range(M)]
    if len(names) != M:
        raise ValueError(f"objective_names length {len(names)} != M={M}")

    if figsize is None:
        figsize = (5.0 * M, 5.0)

    fig, axes = plt.subplots(1, M, figsize=figsize, squeeze=False)
    axes = axes[0]

    rows = []
    for j, name in enumerate(names):
        ax = axes[j]
        if std is not None:
            ax.errorbar(
                true[:, j], pred[:, j],
                yerr=std[:, j],
                fmt="o", color='blue', ecolor="gray", elinewidth=1, capsize=3, alpha=0.75,
            )
        else:
            sns.scatterplot(x=true[:, j], y=pred[:, j], s=35, ax=ax)

        # Parity line and axis handling
        lo = float(min(true[:, j].min(), pred[:, j].min()))
        hi = float(max(true[:, j].max(), pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        if equal_axes:
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)

        # Metrics (original units)
        r2 = r2_score(true[:, j], pred[:, j])
        rmse = root_mean_squared_error(true[:, j], pred[:, j])
        ax.text(
            0.05, 0.95, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        rows.append({"Objective": name, "R2": round(float(r2), 3), "RMSE": round(float(rmse), 3)})

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, pd.DataFrame(rows, columns=["Objective", "R2", "RMSE"])


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: int = 20,
    cmap: str = 'coolwarm',
    annot: bool = True,
    fmt: str = '.2f',
    title: Optional[str] = None,
    save: Optional[str] = None,
    show_plot: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a Pearson correlation heatmap for input features.
    
    Works with any DataFrame containing numeric data, including:
    - LHS design DataFrames from lhs_dataframe() or lhs_dataframe_optimized()
    - X DataFrames from split_XY()
    - Any other DataFrame with numeric columns
    
    Args:
        df: DataFrame containing the data
        columns: Specific columns to include (if None, uses all numeric columns)
        figsize: Figure size (if None, auto-calculates based on number of columns)
        fontsize: Font size for labels and title
        cmap: Colormap for the heatmap
        annot: Whether to show correlation values on the heatmap
        fmt: Format string for correlation values
        title: Plot title (if None, auto-generates)
        save: Path to save the figure (if None, doesn't save)
        show_plot: Whether to display the plot
        
    Returns:
        Tuple of (figure, correlation_matrix)
        
    Example:
        >>> # For LHS dataframe
        >>> lhs_df = lhs_dataframe_optimized(design, n=10, max_abs_corr=0.3)
        >>> fig, corr_matrix = plot_correlation_heatmap(lhs_df, title="LHS Design Correlation")
        >>> 
        >>> # For X data from split_XY
        >>> X_df, Y_df = split_XY(csv_path, design, config)
        >>> fig, corr_matrix = plot_correlation_heatmap(X_df, title="Experimental Data Correlation")
    """
    # Select columns to analyze
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in DataFrame")
        columns = list(numeric_cols)
    else:
        # Verify all specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Filter DataFrame to selected columns
    df_subset = df[columns].copy()
    
    # Calculate correlation matrix
    corr_matrix = df_subset.corr(method='pearson')
    
    # Auto-calculate figure size if not provided

    if figsize is None:
        n_cols = len(columns)
        figsize = (n_cols, n_cols)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set seaborn style
    sns.set(font_scale=1.5)
    sns.set_style("ticks", {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': False,
        'ytick.right': False
    })
    
    # Create mask for upper triangle (to show only lower triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cbar_kws={"shrink": 0.2}, 
        annot=annot, 
        fmt=fmt,
        cmap=cmap, 
        cbar=False, 
        ax=ax, 
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    
    # Set axis limits and labels
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(columns), 0)
    
    # Set title
    if title is None:
        title = f"Pearson Correlation Matrix ({len(columns)} features)"
    ax.set_title(title, fontsize=fontsize)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=75, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if requested
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, corr_matrix.values


def plot_distribution(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_cols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    bins: int = 20,
    fontsize: int = 18,
    title: Optional[str] = None,
    save: Optional[str] = None,
    show_plot: bool = False,
) -> plt.Figure:
    """
    Create distribution plots (histograms) for input features.
    
    Works with any DataFrame containing numeric data, including:
    - LHS design DataFrames from lhs_dataframe() or lhs_dataframe_optimized()
    - X DataFrames from split_XY()
    - Any other DataFrame with numeric columns
    
    Args:
        df: DataFrame containing the data
        columns: Specific columns to include (if None, uses all numeric columns)
        n_cols: Number of columns per row in the figure
        figsize: Figure size (if None, auto-calculates based on n_cols)
        bins: Number of bins for histograms
        fontsize: Font size for labels and title
        title: Plot title (if None, auto-generates)
        save: Path to save the figure (if None, doesn't save)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
        
    Example:
        >>> # For LHS dataframe
        >>> lhs_df = lhs_dataframe_optimized(design, n=10, max_abs_corr=0.3)
        >>> fig = plot_distribution(lhs_df, title="LHS Design Distributions")
        >>> 
        >>> # For X data from split_XY
        >>> X_df, Y_df = split_XY(csv_path, design, config)
        >>> fig = plot_distribution(X_df, title="Experimental Data Distributions")
    """
    # Select columns to analyze
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in DataFrame")
        columns = list(numeric_cols)
    else:
        # Verify all specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Filter DataFrame to selected columns
    df_subset = df[columns].copy()
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (15, 3.5)  # Default size from your notebook
    
    # Set seaborn style
    sns.set_style("ticks", {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': False,
        'ytick.right': False
    })
    
    # Calculate number of rows needed
    n_features = len(columns)
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot histograms
    for i, col in enumerate(columns):
        row_idx = i // n_cols
        col_idx = i % n_cols
        
        ax = axes[row_idx, col_idx]
        
        # Create histogram
        ax.hist(df_subset[col], bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel(col, fontsize=fontsize)
        
        # Set y-label only for first column of each row
        if col_idx == 0:
            ax.set_ylabel('Counts', fontsize=fontsize)
        
        # Style the plot
        ax.tick_params(direction='in', length=5, width=1, labelsize=fontsize*0.8)
        ax.grid(True, linestyle='-.', alpha=0.5)
    
    # Turn off unused subplots
    for i in range(n_features, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].axis('off')
    
    # Set title
    if title is None:
        title = f"Feature Distributions ({n_features} features)"
    fig.suptitle(title, fontsize=fontsize+2, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_PCA(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_components: int = 2,
    figsize: Tuple[float, float] = (6, 4),
    point_labels: Optional[Sequence[str]] = None,
    color: str = 'orange',
    alpha: float = 0.7,
    fontsize: int = 12,
    title: Optional[str] = None,
    save: Optional[str] = None,
    show_plot: bool = True,
) -> Tuple[plt.Figure, np.ndarray, object]:
    """
    Create PCA visualization plots for dimensionality reduction analysis.
    
    Works with any DataFrame containing numeric data, including:
    - LHS design DataFrames from lhs_dataframe() or lhs_dataframe_optimized()
    - X DataFrames from split_XY()
    - Any other DataFrame with numeric columns
    
    Args:
        df: DataFrame containing the data
        columns: Specific columns to include (if None, uses all numeric columns)
        n_components: Number of principal components to compute (default: 2 for 2D visualization)
        figsize: Figure size as (width, height)
        point_labels: Labels for each data point (if None, uses indices)
        color: Color for the scatter plot points
        alpha: Transparency of the points
        fontsize: Font size for labels and title
        title: Plot title (if None, auto-generates)
        save: Path to save the figure (if None, doesn't save)
        show_plot: Whether to display the plot
        
    Returns:
        Tuple of (figure, pca_result, pca_object) where:
        - figure: The matplotlib figure
        - pca_result: The transformed data (N x n_components)
        - pca_object: The fitted PCA object for further analysis
        
    Example:
        >>> # For LHS dataframe
        >>> lhs_df = lhs_dataframe_optimized(design, n=10, max_abs_corr=0.3)
        >>> fig, pca_result, pca_obj = plot_PCA(lhs_df, title="LHS Design PCA")
        >>> 
        >>> # For X data from split_XY
        >>> X_df, Y_df = split_XY(csv_path, design, config)
        >>> fig, pca_result, pca_obj = plot_PCA(X_df, title="Experimental Data PCA")
        >>> 
        >>> # Access explained variance
        >>> print(f"Explained variance: {pca_obj.explained_variance_ratio_}")
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")
    
    # Select columns to analyze
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in DataFrame")
        columns = list(numeric_cols)
    else:
        # Verify all specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Filter DataFrame to selected columns
    df_subset = df[columns].copy()
    
    # Check if we have enough data for PCA
    if df_subset.shape[0] < n_components:
        raise ValueError(f"Not enough samples ({df_subset.shape[0]}) for {n_components} components")
    
    if df_subset.shape[1] < n_components:
        raise ValueError(f"Not enough features ({df_subset.shape[1]}) for {n_components} components")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_subset)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    if n_components == 2:
        # 2D scatter plot
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                           color=color, alpha=alpha, s=50)
        
        # Add point labels if provided
        if point_labels is not None:
            for i, (x, y) in enumerate(pca_result):
                ax.text(x, y, str(point_labels[i]), fontsize=fontsize*0.75, 
                       ha='right', va='bottom')
        else:
            # Use indices as labels
            for i, (x, y) in enumerate(pca_result):
                ax.text(x, y, str(i), fontsize=fontsize*0.75, 
                       ha='right', va='bottom')
        
        # Set labels
        ax.set_xlabel("Principal Component 1", fontsize=fontsize)
        ax.set_ylabel("Principal Component 2", fontsize=fontsize)
        
    elif n_components == 3:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                           color=color, alpha=alpha, s=50)
        
        # Add point labels if provided
        if point_labels is not None:
            for i, (x, y, z) in enumerate(pca_result):
                ax.text(x, y, z, str(point_labels[i]), fontsize=fontsize*0.75)
        else:
            # Use indices as labels
            for i, (x, y, z) in enumerate(pca_result):
                ax.text(x, y, z, str(i), fontsize=fontsize*0.75)
        
        # Set labels
        ax.set_xlabel("Principal Component 1", fontsize=fontsize)
        ax.set_ylabel("Principal Component 2", fontsize=fontsize)
        ax.set_zlabel("Principal Component 3", fontsize=fontsize)
        
    else:
        # For other numbers of components, create a 2D plot of first two components
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                           color=color, alpha=alpha, s=50)
        
        # Add point labels
        if point_labels is not None:
            for i, (x, y) in enumerate(pca_result):
                ax.text(x, y, str(point_labels[i]), fontsize=fontsize*0.75, 
                       ha='right', va='bottom')
        else:
            for i, (x, y) in enumerate(pca_result):
                ax.text(x, y, str(i), fontsize=fontsize*0.75, 
                       ha='right', va='bottom')
        
        ax.set_xlabel("Principal Component 1", fontsize=fontsize)
        ax.set_ylabel("Principal Component 2", fontsize=fontsize)
    
    # Set title
    if title is None:
        title = f"PCA of Data Points ({n_components} components)"
    ax.set_title(title, fontsize=fontsize)
    
    # Style the plot
    ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add explained variance information
    if n_components <= 3:
        var_text = f"Explained variance: {pca.explained_variance_ratio_[:n_components].sum():.3f}"
        ax.text(0.02, 0.98, var_text, transform=ax.transAxes, 
               fontsize=fontsize*0.8, va='top',
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, pca_result, pca


def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = 'orange',
    edgecolor: str = 'orange',
    alpha: float = 0.7,
    fontsize: int = 12,
    title: Optional[str] = None,
    save: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive grid of scatter plots for all feature pairs.
    
    Works with any DataFrame containing numeric data, including:
    - LHS design DataFrames from lhs_dataframe() or lhs_dataframe_optimized()
    - X DataFrames from split_XY()
    - Any other DataFrame with numeric columns
    
    Args:
        df: DataFrame containing the data
        columns: Specific columns to include (if None, uses all numeric columns)
        figsize: Figure size (if None, auto-calculates based on number of features)
        color: Color for the scatter plot points
        edgecolor: Edge color for the scatter plot points
        alpha: Transparency of the points
        fontsize: Font size for labels and title
        title: Plot title (if None, auto-generates)
        save: Path to save the figure (if None, doesn't save)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
        
    Example:
        >>> # For LHS dataframe
        >>> lhs_df = lhs_dataframe_optimized(design, n=10, max_abs_corr=0.3)
        >>> fig = plot_pairplot(lhs_df, title="LHS Design Pairwise Relationships")
        >>> 
        >>> # For X data from split_XY
        >>> X_df, Y_df = split_XY(csv_path, design, config)
        >>> fig = plot_pairplot(X_df, title="Experimental Data Pairwise Relationships")
        >>> 
        >>> # Custom columns and size
        >>> fig = plot_pairplot(X_df, columns=['speed_inorg', 'speed_org', 'temperature_c'], 
        >>>                     figsize=(15, 15))
    """
    # Select columns to analyze
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in DataFrame")
        columns = list(numeric_cols)
    else:
        # Verify all specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Filter DataFrame to selected columns
    df_subset = df[columns].copy()
    
    n_features = len(columns)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_features * 2.5, n_features * 2.5)
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid dimensions for triangular layout
    n_plots = (n_features * (n_features - 1)) // 2  # Total number of pairwise plots
    n_cols = n_features - 1  # Number of columns in the grid
    n_rows = (n_plots + n_cols - 1) // n_cols  # Number of rows needed
    
    # Create scatter plots for all feature pairs
    plot_index = 1
    for i in range(n_features):
        for j in range(i + 1, n_features):
            plt.subplot(n_rows, n_cols, plot_index)
            
            # Create scatter plot
            plt.scatter(df_subset[columns[i]], df_subset[columns[j]], 
                       color=color, edgecolor=edgecolor, alpha=alpha, s=30)
            
            # Set labels
            plt.xlabel(columns[i], fontsize=fontsize)
            plt.ylabel(columns[j], fontsize=fontsize)
            
            # Style the plot
            plt.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
            #plt.grid(True, alpha=0.3, linestyle='-')
            
            plot_index += 1
    
    # Set title
    if title is None:
        title = f"Pairwise Feature Relationships ({n_features} features)"
    plt.suptitle(title, y=0.98, fontsize=fontsize+2)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)
    
    # Save if requested
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig

def plot_bar(
    mean: np.ndarray,
    std: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    alpha: float = 0.7,
    capsize: float = 5,
    save: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Create bar plots for multiple objectives with error bars.
    
    Args:
        mean: Array of mean values, shape (N, M) where N is number of candidates, M is number of objectives
        std: Array of standard deviations, shape (N, M)
        labels: Optional sequence of objective names, defaults to ['obj0', 'obj1', ...]
        figsize: Optional figure size tuple, defaults to (5*M, 5)
        alpha: Transparency level for bars (0-1)
        capsize: Size of error bar caps
        save: Optional path to save the figure
        show_plot: Whether to display the plot using plt.show()
        
    Returns:
        fig: matplotlib figure object
    """
    # Input validation
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    
    if mean.shape != std.shape:
        raise ValueError(f"Shape mismatch: mean {mean.shape} vs std {std.shape}")
    
    if mean.ndim == 1:
        # Single objective case - convert to 2D
        mean = mean.reshape(-1, 1)
        std = std.reshape(-1, 1)
    elif mean.ndim != 2:
        raise ValueError(f"mean must be 1D or 2D array, got {mean.ndim}D")
    
    N, M = mean.shape
    
    # Set up labels
    if labels is None:
        labels = [f'obj{i}' for i in range(M)]
    elif len(labels) != M:
        raise ValueError(f"labels length {len(labels)} != M={M}")
    
    # Set up figure
    if figsize is None:
        figsize = (5*M, 5)
    
    fig, axes = plt.subplots(1, M, figsize=figsize)
    
    # Handle single subplot case
    if M == 1:
        axes = [axes]
    
    # Create bar plots
    for i in range(M):
        axes[i].bar(np.arange(N), mean[:, i], yerr=std[:, i], capsize=capsize, alpha=alpha)
        axes[i].set_title(labels[i], fontsize=16)
        axes[i].set_xlabel('Candidate Index', fontsize=12)
        axes[i].set_ylabel(labels[i], fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_shap(
    design: "Design",
    train_X: np.ndarray,
    model,
    objective_names: Optional[Sequence[str]] = None,
    save: Optional[str] = None,
    show_plot: bool = True,
):
    X_eval = np.asarray(train_X, dtype=float)
    feature_names = design.names
    M = getattr(model, "num_outputs", len(model.models))
    if objective_names is None:
        objective_names = [f"obj{i}" for i in range(M)]

    # SMALL background set helps Kernel SHAP; k-means is robust
    X_bg = shap.kmeans(X_eval, k=min(30, len(X_eval)))

    fig, axes = plt.subplots(1, M, figsize=(5 * M, 5))
    if M == 1:
        axes = [axes]

    for i, obj_name in enumerate(objective_names):
        print(f"Generating SHAP bar plot for {obj_name}...")
        gp = model.models[i]
        gp.eval()
        try:
            gp.likelihood.eval()
        except Exception:
            pass

        device = next(gp.parameters()).device
        dtype = next(gp.parameters()).dtype

        def f(x_np):
            x_np = np.asarray(x_np, dtype=float)
            if x_np.ndim == 1:
                x_np = x_np[None, :]
            x_t = torch.as_tensor(x_np, device=device, dtype=dtype)
            with torch.no_grad():
                # posterior mean is in ORIGINAL units (you use outcome_transform=Standardize)
                y = gp.posterior(x_t).mean.view(-1)
            return y.detach().cpu().numpy()

        explainer = shap.KernelExplainer(f, X_bg)
        shap_vals = explainer.shap_values(X_eval, nsamples=300)

        # In some SHAP versions, even scalar outputs come back as [array]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        mean_abs = np.nanmean(np.abs(shap_vals), axis=0)

        ax = axes[i]
        order = np.argsort(mean_abs)
        ax.barh(np.array(feature_names)[order], mean_abs[order])
        ax.set_title(obj_name, fontsize=13)
        ax.set_xlabel("Mean |SHAP value|")

        if np.allclose(mean_abs, 0):
            print(f"⚠️  SHAP values are ~0 for {obj_name}. "
                  f"The GP may be nearly flat or the background is too large/similar.")

    with torch.no_grad():
        mu = model.models[0].posterior(torch.as_tensor(train_X, dtype=next(model.parameters()).dtype,
                                                       device=next(model.parameters()).device)).mean
        print(f"pred mean range {obj_name}:", float(mu.min()), float(mu.max()))

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    return fig