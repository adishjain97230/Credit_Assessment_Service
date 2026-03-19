import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import constants, switch_properties, logging_config
import os

module_properties = switch_properties.SWITCH_PROPERTIES[constants.eda]
logger = logging_config.get_logger(__name__)

def saveHistograms(df):
    df.hist(bins=30, figsize=(12, 10))
    plt.tight_layout()
    plt.savefig(module_properties[constants.histograms_path])
    plt.close()

def saveBargraphs(df):
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    n_cols = len(str_cols)

    if n_cols == 0:
        print("No string columns")
    else:
        fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(12, 4 * ((n_cols + 1) // 2)))
        axes = axes.flatten() if n_cols > 1 else [axes]
        for ax, col in zip(axes, str_cols):
            df[col].value_counts().head(20).plot(kind="bar", ax=ax)  # top 20 categories
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)
        for j in range(len(str_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(module_properties[constants.bargraphs_path], dpi=150, bbox_inches="tight")
        plt.close()

def saveViolinPlots(df):
    num_cols = df.select_dtypes(include=["number"]).columns

    n = len(num_cols)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 8 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for ax, col in zip(axes, num_cols):
        sns.violinplot(y=df[col], ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(module_properties[constants.violin_plots_path], dpi=150, bbox_inches="tight")
    plt.close()

def saveMeanEncodedHeatmap(df, target="y"):
    """
    One big correlation heat map: numeric columns + categorical columns
    mean-encoded by target. Every cell = Pearson correlation.
    """
    if target not in df.columns:
        print(f"Target '{target}' not in DataFrame")
        return

    # Numeric columns (exclude target so it's not duplicated as predictor)
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Build combined matrix: numerics + mean-encoded categoricals
    combined = df[num_cols].copy()

    for col in cat_cols:
        # Mean encoding: replace category with mean(target) for that category
        combined[f"{col}_mean_enc"] = df.groupby(col)[target].transform("mean")
    combined[target] = df[target]

    if combined.shape[1] < 2:
        print("Need at least 2 columns for heat map")
        return

    corr = combined.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(module_properties[constants.heatmap_path], dpi=150, bbox_inches="tight")
    plt.close()

def main(**kwargs):
    repeat_all_parts = kwargs.get(constants.repeat_all_parts, True)

    if not(repeat_all_parts) and os.path.exists(module_properties[constants.heatmap_path]):
        logger.info("EDA already completed. Skipping...")
        return;

    df = pd.read_parquet(module_properties[constants.dataset_path])

    if repeat_all_parts or not(os.path.exists(module_properties[constants.histograms_path])):
        saveHistograms(df)
    else:
        logger.info("Histograms already completed. Skipping...")

    if repeat_all_parts or not(os.path.exists(module_properties[constants.bargraphs_path])):
        saveBargraphs(df)
    else:
        logger.info("Bargraphs already completed. Skipping...")

    if repeat_all_parts or not(os.path.exists(module_properties[constants.violin_plots_path])):
        saveViolinPlots(df)
    else:
        logger.info("Violin plots already completed. Skipping...")

    if repeat_all_parts or not(os.path.exists(module_properties[constants.heatmap_path])):
        saveMeanEncodedHeatmap(df)
    else:
        logger.info("Heatmap already completed. Skipping...")

if __name__ == "__main__":
    main()

