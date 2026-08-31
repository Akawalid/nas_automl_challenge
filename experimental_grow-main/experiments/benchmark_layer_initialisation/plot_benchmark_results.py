"""
Plotting script for benchmark results from benchmark_results.csv
Creates comprehensive visualizations of neural network initialization method performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_and_clean_data(csv_path: str, exclude_random: bool = False) -> pd.DataFrame:
    """Load and clean the benchmark results data."""
    df = pd.read_csv(csv_path)

    # Remove any infinite or invalid scores
    df = df[df["score"] != float("inf")]
    df = df[df["execution_time"] > 0]

    # Optionally exclude random baseline
    if exclude_random:
        df = df[~df["method"].str.contains("random", case=False)]
        print("Excluded random baseline methods from analysis")

    print(f"Loaded {len(df)} valid benchmark results")
    print(f"Methods: {df['method'].unique()}")
    print(f"Configurations: {df['configuration'].unique()}")
    print(f"Noise levels: {df['noise_level'].unique()}")

    return df


def create_score_vs_time_plots(
    df: pd.DataFrame, output_dir: Path, exclude_random: bool = False
):
    """Create score vs execution time plots for each configuration."""

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    configurations = sorted(df["configuration"].unique())
    n_configs = len(configurations)

    # Create subplots
    if n_configs <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    else:
        ncols = int(np.ceil(np.sqrt(n_configs)))
        nrows = int(np.ceil(n_configs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

    for idx, config in enumerate(configurations):
        ax = axes[idx] if n_configs > 1 else axes
        config_data = df[df["configuration"] == config]

        # Group by method and calculate means and std
        grouped = (
            config_data.groupby("method")
            .agg({"score": ["mean", "std"], "execution_time": ["mean", "std"]})
            .reset_index()
        )

        # Flatten column names
        grouped.columns = ["method", "score_mean", "score_std", "time_mean", "time_std"]

        # Plot with error bars
        colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
        for i, (_, row) in enumerate(grouped.iterrows()):
            ax.errorbar(
                row["time_mean"],
                row["score_mean"],
                xerr=row["time_std"],
                yerr=row["score_std"],
                label=row["method"],
                marker="o",
                capsize=5,
                color=colors[i],
                markersize=8,
                linewidth=2,
            )

        ax.set_xlabel("Execution Time (s)")
        ax.set_ylabel("Score (lower is better)")
        title_suffix = " (excl. random)" if exclude_random else ""
        ax.set_title(f"Score vs Time - {config}{title_suffix}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Only add legend to first subplot to avoid clutter
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    filename = (
        "score_vs_time_by_config_no_random.png"
        if exclude_random
        else "score_vs_time_by_config.png"
    )
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def create_score_comparison_plot(
    df: pd.DataFrame, output_dir: Path, exclude_random: bool = False
):
    """Create bar plot comparing scores across configurations."""

    plt.figure(figsize=(14, 8))

    # Calculate means and stds across all noise levels and seeds
    config_comparison = (
        df.groupby(["method", "configuration"])
        .agg({"score": ["mean", "std"]})
        .reset_index()
    )
    config_comparison.columns = ["method", "configuration", "score_mean", "score_std"]

    methods = df["method"].unique()
    configurations = sorted(df["configuration"].unique())
    x_pos = np.arange(len(configurations))
    width = 0.8 / len(methods)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for idx, method in enumerate(methods):
        method_data = config_comparison[config_comparison["method"] == method]
        # Ensure data is in same order as configurations
        method_data = (
            method_data.set_index("configuration").reindex(configurations).reset_index()
        )

        plt.bar(
            x_pos + idx * width,
            method_data["score_mean"],
            width,
            yerr=method_data["score_std"],
            label=method,
            capsize=5,
            alpha=0.8,
            color=colors[idx],
        )

    plt.xlabel("Configuration")
    plt.ylabel("Score (lower is better)")
    title_suffix = " (excl. random)" if exclude_random else ""
    plt.title(f"Score Comparison Across Configurations{title_suffix}")
    # plt.yscale("log")
    plt.xticks(x_pos + width * (len(methods) - 1) / 2, configurations)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    filename = (
        "score_vs_configuration_no_random.png"
        if exclude_random
        else "score_vs_configuration.png"
    )
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def create_noise_effect_plot(
    df: pd.DataFrame, output_dir: Path, exclude_random: bool = False
):
    """Create plot showing effect of noise level."""

    if len(df["noise_level"].unique()) == 1:
        print("Only one noise level found, skipping noise effect plot")
        return

    plt.figure(figsize=(12, 8))

    noise_comparison = (
        df.groupby(["method", "noise_level"])
        .agg({"score": ["mean", "std"]})
        .reset_index()
    )
    noise_comparison.columns = ["method", "noise_level", "score_mean", "score_std"]

    methods = df["method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for idx, method in enumerate(methods):
        method_data = noise_comparison[noise_comparison["method"] == method]
        plt.errorbar(
            method_data["noise_level"],
            method_data["score_mean"],
            yerr=method_data["score_std"],
            label=method,
            marker="o",
            capsize=5,
            color=colors[idx],
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Noise Level")
    plt.ylabel("Score (lower is better)")
    title_suffix = " (excl. random)" if exclude_random else ""
    plt.title(f"Effect of Noise Level on Performance{title_suffix}")
    # plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = "score_vs_noise_no_random.png" if exclude_random else "score_vs_noise.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def create_execution_time_plot(
    df: pd.DataFrame, output_dir: Path, exclude_random: bool = False
):
    """Create execution time comparison plot."""

    plt.figure(figsize=(14, 8))

    time_comparison = (
        df.groupby(["method", "configuration"])
        .agg({"execution_time": ["mean", "std"]})
        .reset_index()
    )
    time_comparison.columns = ["method", "configuration", "time_mean", "time_std"]

    methods = df["method"].unique()
    configurations = sorted(df["configuration"].unique())
    x_pos = np.arange(len(configurations))
    width = 0.8 / len(methods)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for idx, method in enumerate(methods):
        method_data = time_comparison[time_comparison["method"] == method]
        # Ensure data is in same order as configurations
        method_data = (
            method_data.set_index("configuration").reindex(configurations).reset_index()
        )

        plt.bar(
            x_pos + idx * width,
            method_data["time_mean"],
            width,
            yerr=method_data["time_std"],
            label=method,
            capsize=5,
            alpha=0.8,
            color=colors[idx],
        )

    plt.xlabel("Configuration")
    plt.ylabel("Execution Time (s)")
    title_suffix = " (excl. random)" if exclude_random else ""
    plt.title(f"Execution Time Comparison{title_suffix}")
    # plt.yscale("log")
    plt.xticks(x_pos + width * (len(methods) - 1) / 2, configurations)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    filename = (
        "time_vs_configuration_no_random.png"
        if exclude_random
        else "time_vs_configuration.png"
    )
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def create_heatmap_plots(
    df: pd.DataFrame, output_dir: Path, exclude_random: bool = False
):
    """Create heatmap visualizations."""

    # Score heatmap
    plt.figure(figsize=(12, 8))

    # Pivot data for heatmap
    score_pivot = df.groupby(["method", "configuration"])["score"].mean().unstack()

    sns.heatmap(
        score_pivot,
        annot=True,
        cmap="viridis_r",  # Reverse viridis so darker = better (lower score)
        fmt=".3f",
        cbar_kws={"label": "Score (lower is better)"},
    )

    title_suffix = " (excl. random)" if exclude_random else ""
    plt.title(f"Score Heatmap: Methods vs Configurations{title_suffix}")
    plt.xlabel("Configuration")
    plt.ylabel("Method")
    plt.tight_layout()
    filename = "score_heatmap_no_random.png" if exclude_random else "score_heatmap.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()

    # Time heatmap
    plt.figure(figsize=(12, 8))

    time_pivot = (
        df.groupby(["method", "configuration"])["execution_time"].mean().unstack()
    )

    sns.heatmap(
        time_pivot,
        annot=True,
        cmap="plasma",
        fmt=".2f",
        cbar_kws={"label": "Execution Time (s)"},
    )

    plt.title(f"Execution Time Heatmap: Methods vs Configurations{title_suffix}")
    plt.xlabel("Configuration")
    plt.ylabel("Method")
    plt.tight_layout()
    filename = "time_heatmap_no_random.png" if exclude_random else "time_heatmap.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()


def create_ranking_plot(df: pd.DataFrame, output_dir: Path, exclude_random: bool = False):
    """Create ranking visualization showing best methods."""

    # Calculate rankings based on score for each configuration
    rankings = []

    for config in df["configuration"].unique():
        config_data = df[df["configuration"] == config]
        avg_scores = config_data.groupby("method")["score"].mean().sort_values()

        for rank, (method, score) in enumerate(avg_scores.items(), 1):
            rankings.append(
                {"configuration": config, "method": method, "rank": rank, "score": score}
            )

    ranking_df = pd.DataFrame(rankings)

    plt.figure(figsize=(12, 8))

    # Create ranking plot
    ranking_pivot = ranking_df.pivot(
        index="method", columns="configuration", values="rank"
    )

    sns.heatmap(
        ranking_pivot,
        annot=True,
        cmap="RdYlGn_r",  # Red for high rank (bad), green for low rank (good)
        fmt="d",
        cbar_kws={"label": "Rank (1 = best)"},
    )

    title_suffix = " (excl. random)" if exclude_random else ""
    plt.title(f"Method Rankings by Configuration (1 = best score){title_suffix}")
    plt.xlabel("Configuration")
    plt.ylabel("Method")
    plt.tight_layout()
    filename = (
        "method_rankings_no_random.png" if exclude_random else "method_rankings.png"
    )
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()

    return ranking_df


def print_summary_statistics(df: pd.DataFrame, ranking_df: pd.DataFrame):
    """Print summary statistics."""

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    # Overall best methods
    print("\nOverall Method Performance (average score):")
    avg_scores = df.groupby("method")["score"].mean().sort_values()
    for i, (method, score) in enumerate(avg_scores.items(), 1):
        print(f"  {i:2d}. {method}: {score:.3e}")

    # Average execution times
    print("\nAverage Execution Times:")
    avg_times = df.groupby("method")["execution_time"].mean().sort_values()
    for method, time_val in avg_times.items():
        print(f"  {method}: {time_val:.3f}s")

    # Best method for each configuration
    print("\nBest Method by Configuration:")
    for config in sorted(df["configuration"].unique()):
        config_data = df[df["configuration"] == config]
        best_method = config_data.groupby("method")["score"].mean().idxmin()
        best_score = config_data.groupby("method")["score"].mean().min()
        print(f"  {config}: {best_method} (score: {best_score:.3e})")

    # Winner count
    print("\nNumber of configurations where each method ranks #1:")
    winner_counts = ranking_df[ranking_df["rank"] == 1]["method"].value_counts()
    for method, count in winner_counts.items():
        print(f"  {method}: {count}/{len(df['configuration'].unique())} configurations")


def main(exclude_random: bool = False):
    """Main function to create all plots."""

    # File paths
    csv_path = "benchmark_results/benchmark_results.csv"
    output_dir = Path("benchmark_plots")
    output_dir.mkdir(exist_ok=True)

    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        print(
            "Please make sure the benchmark_results.csv file is in the benchmark_results/ directory"
        )
        return

    # Load data
    print("Loading benchmark results...")
    df = load_and_clean_data(csv_path, exclude_random=exclude_random)

    # Create plots
    print("\nCreating plots...")

    print("  - Score vs Time plots...")
    create_score_vs_time_plots(df, output_dir, exclude_random=exclude_random)

    print("  - Score comparison plot...")
    create_score_comparison_plot(df, output_dir, exclude_random=exclude_random)

    print("  - Noise effect plot...")
    create_noise_effect_plot(df, output_dir, exclude_random=exclude_random)

    print("  - Execution time plot...")
    create_execution_time_plot(df, output_dir, exclude_random=exclude_random)

    print("  - Heatmap plots...")
    create_heatmap_plots(df, output_dir, exclude_random=exclude_random)

    print("  - Ranking plot...")
    ranking_df = create_ranking_plot(df, output_dir, exclude_random=exclude_random)

    # Print summary
    print_summary_statistics(df, ranking_df)

    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("\nPlot files created:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create benchmark plots from CSV results"
    )
    parser.add_argument(
        "--exclude-random",
        action="store_true",
        help="Exclude random baseline methods from plots",
    )

    args = parser.parse_args()
    main(exclude_random=args.exclude_random)
