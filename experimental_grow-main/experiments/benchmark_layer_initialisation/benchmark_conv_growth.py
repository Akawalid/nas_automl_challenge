"""
Comprehensive benchmark script for neural network initialization methods.
Tests different methods across various configurations with multiple seeds.
"""

import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from gromo.utils.utils import global_device, set_device

try:
    from experiments.benchmark_layer_initialisation.auxilliary_files import (
        gpu_memory_usage,
        random_network_iterator,
        score,
    )
    from experiments.benchmark_layer_initialisation.initialisation_methods import (
        compose_initialisation,
        create_random_layers,
        create_zero_layers,
        full_s_optimized_conv_layers,
        natural_gradient_step_layers,
        norm_approx_optimized_conv_layers,
        restricted_fogro_optimized_conv_layers,
        sgd_optimized_layers,
    )
except ImportError as e:
    print(
        "Error importing modules. Try python -m experiments.benchmark_layer_initialisation.benchmark_conv_growth"
    )
    raise e

# Set device
# set_device(torch.device("cpu"))
DEVICE = global_device()
print(f"Using device: {DEVICE}")


def benchmark_single_method(
    method: Callable,
    method_name: str,
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    device: torch.device = DEVICE,
    non_linearity: nn.Module = None,
) -> tuple[float, float]:
    """
    Benchmark a single method and return score and execution time.

    Returns:
    --------
    tuple[float, float]: (score, execution_time_seconds)
    """
    try:
        # Create layers using the method

        print(f"Running {method_name}...")
        print("GPU USAGE:")
        gpu_memory_usage()

        start_time = time.perf_counter()
        layer1, layer2, _ = method(
            layer_1_parameters,
            layer_2_parameters,
            data_iterator=data_iterator,
            device=device,
        )
        execution_time = time.perf_counter() - start_time

        # Create network
        if non_linearity is None:
            non_linearity = nn.Identity()
        net = nn.Sequential(layer1, non_linearity, layer2).to(device)

        # Calculate score
        method_score = score(net, data_iterator, device)

    # except Exception as e:
    except ImportError as e:
        print(f"Error in {method_name}: {e}")
        method_score = float("inf")
        execution_time = -1

    return method_score, execution_time


def create_sgd_composed_method(first_method, first_method_kwargs=None, sgd_kwargs=None):
    if first_method_kwargs is None:
        first_method_kwargs = {}
    if sgd_kwargs is None:
        sgd_kwargs = {}

    def composed_method(
        layer_1_parameters, layer_2_parameters, data_iterator, device, **kwargs
    ):
        return compose_initialisation(
            methods=[first_method, sgd_optimized_layers],
            layer_1_parameters=layer_1_parameters,
            layer_2_parameters=layer_2_parameters,
            kwargs_list=[first_method_kwargs, sgd_kwargs],
            data_iterator=data_iterator,
            device=device,
            **kwargs,
        )

    return composed_method


def create_ng_composed_method(first_method, first_method_kwargs=None, ng_kwargs=None):
    if first_method_kwargs is None:
        first_method_kwargs = {}
    if ng_kwargs is None:
        ng_kwargs = {}

    def composed_method(
        layer_1_parameters, layer_2_parameters, data_iterator, device, **kwargs
    ):
        return compose_initialisation(
            methods=[first_method, natural_gradient_step_layers],
            layer_1_parameters=layer_1_parameters,
            layer_2_parameters=layer_2_parameters,
            kwargs_list=[first_method_kwargs, ng_kwargs],
            data_iterator=data_iterator,
            device=device,
            **kwargs,
        )

    return composed_method


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different configurations."""

    non_linearity = nn.Softplus()  # Non-linearity to use in the network

    # Configuration parameters
    configurations = [
        # (in_channels, intermediate_channels, out_channels, description)
        # (10, 5, 15, "small"),
        # (20, 10, 30, "medium"),
        (50, 10, 50, "50-10"),
        # (50, 50, 50, "large"),
        (100, 10, 100, "100-10"),
        # (100, 100, 100, "medium"),
        (200, 10, 200, "200-10"),
        # (200, 100, 200, "200-100"),
        (300, 10, 200, "300-10"),
    ]

    noise_levels = [0.0]
    seeds = list(range(5))  # 5 different seeds for each configuration

    # Methods to benchmark
    methods = {
        "random (baseline)": create_random_layers,
        "zeros": create_zero_layers,
        "restricted_fogro": restricted_fogro_optimized_conv_layers,
        "NG": natural_gradient_step_layers,
        "restricted_fogro (N) + NG": create_ng_composed_method(
            restricted_fogro_optimized_conv_layers,
            first_method_kwargs={"normalize_layers": True},
            ng_kwargs={"activation_function": non_linearity},
        ),
        # "fogro": norm_approx_optimized_conv_layers,
        # "full_s": full_s_optimized_conv_layers,
        # "AdamW(1e-1, 10)": lambda *args, **kwargs: sgd_optimized_layers(
        #     *args, **kwargs, lr=0.1, epochs=10, optimizer="Adam"
        # ),
        # "AdamW(1e-2, 10)": lambda *args, **kwargs: sgd_optimized_layers(
        #     *args, **kwargs, lr=0.01, epochs=10, optimizer="Adam"
        # ),
        # "AdamW(1e-3, 10)": lambda *args, **kwargs: sgd_optimized_layers(
        #     *args, **kwargs, lr=0.001, epochs=10, optimizer="Adam"
        # ),
        # "restricted_fogro->AdamW(1e-1, 10)": create_sgd_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     sgd_kwargs={"lr": 0.1, "epochs": 10},
        # ),
        # "restricted_fogro (N)->AdamW(1e-1, 10)": create_sgd_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     first_method_kwargs={"normalize_layers": True},
        #     sgd_kwargs={"lr": 0.1, "epochs": 10,},
        # ),
        # "fogro->AdamW(1e-1, 10)": create_sgd_composed_method(
        #     norm_approx_optimized_conv_layers,
        #     sgd_kwargs={"lr": 0.1, "epochs": 10},
        # ),
        # "full_s->AdamW(1e-1, 10)": create_sgd_composed_method(
        #     full_s_optimized_conv_layers,
        #     sgd_kwargs={"lr": 0.1, "epochs": 10},
        # ),
        # "NL AdamW(1e-1, 10)": lambda *args, **kwargs: sgd_optimized_layers(
        #     *args,
        #     **kwargs,
        #     lr=0.1,
        #     epochs=10,
        #     optimizer="Adam",
        #     activation_function=non_linearity,
        # ),
        # "restricted_fogro -> NL AdamW(1e-1, 10)": create_sgd_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     first_method_kwargs={"normalize_layers": False},
        #     sgd_kwargs={"lr": 0.1, "epochs": 10, "activation_function": non_linearity},
        # ),
        # "restricted_fogro (N) -> NL AdamW(1e-1, 10)": create_sgd_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     first_method_kwargs={"normalize_layers": True},
        #     sgd_kwargs={"lr": 0.1, "epochs": 10, "activation_function": non_linearity},
        # ),
        "NL AdamW(1e-2, 1)": lambda *args, **kwargs: sgd_optimized_layers(
            *args,
            **kwargs,
            lr=0.01,
            epochs=1,
            optimizer="Adam",
            activation_function=non_linearity,
        ),
        "NL AdamW(1e-2, 10)": lambda *args, **kwargs: sgd_optimized_layers(
            *args,
            **kwargs,
            lr=0.01,
            epochs=10,
            optimizer="Adam",
            activation_function=non_linearity,
        ),
        # "NL AdamW(1e-3, 10)": lambda *args, **kwargs: sgd_optimized_layers(
        #     *args,
        #     **kwargs,
        #     lr=0.001,
        #     epochs=10,
        #     optimizer="Adam",
        #     activation_function=non_linearity,
        # ),
        # "restricted_fogro -> NL AdamW(1e-2, 10)": create_sgd_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     first_method_kwargs={"normalize_layers": False},
        #     sgd_kwargs={"lr": 0.01, "epochs": 10, "activation_function": non_linearity},
        # ),
        # "restricted_fogro (N) -> NL AdamW(1e-2, 10)": create_sgd_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     first_method_kwargs={"normalize_layers": True},
        #     sgd_kwargs={"lr": 0.01, "epochs": 10, "activation_function": non_linearity},
        # ),
        # "NL - fogro->AdamW(1e-1, 10)": create_sgd_composed_method(
        #     norm_approx_optimized_conv_layers,
        #     sgd_kwargs={"lr": 0.1, "epochs": 10, "activation_function": non_linearity},
        # ),
        # "NL - full_s->AdamW(1e-1, 10)": create_sgd_composed_method(
        #     full_s_optimized_conv_layers,
        #     sgd_kwargs={"lr": 0.1, "epochs": 10, "activation_function": non_linearity},
        # ),
    }

    results = []

    total_experiments = (
        len(configurations) * len(noise_levels) * len(seeds) * len(methods)
    )
    experiment_count = 0

    for config_idx, (in_ch, inter_ch, out_ch, size_desc) in enumerate(configurations):
        print(f"\nTesting configuration: {size_desc} ({in_ch}->{inter_ch}->{out_ch})")

        for noise_level in noise_levels:
            print(f"  Noise level: {noise_level}")

            for seed in seeds:
                print(f"    Seed: {seed}")

                # Set up layer parameters
                c1_params = {
                    "type": nn.Conv2d,
                    "in_channels": in_ch,
                    "out_channels": inter_ch,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False,
                }
                c2_params = {
                    "type": nn.Conv2d,
                    "in_channels": inter_ch,
                    "out_channels": out_ch,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False,
                }

                # Create target network for data generation
                torch.manual_seed(seed)
                target_c1 = nn.Conv2d(
                    in_ch,
                    inter_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=DEVICE,
                    bias=False,
                )
                target_c2 = nn.Conv2d(
                    inter_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=DEVICE,
                    bias=False,
                )
                target_net = nn.Sequential(target_c1, non_linearity, target_c2)

                pre_net = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, device=DEVICE),
                    # non_linearity
                )

                # Create data iterator
                data_iterator = random_network_iterator(
                    f=target_net,
                    p=pre_net,
                    in_size=(in_ch, 16, 16),  # Fixed spatial size
                    batch_size=128,
                    n_samples=100,
                    noise_level=noise_level,
                    device=DEVICE,
                    seed=seed,
                )

                # Test each method
                for method_name, method in methods.items():
                    experiment_count += 1
                    print(
                        f"      Method: {method_name} ({experiment_count}/{total_experiments})"
                    )

                    method_score, execution_time = benchmark_single_method(
                        method,
                        method_name,
                        data_iterator,
                        c1_params,
                        c2_params,
                        DEVICE,
                        non_linearity=non_linearity,
                    )

                    results.append(
                        {
                            "method": method_name,
                            "configuration": size_desc,
                            "in_channels": in_ch,
                            "intermediate_channels": inter_ch,
                            "out_channels": out_ch,
                            "noise_level": noise_level,
                            "seed": seed,
                            "score": method_score,
                            "execution_time": execution_time,
                        }
                    )

                    print(
                        f"        Score: {method_score:.3e}, Time: {execution_time:.3f}s",
                        flush=True,
                    )

    return pd.DataFrame(results)


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Create various plots from the benchmark results."""

    output_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Score vs Time scatter plot for each configuration
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    configurations = df["configuration"].unique()
    for idx, config in enumerate(configurations):
        ax = axes[idx]
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
        for _, row in grouped.iterrows():
            ax.errorbar(
                row["time_mean"],
                row["score_mean"],
                xerr=row["time_std"],
                yerr=row["score_std"],
                label=row["method"],
                marker="o",
                capsize=5,
            )

        ax.set_xlabel("Execution Time (s)")
        ax.set_ylabel("Score")
        ax.set_title(f"Score vs Time - {config}")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "score_vs_time_by_config.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Score vs Configuration size for each method
    plt.figure(figsize=(12, 8))

    # Calculate means and stds across all noise levels and seeds
    config_comparison = (
        df.groupby(["method", "configuration"])
        .agg({"score": ["mean", "std"]})
        .reset_index()
    )
    config_comparison.columns = ["method", "configuration", "score_mean", "score_std"]

    methods = df["method"].unique()
    x_pos = np.arange(len(configurations))
    width = 0.8 / len(methods)

    for idx, method in enumerate(methods):
        method_data = config_comparison[config_comparison["method"] == method]
        plt.bar(
            x_pos + idx * width,
            method_data["score_mean"],
            width,
            yerr=method_data["score_std"],
            label=method,
            capsize=5,
            alpha=0.8,
        )

    plt.xlabel("Configuration")
    plt.ylabel("Score")
    plt.title("Score vs Configuration Size")
    plt.yscale("log")
    plt.xticks(x_pos + width * (len(methods) - 1) / 2, configurations)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "score_vs_configuration.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Effect of noise level
    plt.figure(figsize=(12, 8))

    noise_comparison = (
        df.groupby(["method", "noise_level"])
        .agg({"score": ["mean", "std"]})
        .reset_index()
    )
    noise_comparison.columns = ["method", "noise_level", "score_mean", "score_std"]

    for method in methods:
        method_data = noise_comparison[noise_comparison["method"] == method]
        plt.errorbar(
            method_data["noise_level"],
            method_data["score_mean"],
            yerr=method_data["score_std"],
            label=method,
            marker="o",
            capsize=5,
        )

    plt.xlabel("Noise Level")
    plt.ylabel("Score")
    plt.title("Effect of Noise Level on Performance")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "score_vs_noise.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 4. Execution time comparison
    plt.figure(figsize=(12, 8))

    time_comparison = (
        df.groupby(["method", "configuration"])
        .agg({"execution_time": ["mean", "std"]})
        .reset_index()
    )
    time_comparison.columns = ["method", "configuration", "time_mean", "time_std"]

    x_pos = np.arange(len(configurations))
    for idx, method in enumerate(methods):
        method_data = time_comparison[time_comparison["method"] == method]
        plt.bar(
            x_pos + idx * width,
            method_data["time_mean"],
            width,
            yerr=method_data["time_std"],
            label=method,
            capsize=5,
            alpha=0.8,
        )

    plt.xlabel("Configuration")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Configuration Size")
    plt.yscale("log")
    plt.xticks(x_pos + width * (len(methods) - 1) / 2, configurations)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_configuration.png", dpi=300, bbox_inches="tight")
    plt.show()


def save_results(df: pd.DataFrame, output_dir: Path):
    """Save results to CSV and summary statistics."""

    output_dir.mkdir(exist_ok=True)

    # Save full results
    df.to_csv(output_dir / "benchmark_results.csv", index=False)

    # Save summary statistics
    summary = (
        df.groupby(["method", "configuration", "noise_level"])
        .agg(
            {
                "score": ["mean", "std", "min", "max"],
                "execution_time": ["mean", "std", "min", "max"],
            }
        )
        .round(6)
    )

    summary.to_csv(output_dir / "benchmark_summary.csv")

    # Save best methods for each configuration
    best_methods = df.groupby(["configuration", "noise_level", "seed"])["score"].idxmin()
    best_df = df.loc[best_methods][
        ["method", "configuration", "noise_level", "seed", "score"]
    ]
    best_summary = (
        best_df.groupby(["method", "configuration", "noise_level"])
        .size()
        .reset_index(name="wins")
    )
    best_summary.to_csv(output_dir / "best_methods.csv", index=False)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    print(f"Running comprehensive benchmark on device: {DEVICE}")
    print("This may take a while...")

    # Run benchmark
    results_df = run_comprehensive_benchmark()

    # Create output directory
    output_dir = Path("benchmark_results")

    # Save results
    save_results(results_df, output_dir)

    # Create plots
    create_plots(results_df, output_dir)

    print("\nBenchmark completed!")
    print(f"Results and plots saved to: {output_dir.absolute()}")

    # Print quick summary
    print("\nQuick Summary:")
    print("=" * 50)

    avg_scores = results_df.groupby("method")["score"].mean().sort_values()
    print("Average scores across all configurations:")
    for method, score in avg_scores.items():
        print(f"  {method}: {score:.3e}")

    avg_times = results_df.groupby("method")["execution_time"].mean().sort_values()
    print("\nAverage execution times:")
    for method, time_val in avg_times.items():
        print(f"  {method}: {time_val:.3f}s")
