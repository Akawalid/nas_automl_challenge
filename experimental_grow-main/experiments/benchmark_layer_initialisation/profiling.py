"""
Simple profiling script for a single initialization method.
Provides detailed classical profiling using cProfile and line_profiler.
"""

import cProfile
import io
import pstats
import time
import tracemalloc
from pathlib import Path

import torch
import torch.nn as nn
from gromo.utils.utils import global_device

try:
    from experiments.benchmark_layer_initialisation.auxilliary_files import (
        gpu_memory_usage,
        random_network_iterator,
        score,
    )
    from experiments.benchmark_layer_initialisation.initialisation_methods import (
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
        "Error importing modules. Try python -m experiments.benchmark_layer_initialisation.simple_profiling"
    )
    raise e

DEVICE = global_device()


def setup_test_case(config: str = "medium", activation_function: nn.Module = nn.ReLU()):
    """Setup a single test case configuration."""

    configs = {
        "tiny": (10, 5, 15),
        "small": (20, 10, 30),
        "medium": (50, 10, 50),
        "100-10": (100, 10, 100),
        "200-10": (200, 10, 200),
    }

    in_ch, inter_ch, out_ch = configs[config]

    print(f"Configuration: {config} ({in_ch} -> {inter_ch} -> {out_ch})")

    # Layer parameters
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
    torch.manual_seed(42)
    target_c1 = nn.Conv2d(
        in_ch, inter_ch, kernel_size=3, stride=1, padding=1, device=DEVICE, bias=False
    )
    target_c2 = nn.Conv2d(
        inter_ch, out_ch, kernel_size=3, stride=1, padding=1, device=DEVICE, bias=False
    )
    target_net = nn.Sequential(target_c1, activation_function, target_c2)

    # Create data iterator
    data_iterator = random_network_iterator(
        f=target_net,
        p=None,
        in_size=(in_ch, 16, 16),
        batch_size=64,
        n_samples=100,  # Smaller for profiling focus
        noise_level=0.0,
        device=DEVICE,
        seed=42,
    )

    return c1_params, c2_params, data_iterator


def profile_with_cprofile(
    method, method_name, c1_params, c2_params, data_iterator, output_dir
):
    """Profile using cProfile for detailed function-level analysis."""

    function_calls_to_show = 100  # Number of function calls to show in memory profiling

    print(f"\n{'='*60}")
    print(f"CPROFILING: {method_name}")
    print(f"{'='*60}")

    # Create profiler
    profiler = cProfile.Profile()

    # Profile the method execution
    print("Running method with cProfile...")
    profiler.enable()

    layer1, layer2, extra_output = method(
        c1_params, c2_params, data_iterator=data_iterator, device=DEVICE
    )

    profiler.disable()

    # Create network and calculate score
    net = nn.Sequential(layer1, nn.ReLU(), layer2).to(DEVICE)
    method_score = score(net, data_iterator, DEVICE)

    print(f"Method completed. Score: {method_score:.3e}")

    # Save detailed profile
    output_dir.mkdir(exist_ok=True)

    # Text-based detailed report
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    # Sort by cumulative time and show top functions
    ps.sort_stats("cumulative")
    ps.print_stats(function_calls_to_show)  # Top 50 functions

    profile_file = output_dir / f"cprofile_{method_name.replace(' ', '_')}_detailed.txt"
    with open(profile_file, "w") as f:
        f.write(f"PROFILE REPORT FOR: {method_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Final Score: {method_score:.3e}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(
            f"Configuration: {c1_params['in_channels']} -> {c1_params['out_channels']} -> {c2_params['out_channels']}\n"
        )
        f.write(f"{'='*60}\n\n")
        f.write(f"TOP {function_calls_to_show} FUNCTIONS BY CUMULATIVE TIME:\n")
        f.write("-" * 60 + "\n")
        f.write(s.getvalue())

    # Alternative sorting by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats("tottime")
    ps2.print_stats(function_calls_to_show)

    with open(profile_file, "a") as f:
        f.write(f"\n\n{'='*60}\n")
        f.write(f"TOP {function_calls_to_show} FUNCTIONS BY TOTAL TIME:\n")
        f.write("-" * 60 + "\n")
        f.write(s2.getvalue())

    # Function call statistics
    s3 = io.StringIO()
    ps3 = pstats.Stats(profiler, stream=s3)
    ps3.sort_stats("calls")
    ps3.print_stats(function_calls_to_show)

    with open(profile_file, "a") as f:
        f.write(f"\n\n{'='*60}\n")
        f.write(f"TOP {function_calls_to_show} FUNCTIONS BY NUMBER OF CALLS:\n")
        f.write("-" * 60 + "\n")
        f.write(s3.getvalue())

    # Save binary profile for external tools
    binary_profile_file = output_dir / f"cprofile_{method_name.replace(' ', '_')}.prof"
    profiler.dump_stats(str(binary_profile_file))

    print(f"Detailed profile saved to: {profile_file}")
    print(f"Binary profile saved to: {binary_profile_file}")
    print(f"  (Use 'python -m pstats {binary_profile_file}' for interactive analysis)")

    # Print key statistics to console
    stats = pstats.Stats(profiler)
    total_calls = stats.total_calls
    total_time = stats.total_tt

    print(f"\nProfile Summary:")
    print(f"  Total function calls: {total_calls:,}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Final score: {method_score:.3e}")

    # Top 10 most time-consuming functions
    print(f"\nTop {function_calls_to_show} functions by cumulative time:")
    stats.sort_stats("cumulative")
    stats.print_stats(function_calls_to_show)

    return method_score, total_time


def profile_memory_usage(
    method, method_name, c1_params, c2_params, data_iterator, output_dir
):
    """Profile memory usage during execution."""

    print(f"\n{'='*60}")
    print(f"MEMORY PROFILING: {method_name}")
    print(f"{'='*60}")

    # Start memory tracking
    tracemalloc.start()

    # GPU memory tracking
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
        gpu_before = torch.cuda.memory_allocated(DEVICE)

    # Record initial state
    print("Initial GPU memory usage:")
    gpu_memory_usage()

    # Execute method
    print("Executing method...")
    start_time = time.perf_counter()

    layer1, layer2, extra_output = method(
        c1_params, c2_params, data_iterator=data_iterator, device=DEVICE
    )

    end_time = time.perf_counter()

    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Create network and get final score
    net = nn.Sequential(layer1, nn.ReLU(), layer2).to(DEVICE)
    method_score = score(net, data_iterator, DEVICE)

    # GPU memory stats
    gpu_stats = {}
    if DEVICE.type == "cuda":
        gpu_after = torch.cuda.memory_allocated(DEVICE)
        gpu_peak = torch.cuda.max_memory_allocated(DEVICE)

        gpu_stats = {
            "gpu_memory_used_mb": (gpu_after - gpu_before) / 1024 / 1024,
            "gpu_memory_peak_mb": gpu_peak / 1024 / 1024,
            "gpu_memory_allocated_after_mb": gpu_after / 1024 / 1024,
        }

    # Save memory report
    output_dir.mkdir(exist_ok=True)
    memory_file = output_dir / f"memory_{method_name.replace(' ', '_')}.txt"

    with open(memory_file, "w") as f:
        f.write(f"MEMORY USAGE REPORT FOR: {method_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Execution time: {end_time - start_time:.3f} seconds\n")
        f.write(f"Final score: {method_score:.3e}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(
            f"Configuration: {c1_params['in_channels']} -> {c1_params['out_channels']} -> {c2_params['out_channels']}\n\n"
        )

        f.write("CPU MEMORY USAGE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Current: {current / 1024 / 1024:.2f} MB\n")
        f.write(f"Peak: {peak / 1024 / 1024:.2f} MB\n\n")

        if gpu_stats:
            f.write("GPU MEMORY USAGE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Used by method: {gpu_stats['gpu_memory_used_mb']:.2f} MB\n")
            f.write(f"Peak during execution: {gpu_stats['gpu_memory_peak_mb']:.2f} MB\n")
            f.write(
                f"Total allocated after: {gpu_stats['gpu_memory_allocated_after_mb']:.2f} MB\n"
            )

    print(f"Memory report saved to: {memory_file}")
    print(f"\nMemory Summary:")
    print(
        f"  CPU Memory - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB"
    )
    if gpu_stats:
        print(
            f"  GPU Memory - Used: {gpu_stats['gpu_memory_used_mb']:.2f} MB, Peak: {gpu_stats['gpu_memory_peak_mb']:.2f} MB"
        )

    print("Final GPU memory usage:")
    gpu_memory_usage()

    return method_score, end_time - start_time


def profile_line_by_line(
    method, method_name, c1_params, c2_params, data_iterator, output_dir
):
    """Attempt line-by-line profiling if line_profiler is available."""

    try:
        from line_profiler import LineProfiler

        print(f"\n{'='*60}")
        print(f"LINE-BY-LINE PROFILING: {method_name}")
        print(f"{'='*60}")

        # Create line profiler
        profiler = LineProfiler()
        profiler.add_function(method)

        # Enable profiling
        profiler.enable_by_count()

        # Execute method
        layer1, layer2, extra_output = method(
            c1_params, c2_params, data_iterator=data_iterator, device=DEVICE
        )

        # Disable profiling
        profiler.disable_by_count()

        # Get results
        output_dir.mkdir(exist_ok=True)
        line_profile_file = (
            output_dir / f"line_profile_{method_name.replace(' ', '_')}.txt"
        )

        with open(line_profile_file, "w") as f:
            profiler.print_stats(stream=f)

        print(f"Line-by-line profile saved to: {line_profile_file}")

        # Also print to console
        profiler.print_stats()

    except ImportError:
        print("line_profiler not available. Install with: pip install line_profiler")
        print("Skipping line-by-line profiling.")


def main():
    """Main profiling function."""

    print(f"Simple Profiling Script")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Configuration
    config_size = "200-10"  # Change this: "tiny", "small", "medium", "large", "huge"
    method_name = "NL AdamW(1e-1, 5)"  # Change this to the method you want to profile
    full_profiling = False
    activation_function = nn.ReLU()  # Change this if needed, e.g., nn.Sigmoid, nn.Tanh

    # Available methods
    methods = {
        # "random": create_random_layers,
        # "zeros": create_zero_layers,
        "restricted_fogro": restricted_fogro_optimized_conv_layers,
        "norm_approx": norm_approx_optimized_conv_layers,
        # "full_s": full_s_optimized_conv_layers,
        # "natural_gradient": natural_gradient_step_layers,
        "sgd_adam_5": lambda *args, **kwargs: sgd_optimized_layers(
            *args, **kwargs, lr=0.01, epochs=5, optimizer="Adam"
        ),
        "NL AdamW(1e-1, 5)": lambda *args, **kwargs: sgd_optimized_layers(
            *args,
            **kwargs,
            lr=0.1,
            epochs=5,
            optimizer="Adam",
            activation_function=activation_function,
        ),
    }

    if method_name not in methods:
        print(f"Error: Method '{method_name}' not found.")
        print(f"Available methods: {list(methods.keys())}")
        return

    method = methods[method_name]

    # Setup test case
    c1_params, c2_params, data_iterator = setup_test_case(
        config_size, activation_function=activation_function
    )

    # Create output directory
    output_dir = Path(f"profiling_results_{method_name}_{config_size}")
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}")

    # Run different types of profiling

    # 1. Detailed cProfile analysis
    score1, time1 = profile_with_cprofile(
        method, method_name, c1_params, c2_params, data_iterator, output_dir
    )

    if full_profiling:
        # 2. Memory usage profiling
        score2, time2 = profile_memory_usage(
            method, method_name, c1_params, c2_params, data_iterator, output_dir
        )

        # 3. Line-by-line profiling (if available)
        profile_line_by_line(
            method, method_name, c1_params, c2_params, data_iterator, output_dir
        )

    # Summary
    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print(f"{'='*60}")
    print(f"Method: {method_name}")
    print(f"Configuration: {config_size}")
    if full_profiling:
        print(f"Scores: {score1:.3e} / {score2:.3e}")
        print(f"Times: {time1:.3f}s / {time2:.3f}s")
    print(f"Output saved to: {output_dir.absolute()}")

    print(f"\nTo analyze the binary profile interactively:")
    print(
        f"  python -m pstats {output_dir}/cprofile_{method_name.replace(' ', '_')}.prof"
    )

    print(f"\nProfile files created:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
