import json
import argparse
import sys

if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")
if '/home/tau/sdouka/codebase/experimental_grow' not in sys.path:
    sys.path.append('/home/tau/sdouka/codebase/experimental_grow')

from experiments.pipeline.run_pipeline import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--result", type=str)
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--node_name", type=str)
    args = parser.parse_args()

    # 1. Load config (JSON dumped by SMAC)
    config = json.loads(args.config)

    # 2. Train on GPU
    print(f"Starting smac configuration {config}")
    context = main(extra_config=config)
    metrics = [
        context["nb_params"],
        context["loss_train"][-1], context["loss_val"][-1],
        1-context["acc_train"][-1], 1-context["acc_val"][-1]
    ]

    # 3. Save result for SMAC
    with open(args.result, "w") as f:
        json.dump(metrics, f)
