import json
import numpy as np
import os
import matplotlib.pyplot as plt



def load_pareto_front(runhistory: dict, objectives: list):
    data = runhistory["data"]
    configs = runhistory["configs"]

    # Extract all evaluations
    config_ids = []
    costs = []

    for entry in data:
        cfg_id = entry["config_id"]
        cost = entry["cost"]
        config_ids.append(cfg_id)
        costs.append(cost)

    costs = np.array(costs)  # shape (N, 3)

    pareto_mask = compute_pareto_mask(costs)

    return config_ids, costs, pareto_mask


def compute_pareto_mask(costs, obj:list=None):
    def is_dominated(a, b, obj:list=None):
        """Return True if a is dominated by b (both are vectors)."""
        if obj is None:
            return np.all(b <= a) and np.any(b < a)
        else:
            return np.all(b[obj] <= a[obj]) and np.any(b[obj] < a[obj])

    pareto_mask = np.zeros(len(costs), dtype=bool)

    for i, a in enumerate(costs):
        dominated = False
        for j, b in enumerate(costs):
            if i != j and is_dominated(a, b, obj):
                dominated = True
                break
        pareto_mask[i] = not dominated

    return pareto_mask


def main():
    # path = "smac3_output_keep/0f9f59704b0d221d913f05a158edb4bd/0"
    path = "smac3_output_keep/ee027c92c7e3ac102b61e63a43b3dcc7/0"

    with open(os.path.join(path, "runhistory.json"), "r") as f:
        rh = json.load(f)
    with open(os.path.join(path, "scenario.json"), "r") as f:
        sc = json.load(f)
    objectives = sc["objectives"]

    config_ids, costs, pareto_mask = load_pareto_front(rh, objectives)

    pareto_costs = costs[pareto_mask]
    pareto_config_ids = np.array(config_ids)[pareto_mask]


    print("Pareto optimal configurations:\n")
    for cfg_id, cost in zip(pareto_config_ids, pareto_costs):
        print("Config ID:", cfg_id)
        print("Hyperparameters:", rh["configs"][str(cfg_id)])
        print("Objectives:", cost)
        print()

    plot_pareto(costs, pareto_costs, objectives, 0, 4)

def plot_pareto(costs, pareto_costs, objectives, obj_x: int, obj_y: int):
    # obj_x = 0   # first objective
    # obj_y = 4   # second objective
    costs = np.array(costs)
    pareto_costs = np.array(pareto_costs)
    # Sort them a bit
    pareto_costs = pareto_costs[pareto_costs[:, obj_x].argsort()]

    plt.figure(figsize=(7,6))

    # Plot all evaluations
    plt.scatter(costs[:, obj_x], costs[:, obj_y], 
                color="gray", s=50, label="All evaluations")

    # Plot Pareto points
    plt.scatter(pareto_costs[:, obj_x], pareto_costs[:, obj_y],
                color="red", s=100, label="Pareto front")

    plt.plot(pareto_costs[:, obj_x], pareto_costs[:, obj_y],
            color="red", linewidth=2)

    plt.xlabel(f"Objective {objectives[obj_x]} (minimize)")
    plt.ylabel(f"Objective {objectives[obj_y]} (minimize)")
    plt.title("Pareto Front from SMAC RunHistory")
    plt.legend()
    # plt.xscale("log")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()