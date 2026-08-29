import math
import os
import sys
from time import time
from typing import Callable
from warnings import warn

# Make the vendored gromo (python_packages/gromo/) importable regardless of cwd or whatever else
# named 'gromo' might already be on sys.path -- see README_DEMETER.md "Local gromo patches".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import v2 as Tv2

from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.utils.utils import global_device

# ====================================================================================================================
# This is "Demeter" from Stella_s_neurips_2026.pdf ("Should I train or should I grow? Single-Run
# Architecture Search by Network Growth"), ported to the nas-challenge-submission-main contract.
#
# Macro-architecture (Figure 4 / model.py's CellArchitecture, vendored ~verbatim from
# experimental_grow-main/experiments/pipeline/models/cell_arch.py): 4 independent growable DAG
# cells (one per resolution), connected by avg-pooling, then GAP -> 2 FIXED-size linear layers
# (Appendix A: "The hidden size of the first linear layer is fixed at 128 neurons" -- confirmed by
# pipeline.py's cycle_layers(), which only ever visits GrowingGraphNetwork instances, never the
# linear head).
#
# Growth algorithm = Algorithm 1 ("Growth Step", restricted strategy) + Algorithm 2 ("Demeter") from
# the paper, reproduced here using gromo's native GrowingDAG/GrowingGraphNetwork API rather than
# experimental_grow's own wandb/ablation-oriented pipeline.py. See README_DEMETER.md for the full
# mapping and the one place where the paper's own pseudocode and its reference implementation
# disagree (resolved in favor of the reference implementation, which is what actually produced the
# published results).
# ====================================================================================================================


# ====================================================================================================================
# Generic helpers (metrics, transforms) -- same as submission_gromo_resnet
# ====================================================================================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val == val and val not in (np.inf, -np.inf):  # skip nan/inf
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def topk_accuracy(y_pred, y, k=1):
    result = y_pred.topk(k, dim=1).indices == y.unsqueeze(1)
    return result.sum() / y.size(0)


def div_remainder(n, interval):
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < 60 * 60:
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


def get_transforms(dataset_name: str, sample_data=None) -> list:
    """
    Base (non-augmenting) transform pipeline. Appendix C: Gutenberg and Chesseract use no
    augmentation at all in the paper; the others use dataset-specific augmentations we don't
    replicate here (RandomAffine/HorizontalFlip/etc. -- see README_DEMETER.md) to keep this
    submission dataset-agnostic, since the actual competition datasets are hidden and may not be
    exactly MultNIST/CifarTile/Gutenberg/Geoclassing/Chesseract.
    """
    known = {
        "mnist": [Tv2.ToDtype(torch.float32, scale=True), Tv2.Normalize(mean=(0.1307,), std=(0.3081,))],
        "cifar10": [Tv2.ToDtype(torch.float32, scale=True),
                    Tv2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))],
        "cifar100": [Tv2.ToDtype(torch.float32, scale=True),
                     Tv2.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))],
    }
    if dataset_name in known:
        return known[dataset_name]
    if sample_data is not None:
        # sample_data is already (N, C, H, W) -- see data_processor.py's TorchDataset -- so no
        # permute is needed before computing per-channel stats over dim 1.
        data = torch.as_tensor(sample_data, dtype=torch.float32)
        if data.max() > 1.0:
            data = data / 255.0
        dims = tuple(d for d in range(data.dim()) if d != 1)
        mean = data.mean(dim=dims).tolist()
        std = [s if s > 1e-6 else 1.0 for s in data.std(dim=dims).tolist()]
        return [Tv2.ToDtype(torch.float32, scale=True), Tv2.Normalize(mean=mean, std=std)]
    warn(f"Unknown dataset '{dataset_name}' and no sample_data provided; using scale-only transform")
    return [Tv2.ToDtype(torch.float32, scale=True)]


def evaluate_model(model, dataloader, loss_function, aux_loss_function=None, device=None):
    """/!/ loss_function must be mean-reduced."""
    if device is None:
        device = global_device()
    loss_meter, aux_meter = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss_meter.update(loss.item(), x.size(0))
            if aux_loss_function is not None:
                aux_meter.update(aux_loss_function(y_pred, y).item(), x.size(0))
    return loss_meter.avg, aux_meter.avg


# ====================================================================================================================
# Data splitting -- Appendix C: 10% held out as a FIXED validation split (used for early stopping
# and for picking the best checkpoint); the remaining 90% is reshuffled, AT EACH GROWTH STEP, into
# 60% optimization (gradient descent for the new weights) / 40% development (line search for the
# new weights' scaling factor).
# ====================================================================================================================

def split_off_validation(dataset: Dataset, val_fraction: float = 0.10, seed: int = 0):
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_growth_pool = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    growth_pool, validation = random_split(dataset, [n_growth_pool, n_val], generator=generator)
    return growth_pool, validation


def resplit_optimization_development(growth_pool: Dataset, opt_fraction: float = 0.60, seed: int | None = None):
    n_opt = max(1, int(round(len(growth_pool) * opt_fraction)))
    n_dev = len(growth_pool) - n_opt
    if n_dev < 1:
        n_opt -= 1
        n_dev = 1
    generator = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
    optimization, development = random_split(growth_pool, [n_opt, n_dev], generator=generator)
    return optimization, development


def bounded_loader(dataset: Dataset, batch_size: int, max_samples: int | None = None, shuffle: bool = True):
    if max_samples is not None and len(dataset) > max_samples:
        indices = torch.randperm(len(dataset))[:max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def desired_update_norm(grad: torch.Tensor) -> float:
    """||v_nabla|| normalized by sqrt(numel) -- what Algorithm 2 thresholds against `u`."""
    return (torch.linalg.norm(grad) / math.sqrt(grad.numel())).item()


def clear_transient_tensors(model) -> None:
    """
    Clear cached computation-scratch tensors (_pre_activity/_input/activity/input) before
    deep-copying a checkpoint. These are pure scratch (re-populated on the next instrumented
    forward pass, not part of the model's real state -- the real state is the nn.Parameters), but
    GrowingModule.forward calls .retain_grad() on _pre_activity, deliberately keeping it live and
    graph-connected; reset_computation() only flips the store_input/store_activity flags, it
    doesn't clear the tensors already cached from an earlier pass. A stale one is a non-leaf
    tensor that still requires grad, which torch.Tensor.__deepcopy__ refuses to copy outright
    (caught by a smoke test).
    """
    for module in model.modules():
        if hasattr(module, "_pre_activity"):
            module._pre_activity = None
        if hasattr(module, "_input"):
            module._input = None
        if "activity" in module.__dict__:
            module.__dict__["activity"] = None
        if "input" in module.__dict__:
            module.__dict__["input"] = None


# ====================================================================================================================
# Algorithm 1 ("Growth Step", restricted strategy)
#
# IMPORTANT: dag1..dag4 are sub-components of the composite CellArchitecture, not standalone
# models. gromo's own GrowingDAG.calculate_bottleneck / GrowingGraphNetwork.find_amplitude_factor
# assume the DAG they're called on IS the whole model (they do `self(x)` / `self.extended_forward(x)`
# and compare the raw output against the true labels) -- correct for e.g.
# wrappers/exp_graph_growing_net.py's ExpGrowingGraphNetwork (a standalone DAG), wrong for dag2/
# dag3/dag4 here, whose raw output is an intermediate feature map, not classification logits.
# experiments/pipeline/pipeline.py handles this by gathering statistics from a whole-*model*
# forward-backward pass and evaluating candidates via `model.extended_forward(x, mask=...)` /
# `Expansion.evaluate(model=model, ...)` instead of the DAG-local equivalents. The functions below
# reproduce that approach (this was the first thing the smoke test caught: dag_container(x) alone
# returns a spatial feature map, not logits, so a plain nn.CrossEntropyLoss against it fails
# immediately for any DAG that isn't the last one).
# ====================================================================================================================

def gather_full_model_statistics(
    model,
    dag_cells: dict[str, GrowingGraphNetwork],
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device,
    extra_node_modules=(),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    One whole-model forward-backward pass: accumulates the raw pre-activity gradient (v_nabla) and
    post-activity (input for the bottleneck's v* term) at every node of every DAG cell, then lets
    gromo compute each edge's optimal delta from the statistics just accumulated. Mirrors
    pipeline.py's `update_computation` + `compute_optimal_delta` steps, run once per round and
    shared by both the "where to grow" bottleneck computation and the "should this DAG freeze"
    check, instead of doing a separate pass for each.

    `extra_node_modules`: dag4's own growth candidates can reach forward past its own boundary
    into the (fixed-size, never itself grown) classification head's merge modules
    (model.linear_merge / model.mlp_merge in CellArchitecture) via an InterMergeExpansion, exactly
    the same way an inter-DAG boundary action reaches into the next cell's root -- caught by a
    smoke test (KeyError: 'mlp_merge' from GrowingGraphNetwork.expand_node, which needs that
    node's bottleneck too). Pass those modules here so their statistics get gathered in the same
    pass instead of only scanning the 4 DAG cells' own nodes.
    """
    model.init_computation()
    all_node_modules = set(extra_node_modules)
    for dag_container in dag_cells.values():
        all_node_modules.update(dag_container.dag.get_all_node_modules())

    pre_activities_grad = {m._name: [] for m in all_node_modules}
    input_B = {m._name: [] for m in all_node_modules}

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        model.update_computation()
        for node_module in all_node_modules:
            if node_module.activity is not None:
                input_B[node_module._name].append(node_module.activity.clone().detach().cpu())
            if node_module.pre_activity is not None and node_module.pre_activity.grad is not None:
                pre_activities_grad[node_module._name].append(
                    node_module.pre_activity.grad.clone().detach().cpu()
                )

    model.compute_optimal_delta()

    pre_activities_grad = {k: torch.cat(v) for k, v in pre_activities_grad.items() if v}
    input_B = {k: torch.cat(v) for k, v in input_B.items() if v}
    return pre_activities_grad, input_B


def compute_bottleneck_for_dag(
    dag_container: GrowingGraphNetwork,
    actions,
    pre_activities_grad: dict[str, torch.Tensor],
    input_B: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Per-node expressivity bottleneck v_perp = v_nabla - v*, normalized to unit mean-per-sample
    squared norm (matching pipeline.py's `calculate_bottleneck` step -- this normalization is
    NOT present in gromo's own GrowingDAG.calculate_bottleneck).

    `actions` matters here: with `expand_end=True`, `define_next_actions` can return
    InterMergeExpansion actions whose target node lives in the *next* DAG cell, not this one --
    the bottleneck needs to be computed for that node too (gather_full_model_statistics already
    recorded its gradient/activity, since it scans every cell's nodes).
    """
    from gromo.containers.growing_dag import InterMergeExpansion

    dag = dag_container.dag
    nodes_of_interest = set(dag.get_all_node_modules())
    for action in actions:
        if isinstance(action, InterMergeExpansion):
            nodes_of_interest.update(action.next_nodes)

    bottleneck: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for node_module in nodes_of_interest:
            name = node_module._name
            if name == dag.root or name not in pre_activities_grad:
                continue
            v_proj = pre_activities_grad[name].clone()
            for edge_module in node_module.previous_modules:
                prev_name = edge_module.previous_module._name
                if edge_module.optimal_delta_layer is None or prev_name not in input_B:
                    continue
                v_proj -= edge_module.optimal_delta_layer(
                    input_B[prev_name].to(edge_module.device)
                ).detach().cpu()
            v_proj_norm = torch.einsum("b...,b...->b", v_proj, v_proj).mean()
            bottleneck[name] = v_proj / torch.sqrt(v_proj_norm + 1e-8)
    return bottleneck


def line_search_amplitude(
    model,
    expansion,
    dev_loader: DataLoader,
    first_order_improvement: float,
    device,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_iter: int = 20,
    epsilon: float = 1e-3,
    extended_search: bool = False,
) -> float:
    """
    Backtracking (Armijo) line search for one candidate expansion's scaling factor, evaluated on
    the WHOLE model via model.set_scaling_factor + model.extended_forward(x, mask=...) -- same
    algorithm as submission_gromo_resnet's line_search, generalized from a single growing layer to
    a masked composite-model extended_forward.
    """
    mask = expansion.create_mask()
    eval_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def eval_gamma(sqrt_gamma: float) -> float:
        # Not wrapped in torch.no_grad(): the model is still in "computation mode" here (hooks
        # from init_computation() are still attached, since apply_change()/reset_computation()
        # haven't run yet for this growth step) and MergeGrowingModule.forward unconditionally
        # calls .retain_grad() on its activity, which requires grad-tracking to be live -- caught
        # by a smoke test ("can't retain_grad on Tensor that has requires_grad=False").
        model.set_scaling_factor(sqrt_gamma)
        meter = AverageMeter()
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model.extended_forward(x, mask=mask)
            meter.update(eval_loss_fn(pred, y).item() / x.size(0), x.size(0))
        return meter.avg

    beta_sqrt = math.sqrt(beta)
    epsilon_sqrt = math.sqrt(epsilon)
    first_order_improvement = max(float(first_order_improvement), 1e-7)
    initial_loss = eval_gamma(0.0)

    def under_bound(sqrt_gamma: float, loss: float) -> bool:
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    t = math.sqrt(2 * initial_loss / first_order_improvement) if initial_loss > 0 else 1.0
    l1 = eval_gamma(t)
    i = 0
    if under_bound(t, l1):
        if extended_search:
            go = True
            while go:
                l0 = l1
                t /= beta_sqrt
                l1 = eval_gamma(t)
                go = l1 < l0 and i < max_iter
                i += 1
            t *= beta_sqrt
    else:
        go = True
        while go:
            t *= beta_sqrt
            l1 = eval_gamma(t)
            go = (not under_bound(t, l1)) and i < max_iter and t > epsilon_sqrt
            i += 1
    model.set_scaling_factor(t)
    return t


def growth_step_for_dag(
    model,
    dag_container: GrowingGraphNetwork,
    pre_activities_grad: dict[str, torch.Tensor],
    input_B: dict[str, torch.Tensor],
    development_loader: DataLoader,
    val_loader: DataLoader,
    neuron_selection_threshold: float,
    device,
    verbose: bool = False,
) -> bool:
    """
    Algorithm 1 (restricted strategy) for one DAG cell, using statistics already gathered by
    `gather_full_model_statistics`. Returns True if a growth action was applied.
    """
    dag = dag_container.dag
    actions = dag.define_next_actions(expand_end=True)
    if not actions:
        return False

    # [WHERE] bottleneck must be computed after `actions` is known: expand_end=True can produce
    # InterMergeExpansion actions reaching into the next DAG cell.
    bottleneck = compute_bottleneck_for_dag(dag_container, actions, pre_activities_grad, input_B)
    if not bottleneck:
        return False
    bott_norms = {name: torch.linalg.norm(t) for name, t in bottleneck.items()}
    j_star = max(bott_norms, key=lambda k: bott_norms[k])
    if verbose:
        print(f"  [{dag_container._name}] restricted to node {j_star} (||bottleneck||={bott_norms[j_star]:.4e})")
    restricted_actions = dag_container.restrict_action_space(actions, chosen_outputs=[j_star])
    if not restricted_actions:
        return False

    # Matching pipeline.py's exact step order (compute_optimal_delta -> calculate_bottleneck ->
    # reset_computation -> ... -> execute_expansion): reset_computation() only clears the
    # store_input/store_activity tracking flags (confirmed by reading MergeGrowingModule's
    # implementation), not optimal_delta_layer or any extension -- so it's safe here, now that the
    # bottleneck computation above is done with them. It's necessary here: leaving store_input
    # active makes every later forward pass call retain_grad() on its activity, which crashes
    # inside the (correctly gradient-tracking, not no_grad) amplitude-factor search and
    # Expansion.evaluate() below (both caught by smoke tests).
    model.reset_computation()

    # [HOW] optimize + prune each remaining candidate's new weights (no amplitude/eval here --
    # those need the whole model, done below)
    dag_container.execute_expansions(
        actions=restricted_actions,
        bottleneck=bottleneck,
        input_B=input_B,
        amplitude_factor=False,
        evaluate=False,
        neuron_selection_threshold=neuron_selection_threshold,
        verbose=verbose,
    )
    restricted_actions = [a for a in restricted_actions if not a.metrics.get("skip", False)]
    if not restricted_actions:
        return False

    # [HOW MUCH: amplitude] + score each surviving candidate on validation
    fo_improvement = bott_norms[j_star].item() ** 2
    for expansion in restricted_actions:
        factor = line_search_amplitude(model, expansion, development_loader, fo_improvement, device)
        expansion.metrics["scaling_factor"] = factor
        expansion.evaluate(
            model=model, train_dataloader=None, dev_dataloader=None,
            val_dataloader=val_loader, loss_fn=nn.CrossEntropyLoss(reduction="mean"),
        )

    dag_container.choose_growth_best_action(restricted_actions, verbose=verbose)

    # Propagate the chosen action to every other growing layer and apply on ALL of them -- this
    # (not just dag_container.apply_change()) is what correctly applies a growth action that spans
    # a DAG boundary (an InterMergeExpansion, e.g. growing this DAG's own output width, which
    # reaches into the next cell's root node). This is pipeline.py's actual
    # choose_best_growth_action + apply_change steps; calling apply_change only on dag_container
    # (what an earlier version of this function did) silently leaves the adjacent cell's input
    # size stale, causing a channel-count mismatch on the very next forward pass -- caught by a
    # smoke test, and exactly the cross-DAG-boundary growth the paper calls out (Section 3.5) as
    # having required "major engineering changes" from the original authors.
    from gromo.modules.growing_module import GrowingModule

    mask = dag_container.chosen_action.create_mask()
    for layer in model._growing_layers:
        if isinstance(layer, GrowingGraphNetwork) and layer is not dag_container:
            layer.chosen_action = dag_container.chosen_action
            layer.clean_graph_with_chosen_action(restricted_actions)
        elif isinstance(layer, GrowingModule):
            layer.delete_update(
                include_previous=False,
                delete_delta=False,
                delete_input=(
                    True if layer.previous_module is None
                    else layer.previous_module._name not in mask.get("nodes", [])
                ),
                delete_output=(
                    True if layer.next_module is None
                    else layer.next_module._name not in mask.get("nodes", [])
                ),
            )

    for layer in model._growing_layers:
        if isinstance(layer, GrowingGraphNetwork):
            layer.apply_change()
        elif isinstance(layer, GrowingModule):
            layer.apply_change(apply_previous=False)
            if layer.extended_output_layer is not None:
                layer._apply_output_changes(
                    scaling_factor=dag_container.chosen_action.metrics["scaling_factor"],
                    extension_size=dag_container.chosen_action.metrics.get("active_neurons", 0),
                )

    model.update_size()
    return True


# ====================================================================================================================
# Training phase with early stopping ("when to grow") -- Appendix B: AdamW, cosine schedule over a
# max of `epochs`, early stopping when the validation-loss improvement is below `es_abs_delta` for
# `es_patience` consecutive epochs.
# ====================================================================================================================

def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    aux_loss_fn: Callable,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    eps: float,
    eta_min: float,
    es_abs_delta: float,
    es_patience: int,
    grad_clip: float | None = 1.0,
    show: bool = False,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    best_val_loss = math.inf
    patience_counter = 0
    val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, aux_loss_fn, device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, aux_loss_fn, device)
        if show:
            print(f"    epoch {epoch + 1}/{epochs} | val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%")

        if best_val_loss - val_loss > es_abs_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= es_patience:
                break

    return val_loss, val_acc
