import copy
import os
import sys
from functools import partial
from time import time

# Make the vendored gromo (python_packages/gromo/) importable regardless of cwd or whatever else
# named 'gromo' might already be on sys.path -- see README_DEMETER.md "Local gromo patches".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages"))

import torch
from torch.utils.data import DataLoader

from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.utils.utils import global_device, set_device

from helpers import (
    evaluate_model,
    topk_accuracy,
    show_time,
    resplit_optimization_development,
    bounded_loader,
    desired_update_norm,
    gather_full_model_statistics,
    growth_step_for_dag,
    train_with_early_stopping,
    clear_transient_tensors,
)
from model import CellArchitecture
from trainer import Trainer


class NAS:
    """
    ====================================================================================================================
    "Demeter" (Stella_s_neurips_2026.pdf), ported from experimental_grow-main's
    experiments/pipeline/{pipeline.py,models/cell_arch.py} into this challenge's contract.
    Algorithm 2 from the paper, reproduced faithfully: cycle through the 4 growable DAG cells,
    training each with early stopping and growing it (restricted strategy, Algorithm 1) until its
    own desired-update norm has been below threshold `u` for 2 consecutive rounds, then freeze it;
    once all 4 are frozen, fine-tune the whole network for `final_epochs` more epochs. See
    README_DEMETER.md for the full parameter-by-parameter mapping (Appendix A/B/C of the paper).

    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset
    """

    def __init__(self, train_loader, valid_loader, metadata, clock=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.set_metadata()

    def set_metadata(self):
        self.metadata.update({
            "no_cuda": False,

            # Macro-architecture -- Appendix A: seed = 1 conv (3x3, padded) per DAG, 64 channels,
            # SeLU + growable LayerNorm, first linear layer hidden size fixed at 128.
            "init_hidden_channels": 64,
            "hidden_features": 128,
            "use_layer_norm": True,
            "use_batch_norm": False,
            "use_group_norm": False,

            # Training-phase hyperparameters -- Appendix B
            "train_epochs": 100,
            "train_lr": 1e-3,
            "train_weight_decay": 8e-3,
            "train_eps": 1e-4,
            "train_eta_min": 1e-5,
            "train_grad_clip": 1.0,
            "es_abs_delta": 1.22e-3,
            "es_patience": 3,
            "train_batch_size": 256,

            # Growth-phase hyperparameters -- Appendix B
            "neuron_selection_threshold": 2.8e-4,   # s
            "bottleneck_samples": 2048,
            "bottleneck_batch_size": 64,
            "dag_freeze_threshold": 1e-5,            # u
            "dag_freeze_strikes": 2,
            "candidate_neurons": 100,      # new neurons tried per growth action
            "candidate_neuron_epochs": 100,
            "candidate_neuron_lrate": 1e-3,
            "candidate_neuron_batch_size": 256,

            # Data split -- Appendix C (the fixed 10% validation split is the challenge's own
            # valid_loader; only the growth-pool's internal opt/dev resplit is our concern here)
            "optimization_fraction": 0.60,

            # Final fine-tune -- Algorithm 2, line 18
            "final_epochs": 100,

            "max_rounds": 200,  # safety cap on the outer while-loop (paper's growth.steps: 60)

            # Section 4.3 of the paper reports "the output architecture that maximizes the
            # validation accuracy" rather than whatever the run ends on -- that's what
            # use_checkpointing implements (deepcopy of the whole model kept on every improvement,
            # see search()). Off by default: Algorithm 2's own pseudocode never reverts to an
            # earlier snapshot either, and this matches submission_gromo/submission_gromo_resnet's
            # behavior. Turn it on to reproduce the paper's own reporting convention.
            "use_checkpointing": False,
        })

        # Opt-in fast/approximate profile (DEMETER_FAST_RUN=1), off by default -- this submission's
        # real hyperparameters above are untouched unless explicitly requested. Purpose: a margaret
        # sanity-check run across all 5 paper benchmark datasets in ~4h total instead of the paper's
        # own ~50.6 GPU-hours (Table 4), to check the growth trajectory / validation-accuracy curve
        # looks sane -- not to reproduce paper-quality final numbers. Only the three epoch-count
        # knobs that dominate wall-clock are cut (~10x, roughly matching the ~12x total speedup
        # needed), plus max_rounds tightened to the paper's own actual convergence point (60,
        # already noted above) as a safety net. Every knob governing WHICH growth decisions get
        # made (candidate_neurons, bottleneck_samples, thresholds, batch sizes) is left untouched,
        # so the growth trajectory itself should still resemble real Demeter, just compressed.
        if os.environ.get("DEMETER_FAST_RUN"):
            self.metadata.update({
                "train_epochs": 10,
                "candidate_neuron_epochs": 10,
                "final_epochs": 10,
                "max_rounds": 60,
            })

    def search(self):
        if self.metadata.get("no_cuda"):
            set_device(torch.device("cpu"))
        self.device = global_device()

        # metadata['input_shape'] is (N, C, H, W) -- verified against the actual competition data
        # (e.g. Chesseract's real train_x.npy header is (49998, 12, 8, 8): 12 is unambiguously the
        # channel count, not a spatial dim). See data_processor.py's TorchDataset for the matching
        # fix on the raw-array side.
        channels, height, width = self.metadata["input_shape"][1:]
        out_features = self.metadata["num_classes"]

        loss_fn = torch.nn.CrossEntropyLoss()
        top_1_accuracy = partial(topk_accuracy, k=1)

        model = CellArchitecture(
            input_shape=(height, width),
            in_channels=channels,
            init_hidden_channels=self.metadata["init_hidden_channels"],
            hidden_features=self.metadata["hidden_features"],
            out_features=out_features,
            neurons=self.metadata["candidate_neurons"],
            neuron_epochs=self.metadata["candidate_neuron_epochs"],
            neuron_lrate=self.metadata["candidate_neuron_lrate"],
            neuron_batch_size=self.metadata["candidate_neuron_batch_size"],
            loss_fn=loss_fn,
            use_layer_norm=self.metadata["use_layer_norm"],
            use_batch_norm=self.metadata["use_batch_norm"],
            use_group_norm=self.metadata["use_group_norm"],
            device=self.device,
        )
        print("Starting model:")
        print(f"  params: {model.number_of_parameters()}")

        # Appendix C: a fixed validation split is used for early stopping and best-checkpoint
        # selection; the rest ("growth pool") gets reshuffled into optimization/development at
        # every growth step, and is also what training phases train on (both sets combined). The
        # challenge already hands us a held-out valid_loader, so we use that as the paper's fixed
        # validation split instead of re-deriving our own 10% out of train_loader (which would
        # waste the data the DataProcessor already set aside).
        growth_pool = self.train_loader.dataset
        val_loader = self.valid_loader

        dag_cells: dict[str, GrowingGraphNetwork] = {
            "dag1": model.dag1, "dag2": model.dag2, "dag3": model.dag3, "dag4": model.dag4,
        }
        strikes = {name: 0 for name in dag_cells}

        # Demeter's architecture changes shape as it grows, so a state_dict snapshot from an
        # earlier (smaller) round can't be load_state_dict'd back onto the later (larger) live
        # model -- caught by a smoke test (shape mismatches on every grown tensor). Snapshotting
        # the whole model object instead sidesteps this: each snapshot carries its own matching
        # architecture, no restoration needed.
        #
        # (A disk-spooled torch.save() variant was tried to avoid holding a second live copy in
        # RAM, but it's a dead end, not a bug to fix: some conv edges are monkey-patched with a
        # locally-defined closure as their forward function (conv2d_growing_module.py:977), and
        # Python's pickle -- what torch.save uses -- cannot serialize closures at all. deepcopy
        # works here specifically because it never serializes anything, just duplicates live
        # objects in memory. Revisiting the memory-vs-simplicity tradeoff is left for later.)
        use_checkpointing = self.metadata["use_checkpointing"]
        best_model = model
        best_val_acc = -1.0

        def maybe_checkpoint():
            # Always evaluate (used for the per-round log line below) even when checkpointing
            # itself is off; only the deepcopy-and-remember part is conditional.
            nonlocal best_model, best_val_acc
            _, val_acc = evaluate_model(model, val_loader, loss_fn, top_1_accuracy, self.device)
            if not use_checkpointing:
                return val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                clear_transient_tensors(model)
                best_model = copy.deepcopy(model)
            return val_acc

        start_time = time()
        round_idx = 0
        while any(s < self.metadata["dag_freeze_strikes"] for s in strikes.values()) and round_idx < self.metadata["max_rounds"]:
            round_idx += 1
            for name, dag_container in dag_cells.items():
                if strikes[name] >= self.metadata["dag_freeze_strikes"]:
                    continue

                round_start = time()
                train_loader = DataLoader(growth_pool, batch_size=self.metadata["train_batch_size"], shuffle=True)

                # [WHEN] train the whole network with early stopping
                train_with_early_stopping(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    loss_fn=loss_fn,
                    aux_loss_fn=top_1_accuracy,
                    device=self.device,
                    epochs=self.metadata["train_epochs"],
                    lr=self.metadata["train_lr"],
                    weight_decay=self.metadata["train_weight_decay"],
                    eps=self.metadata["train_eps"],
                    eta_min=self.metadata["train_eta_min"],
                    es_abs_delta=self.metadata["es_abs_delta"],
                    es_patience=self.metadata["es_patience"],
                    grad_clip=self.metadata["train_grad_clip"],
                )
                val_acc = maybe_checkpoint()

                # One whole-model forward-backward pass on a fresh optimization/development
                # resplit (Appendix C) feeds both the freeze check and the growth step below.
                optimization, development = resplit_optimization_development(
                    growth_pool, opt_fraction=self.metadata["optimization_fraction"]
                )
                bottleneck_loader = bounded_loader(
                    optimization,
                    batch_size=self.metadata["bottleneck_batch_size"],
                    max_samples=self.metadata["bottleneck_samples"],
                )
                development_loader = DataLoader(
                    development, batch_size=self.metadata["train_batch_size"], shuffle=True
                )

                pre_activities_grad, input_B = gather_full_model_statistics(
                    model, dag_cells, bottleneck_loader, loss_fn, self.device,
                    extra_node_modules=[model.linear_merge, model.mlp_merge],
                )
                end_name = dag_container.dag.get_node_module(dag_container.dag.end)._name
                du_norm = (
                    desired_update_norm(pre_activities_grad[end_name])
                    if end_name in pre_activities_grad else 0.0
                )
                if du_norm < self.metadata["dag_freeze_threshold"]:
                    strikes[name] += 1
                    print(f"round {round_idx} [{name}]: desired-update norm {du_norm:.3e} < u -- strike {strikes[name]}")
                else:
                    strikes[name] = 0

                print(
                    "round {:>3} [{}] | Valid Acc: {:>6.2f}% | strikes={} | T/round: {:<7} |".format(
                        round_idx, name, val_acc * 100, strikes[name], show_time(time() - round_start)
                    )
                )

                grew = False
                if strikes[name] < self.metadata["dag_freeze_strikes"]:
                    grew = growth_step_for_dag(
                        model=model,
                        dag_container=dag_container,
                        pre_activities_grad=pre_activities_grad,
                        input_B=input_B,
                        development_loader=development_loader,
                        val_loader=val_loader,
                        neuron_selection_threshold=self.metadata["neuron_selection_threshold"],
                        device=self.device,
                    )
                # Reset BEFORE checkpointing: compute_optimal_delta() above left transient
                # optimal_delta_layer submodules attached; snapshotting while they're still there
                # would carry that transient state into the checkpoint for no reason.
                model.reset_computation()
                if grew:
                    maybe_checkpoint()

        print(f"All DAGs frozen after {round_idx} rounds ({show_time(time() - start_time)})")

        # Final fine-tuning (Algorithm 2, line 18). Algorithm 2's pseudocode doesn't revert to the
        # best-so-far checkpoint before this step -- it just keeps training whatever network the
        # growth loop ended with -- so the live (fully grown) `model` continues unmodified here;
        # best-checkpoint tracking (Section 4.3: "the output architecture of Demeter is that which
        # maximizes the validation accuracy") covers this phase too via the maybe_checkpoint() call
        # below, alongside every checkpoint taken during the growth loop above.
        train_loader = DataLoader(growth_pool, batch_size=self.metadata["train_batch_size"], shuffle=True)
        self.metadata["training_epochs"] = self.metadata["final_epochs"]
        trainer = Trainer(model, device=self.device, train_dataloader=train_loader,
                           valid_dataloader=val_loader, metadata=self.metadata, clock=self.clock)
        model = trainer.train()
        final_val_acc = maybe_checkpoint()

        if use_checkpointing:
            print(f"Best validation accuracy achieved: {best_val_acc * 100:.2f}%")
        else:
            print(f"Final validation accuracy (checkpointing off, returning the final model as-is): "
                  f"{final_val_acc * 100:.2f}%")
        print(f"Final params: {best_model.number_of_parameters()}")
        print(f"Total duration: {show_time(time() - start_time)}")
        return best_model
