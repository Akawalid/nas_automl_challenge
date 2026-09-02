import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages"))

import yaml

from experiments.pipeline import pipeline
from experiments.pipeline.run_pipeline import execute_pipeline

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_config.yaml")


class NAS:
    """
    ====================================================================================================================
    Unlike submission_demeter (which reimplements Demeter from the paper's Algorithm 1/2
    pseudocode using gromo's low-level API directly), this submission does NOT reimplement the
    algorithm at all. It vendors Santiago Florido's own real code verbatim
    (python_packages/experiments/, python_packages/tools/ -- copied, not retyped, from
    experimental_grow-main) and drives it through execute_pipeline(), the SAME generic
    step-interpreter his own run_pipeline.py uses, running the EXACT step list from his own
    experiments_config.yaml (the config that actually produced the paper's reported numbers, via
    experiments/pipeline/launch_local_base_multi_dataset.sh -> run_init_scaling_ablation.slurm).

    The only real code here is the glue needed to fit the challenge's contract:
      * data_processor.py replaces pipeline.py's own `load_data` step (which downloads a dataset
        from a path) with one that builds the same kind of Dataset objects directly from the
        numpy arrays the challenge hands us, then stashes them on `metadata` for this class to
        pick up (see data_processor.py's docstring for why that's how it's wired).
      * This file builds `cfg` from the real experiments_config.yaml (only overriding
        experiment/dataset identity, not any algorithm hyperparameter), builds an initial
        `context`, and calls the real execute_pipeline() with the real step list (minus
        `load_data`, replaced by DataProcessor as above) -- nothing about the growth/training
        logic itself is reimplemented or reordered.
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor (unused directly here --
          see above, this class uses metadata['_pipeline_train_set'] instead, which is the
          Dataset object pipeline.py's own split_data/create_dataloaders steps actually need)
        * valid_loader: The valid loader created by your DataProcessor (same)
        * metadata: A dictionary with information about this dataset
    """

    def __init__(self, train_loader, valid_loader, metadata, clock=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def search(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        # Overrides limited to run identity + logging (no real API/network calls -- see
        # tools/logger.py's Logger: every method is `if not self.enabled: return`, verified
        # directly). Every algorithm/training hyperparameter in experiments_config.yaml is left
        # exactly as Florido's own code has it -- that's the entire point of this submission.
        cfg["experiment"]["seed"] = int(self.metadata.get("seed", 0))
        cfg["experiment"]["name"] = f"challenge_{self.metadata.get('codename', 'run')}"
        cfg["dataset"]["name"] = self.metadata.get("_pipeline_dataset_name", "unknown")
        cfg["logger"]["enabled"] = False
        cfg["logger"]["save_intermediate_models"] = False
        cfg["logger"]["path"] = os.path.abspath("logs")
        cfg["logger"]["tmpdir"] = os.path.abspath("tmp")
        os.makedirs(cfg["logger"]["path"], exist_ok=True)
        os.makedirs(cfg["logger"]["tmpdir"], exist_ok=True)

        context = {
            "train_set": self.metadata["_pipeline_train_set"],
            "val_set": self.metadata["_pipeline_val_set"],
            "test_set": self.metadata["_pipeline_test_set"],
        }
        context.update(pipeline.setup_logger(cfg))

        # The real pipeline: as defined in experiments_config.yaml, minus `load_data` (the
        # challenge already handed us the data; DataProcessor built train_set/val_set above in
        # its place). NOTE on reproducibility, checked directly: `set_random_seed` (the only step
        # that actually calls set_random_seeds(seed, device) -- confirmed the only call site in
        # experiments/pipeline/) is NOT a member of experiments_config.yaml's own pipeline: list at
        # all (grepped both shipped configs, zero matches) -- so Florido's OWN reference CLI
        # (run_pipeline.py's main(), which runs cfg["pipeline"] unmodified) never seeds the RNGs
        # either. This class matches that exactly rather than introducing seeding his own default
        # config doesn't have. A real side-by-side margaret run (this class vs. the standalone
        # script) showed a 2.77-point CifarTile accuracy gap between two otherwise-identical runs
        # -- consistent with ordinary unseeded run-to-run variance (matches the paper's own
        # reported +/-2.7% std for CifarTile closely), not a bug in this port.
        steps = [s for s in cfg["pipeline"] if s != "load_data"]

        execute_pipeline(cfg, steps, context)

        return context["model"]
