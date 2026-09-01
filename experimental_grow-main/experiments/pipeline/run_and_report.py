"""
Thin wrapper around run_pipeline.main() that prints the final results before exiting.

run_pipeline.py's own `if __name__ == "__main__": main(argv=sys.argv[1:])` discards main()'s
return value -- main() DOES return the full `context` dict (acc_val/acc_test/loss_val/loss_test
lists, one entry appended per growth round by pipeline.evaluate(), plus the trained `model` object
itself from pipeline.create_model()), but nothing ever prints it. Combined with --logger.enabled
false (needed to skip wandb auth/network on a compute node -- see tools/logger.py's Logger, every
method including log_pytorch_model is gated `if not self.enabled: return`), a run submitted via
the plain CLI entrypoint computes real numbers and then discards all of them on exit -- nothing is
printed, logged, or saved to disk anywhere.

This changes NOTHING about the pipeline itself -- same main(), same argv parsing, same config,
same steps. It only adds a print of what main() already had. Use exactly like run_pipeline.py:

    python experiments/pipeline/run_and_report.py --config ... --dataset.name cifartile ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pipeline.run_pipeline import main

if __name__ == "__main__":
    context = main(argv=sys.argv[1:])

    print("\n================ FINAL RESULTS ================")
    model = context.get("model")
    if model is not None:
        try:
            print(f"Final parameter count: {model.number_of_parameters()}")
        except Exception as error:
            print(f"Could not read parameter count: {error}")
    else:
        print("No 'model' key in context -- nothing was trained?")

    for key in ("acc_val", "acc_test", "loss_val", "loss_test"):
        values = context.get(key)
        if values:
            print(f"Final {key}: {values[-1]}  (all {len(values)} rounds: {values})")
        else:
            print(f"No '{key}' recorded in context.")
    print("=================================================")
