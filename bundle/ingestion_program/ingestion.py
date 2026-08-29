#tau-frugal@inria.fr: to send the submission of NAS
#demeter

import json
import math
import numpy as np
import pickle as pkl
import os
import sys
import time
import traceback

import torch
from torch.utils.data import RandomSampler

SUBMISSION_DIR = sys.argv[3]
sys.path.insert(0, SUBMISSION_DIR)

from nas import NAS
from data_processor import DataProcessor
from trainer import Trainer

# === CODABENCH I/O ===

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Total wall-clock budget (seconds) shared across every dataset in this run.
# This is the Phase 2 copy of ingestion_program: it must match Phase 2's
# `execution_time_limit` in competition.yaml (3600s = 1h smoke test). Phase 3
# uses its own copy (ingestion_program_phase3/ingestion.py, 86400s) since
# Codabench does not expose execution_time_limit to the running container --
# this value has to be kept in sync by hand between the two copies.
TIME_LIMIT_SECONDS = int(os.environ.get("TIME_LIMIT_SECONDS", 3600))

# === DATA LOADING HELPERS ===
def load_dataset_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'metadata'), "r") as f:
        metadata = json.load(f)
    return metadata

def load_datasets(dataset):
    data_path = os.path.join(INPUT_DIR, dataset)
    train_x = np.load(os.path.join(data_path, 'train_x.npy'))
    train_y = np.load(os.path.join(data_path, 'train_y.npy'))
    valid_x = np.load(os.path.join(data_path, 'valid_x.npy'))
    valid_y = np.load(os.path.join(data_path, 'valid_y.npy'))
    test_x = np.load(os.path.join(data_path, 'test_x.npy'))
    metadata = load_dataset_metadata(data_path)
    return (train_x, train_y), (valid_x, valid_y), (test_x), metadata

# === MODEL ANALYSIS ===
def general_num_params(model):
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

# === TIME TRACKING ===
def div_remainder(n, interval):
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder

def show_time(seconds):
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)

# keeps a running counter of the compute time left for the whole submission
class Clock:
    def __init__(self, time_limit_in_seconds):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + time_limit_in_seconds

    def check(self):
        return self.time_limit - time.perf_counter()

# === MAIN ===
def main():
    print("=" * 78)
    print("="*13 + "    Your NAS Unseen-Data 2026 Submission is running     " + "="*13)
    print("="*78)

    runclock = Clock(TIME_LIMIT_SECONDS)
    print(f"Total allotted compute time: {show_time(TIME_LIMIT_SECONDS)}")

    for dataset in os.listdir(INPUT_DIR):
        metadata = load_dataset_metadata(os.path.join(INPUT_DIR, dataset))

        if runclock.check() <= 0:
            print(f"No compute time remaining, skipping dataset: {metadata['codename']}")
            fail_dataset(metadata)
            continue

        try:
            run_submission(dataset, runclock)
        except Exception as ex:
            print(ex)
            print(traceback.format_exc())
            fail_dataset(metadata)

        print(f'finished dataset: {metadata["codename"]}')
    print('done')
    return

def fail_dataset(metadata):
    print(f'Dataset {metadata["codename"]} failed.')
    run_data = {'Failed': True, 'Runtime': -1, 'Params': None}
    with open(os.path.join(OUTPUT_DIR, "{}_stats.pkl".format(metadata['codename'])), "wb") as f:
        pkl.dump(run_data, f)

def run_submission(dataset: str, runclock: Clock):
    (train_x, train_y), (valid_x, valid_y), (test_x), metadata = load_datasets(dataset)
    this_dataset_start_time = time.perf_counter()
    metadata['time_remaining'] = runclock.check()

    print("="*10 + " Dataset {:^10} ".format(metadata['codename']) + "="*45)
    print("  Metadata:")
    [print("   - {:<20}: {}".format(k, v)) for k, v in metadata.items()]

    print("\n=== Processing Data ===")
    print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
    data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, runclock)
    train_loader, valid_loader, test_loader = data_processor.process()
    metadata['time_remaining'] = runclock.check()

    assert_string = "Test Dataloader is {}, this will break evaluation. Please fix this in your DataProcessor init."
    assert not isinstance(test_loader.sampler, RandomSampler), assert_string.format("shuffling")
    assert not test_loader.drop_last, assert_string.format("dropping last batch")

    print("\n=== Performing NAS ===")
    print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
    model = NAS(train_loader, valid_loader, metadata, runclock).search()
    model_params = int(general_num_params(model))
    metadata['time_remaining'] = runclock.check()

    print("\n=== Training ===")
    print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(model, device, train_loader, valid_loader, metadata, runclock)
    trained_model = trainer.train()
    metadata['time_remaining'] = runclock.check()

    print("\n=== Predicting ===")
    print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
    predictions = trainer.predict(test_loader)

    run_time = time.perf_counter() - this_dataset_start_time
    print(f'Run time for {metadata["codename"]}: {run_time}')

    run_data = {'Failed': False, 'Runtime': float(np.round(run_time, 2)), 'Params': model_params}
    with open(os.path.join(OUTPUT_DIR, "{}_stats.pkl".format(metadata['codename'])), "wb") as f:
        pkl.dump(run_data, f)
    np.save(os.path.join(OUTPUT_DIR, '{}.npy'.format(metadata['codename'])), predictions)
    print("Model Training and Prediction Complete")

if __name__ == '__main__':
    main()
