import json
import numpy as np
import pickle as pkl
import os
import sys
import time
import traceback

import torch
from torch.utils.data import RandomSampler

from nas import NAS
from data_processor import DataProcessor
from trainer import Trainer

# === CODABENCH I/O ===

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
SUBMISSION_DIR = sys.argv[3]

sys.path.insert(0, SUBMISSION_DIR)

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

# === MAIN ===
def main():
    print("=" * 78)
    print("="*13 + "    Your NAS Unseen-Data 2026 Submission is running     " + "="*13)
    print("="*78)

    for dataset in os.listdir(INPUT_DIR):
        metadata = load_dataset_metadata(os.path.join(INPUT_DIR, dataset))
        try:
            run_submission(dataset)
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

def run_submission(dataset: str):
    (train_x, train_y), (valid_x, valid_y), (test_x), metadata = load_datasets(dataset)
    this_dataset_start_time = time.perf_counter()

    print("="*10 + " Dataset {:^10} ".format(metadata['codename']) + "="*45)
    print("  Metadata:")
    [print("   - {:<20}: {}".format(k, v)) for k, v in metadata.items()]

    print("\n=== Processing Data ===")
    data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata)
    train_loader, valid_loader, test_loader = data_processor.process()

    assert_string = "Test Dataloader is {}, this will break evaluation. Please fix this in your DataProcessor init."
    assert not isinstance(test_loader.sampler, RandomSampler), assert_string.format("shuffling")
    assert not test_loader.drop_last, assert_string.format("dropping last batch")

    print("\n=== Performing NAS ===")
    model = NAS(train_loader, valid_loader, metadata).search()
    model_params = int(general_num_params(model))

    print("\n=== Training ===")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(model, device, train_loader, valid_loader, metadata)
    trained_model = trainer.train()

    print("\n=== Predicting ===")
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