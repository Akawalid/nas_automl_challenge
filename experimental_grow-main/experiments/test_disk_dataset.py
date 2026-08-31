import time

import numpy as np
import torch
from gromo.utils.disk_dataset import DiskDataset, MemMapDataset, SimpleMemMapDataset
from pipeline.run_pipeline import show_time
from torch.utils.data import DataLoader
from tqdm import tqdm


def convert_dict_to_npy(pt_path: str, prefix: str):
    data = torch.load(pt_path, map_location="cpu")
    for key, tensor in data.items():
        np.save(f"temp/{prefix}_{key}.npy", tensor.numpy())


if __name__ == "__main__":

    start = time.time()
    dataset = DiskDataset(
        input_filename="temp/input_B.pt",
        target_filename="temp/bottleneck.pt",
        input_keys=["start@dag1"],
        target_keys=["end@dag1"],
    )
    print("DiskDataset creation time:", show_time(time.time() - start))
    dataloader = DataLoader(dataset, batch_size=4)

    start = time.time()
    for x, y in tqdm(dataloader):
        pass
    print("DiskDataset execution time:", show_time(time.time() - start))
    print(f"{x.shape=} {y.shape=}")
    print()

    start = time.time()
    dataset = MemMapDataset(
        input_filename="temp/input_B.pt",
        target_filename="temp/bottleneck.pt",
        input_keys=["start@dag1"],
        target_keys=["end@dag1"],
    )
    print("MemMapDataset creation time:", show_time(time.time() - start))
    dataloader = DataLoader(dataset, batch_size=4)

    start = time.time()
    for x, y in tqdm(dataloader):
        pass
    print("MemMapDataset execution time:", show_time(time.time() - start))
    print(f"{x.shape=} {y.shape=}")
    print()

    start0 = time.time()
    convert_dict_to_npy("temp/input_B.pt", prefix="__input")
    convert_dict_to_npy("temp/bottleneck.pt", prefix="__bottleneck")
    print("Conversion time:", show_time(time.time() - start0))
    start = time.time()
    dataset = SimpleMemMapDataset(
        input_filenames=["temp/__input_start@dag1.npy", "temp/__input_1.npy"],
        target_filenames=["temp/__bottleneck_end@dag1.npy", "temp/__bottleneck_1.npy"],
    )
    print("SimpleMemMapDataset creation time:", show_time(time.time() - start))
    dataloader = DataLoader(dataset, batch_size=4)

    start = time.time()
    for x, y in tqdm(dataloader):
        pass
    end = time.time()
    print("SimpleMemMapDataset execution time:", show_time(end - start))
    print("SimpleMemMapDataset execution time with conversion:", show_time(end - start0))
    print(f"{x.shape=} {y.shape=}")
