"""
The robomimic demonstration datasets: their paths in the Hugging Face repo, filter keys, metadata, and downloading.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

HF_REPO_ID = "robomimic/robomimic_datasets"
# default directory of the downloaded datasets
DATA_DIR = Path(os.environ.get("ROBOMIMIC_DATA", "~/data/robomimic_v15")).expanduser()

# (tasks, hdf5 types) released for each dataset type
_RELEASES = {
    "ph": (("lift", "can", "square", "transport", "tool_hang"), ("raw", "low_dim")),
    "mh": (("lift", "can", "square", "transport"), ("raw", "low_dim")),
    "mg": (("lift", "can"), ("raw", "low_dim_sparse", "low_dim_dense")),
    "paired": (("can",), ("raw", "low_dim")),
}
# path in the Hugging Face repo of each (task, dataset type, hdf5 type)
DATASETS = {
    (t, dt, h): f"v1.5/{t}/{dt}/{'demo' if h == 'raw' else h}_v15.hdf5"
    for dt, (tasks, types) in _RELEASES.items() for t in tasks for h in types
}


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """Store the demos `demo_keys` as filter key `key_name` of the hdf5 file `hdf5_path`; returns their lengths."""
    with h5py.File(hdf5_path, "a") as f:
        lengths = [f[f"data/{k}"].attrs["num_samples"] for k in demo_keys]
        key = f"mask/{key_name}"
        if key in f:
            del f[key]
        f[key] = np.array(demo_keys, dtype="S")
    return lengths


def get_demos_for_filter_key(hdf5_path, filter_key):
    """Demo keys of filter key `filter_key` of the hdf5 file `hdf5_path`."""
    with h5py.File(hdf5_path, "r") as f:
        return list(f[f"mask/{filter_key}"].asstr()[()])


def get_env_metadata_from_dataset(dataset_path):
    """The dataset's environment metadata: "env_name", "type", and the constructor's "env_kwargs"."""
    with h5py.File(Path(dataset_path).expanduser(), "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta["env_kwargs"].pop("env_lang", None)
    return env_meta


def download_file_from_hf(repo_id, filename, download_dir, check_overwrite=True):
    """
    Download `filename` of the Hugging Face dataset repo `repo_id` into `download_dir`; with `check_overwrite`,
    ask before overwriting an existing file.
    """
    file_to_write = Path(download_dir) / Path(filename).name
    if check_overwrite and file_to_write.exists():
        response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n\n")
        if response.lower() not in {"yes", "y"}:
            raise FileExistsError(f"{file_to_write} already exists")
    with tempfile.TemporaryDirectory() as td:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=td)
        shutil.move(Path(path).resolve(), file_to_write)
