"""
The released robomimic demonstration datasets: where they live in the Hugging Face repo, and downloading them.
"""
import os
import shutil
import tempfile
from pathlib import Path

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
