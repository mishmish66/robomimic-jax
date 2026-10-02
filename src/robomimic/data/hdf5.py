"""
robomimic hdf5 datasets: filter keys, environment metadata, and packing demos into compact npz files.
"""
import json
from pathlib import Path

import h5py
import numpy as np


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


# per-step datasets of a demo, with their packed and unpacked dtypes; observations are recomputed from the states
_PER_STEP = {
    "states": (np.float32, np.float64), "actions": (np.float32, np.float64),
    "rewards": (np.float32, np.float64), "dones": (np.int8, np.int64),
}


def pack_demos(hdf5_path, npz_path):
    """
    Writes the demos of robomimic dataset `hdf5_path` to compressed `npz_path`: per-step arrays concatenated
    over demos (states, actions, and rewards in float32), each distinct model xml once, masks, and
    per-step annotations. Observations are not kept.
    """
    with h5py.File(Path(hdf5_path).expanduser(), "r") as f:
        names = sorted(f["data"], key=lambda k: int(k.removeprefix("demo_")))
        demos = [f[f"data/{n}"] for n in names]
        models = list(dict.fromkeys(d.attrs["model_file"] for d in demos))
        packed = dict(
            env_args=np.array(f["data"].attrs["env_args"]),
            names=np.array(names),
            lengths=np.array([len(d["actions"]) for d in demos]),
            models=np.array(models),
            model_index=np.array([models.index(d.attrs["model_file"]) for d in demos]),
            success=np.array([bool(d.attrs.get("success", True)) for d in demos]),
        )
        annotations = sorted({k for d in demos for k, v in d.items() if isinstance(v, h5py.Dataset)} - set(_PER_STEP))
        for key in (*_PER_STEP, *annotations):
            values = np.concatenate([d[key][()] for d in demos])
            packed[f"step/{key}"] = values.astype(_PER_STEP[key][0]) if key in _PER_STEP else values
        for key in f.get("mask", {}):
            packed[f"mask/{key}"] = np.array(get_demos_for_filter_key(hdf5_path, key))
    np.savez_compressed(Path(npz_path).expanduser(), **packed)


def unpack_demos(npz_path, hdf5_path):
    """Writes the demos packed in `npz_path` as robomimic dataset `hdf5_path`, without observations."""
    with np.load(Path(npz_path).expanduser()) as packed, h5py.File(Path(hdf5_path).expanduser(), "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = str(packed["env_args"])
        lengths = packed["lengths"]
        data.attrs["total"] = int(lengths.sum())
        starts = np.concatenate([[0], np.cumsum(lengths)])
        steps = {k.removeprefix("step/"): packed[k] for k in packed.files if k.startswith("step/")}
        for i, name in enumerate(packed["names"]):
            demo = data.create_group(str(name))
            demo.attrs["model_file"] = str(packed["models"][packed["model_index"][i]])
            demo.attrs["num_samples"] = int(lengths[i])
            demo.attrs["success"] = bool(packed["success"][i])
            for key, values in steps.items():
                value = values[starts[i]:starts[i + 1]]
                demo.create_dataset(key, data=value.astype(_PER_STEP[key][1]) if key in _PER_STEP else value)
        for key in packed.files:
            if key.startswith("mask/"):
                f[key] = packed[key].astype("S")
