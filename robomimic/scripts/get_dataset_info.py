"""
Report a dataset's trajectory length statistics, action bounds, filter keys, environment metadata, and the
structure of its first demonstration. With --verbose, also list the demos of each filter key and the structure
of every demonstration.

Example usage:

    # run script on example hdf5 packaged with repository
    python get_dataset_info.py --dataset ../../tests/assets/test_v15.hdf5

    # run script only on validation data
    python get_dataset_info.py --dataset ../../tests/assets/test_v15.hdf5 --filter_key valid
"""
import argparse
import json

import h5py
import numpy as np

from robomimic import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to hdf5 dataset")
    parser.add_argument(
        "--filter_key",
        help="(optional) if provided, report statistics on the subset of trajectories in the file that correspond "
        "to this filter key",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()

    all_filter_keys = None
    if args.filter_key is not None:
        print(f"NOTE: using filter key {args.filter_key}")
        demos = datasets.get_demos_for_filter_key(args.dataset, args.filter_key)
    else:
        with h5py.File(args.dataset, "r") as f:
            demos = list(f["data"])
            masks = list(f["mask"]) if "mask" in f else None
        if masks is not None:
            all_filter_keys = {fk: sorted(datasets.get_demos_for_filter_key(args.dataset, fk)) for fk in masks}
    demos.sort(key=lambda k: int(k.removeprefix("demo_")))

    with h5py.File(args.dataset, "r") as f:
        traj_lengths = []
        action_min = np.inf
        action_max = -np.inf
        for ep in demos:
            actions = f[f"data/{ep}/actions"][()]
            traj_lengths.append(actions.shape[0])
            action_min = min(action_min, np.min(actions))
            action_max = max(action_max, np.max(actions))
        traj_lengths = np.array(traj_lengths)

        print("")
        print(f"total transitions: {np.sum(traj_lengths)}")
        print(f"total trajectories: {traj_lengths.shape[0]}")
        print(f"traj length mean: {np.mean(traj_lengths)}")
        print(f"traj length std: {np.std(traj_lengths)}")
        print(f"traj length min: {np.min(traj_lengths)}")
        print(f"traj length max: {np.max(traj_lengths)}")
        print(f"action min: {action_min}")
        print(f"action max: {action_max}")
        print("")
        print("==== Filter Keys ====")
        if all_filter_keys is not None:
            for fk, fk_demos in all_filter_keys.items():
                print(f"filter key {fk} with {len(fk_demos)} demos")
        else:
            print("no filter keys")
        print("")
        if args.verbose:
            if all_filter_keys is not None:
                print("==== Filter Key Contents ====")
                for fk, fk_demos in all_filter_keys.items():
                    print(f"filter_key {fk} with {len(fk_demos)} demos: {fk_demos}")
            print("")
        print("==== Env Meta ====")
        print(json.dumps(datasets.get_env_metadata_from_dataset(args.dataset), indent=4))
        print("")

        print("==== Dataset Structure ====")
        for ep in demos:
            demo = f[f"data/{ep}"]
            print(f"episode {ep} with {demo.attrs['num_samples']} transitions")
            for k in demo:
                if k in ["obs", "next_obs"]:
                    print(f"    key: {k}")
                    for obs_k in demo[k]:
                        print(f"        observation key {obs_k} with shape {demo[f'{k}/{obs_k}'].shape}")
                elif isinstance(demo[k], h5py.Dataset):
                    print(f"    key: {k} with shape {demo[k].shape}")
            if not args.verbose:
                break

    print("")
    if action_min < -1. or action_max > 1.:
        raise SystemExit(f"Dataset should have actions in [-1., 1.] but got bounds [{action_min}, {action_max}]")


if __name__ == "__main__":
    main()
