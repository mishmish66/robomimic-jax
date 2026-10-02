"""
Split a dataset's demonstrations, or those of --filter_key, into training and validation filter keys "train" and
"valid", prefixed by "<filter_key>_" when given.

Example usage:

    python -m robomimic.scripts.split_train_val --dataset /path/to/demo.hdf5 --ratio 0.1
"""
import argparse

import h5py
import numpy as np

from robomimic.data import hdf5


def split_train_val_from_hdf5(hdf5_path, val_ratio=0.1, filter_key=None):
    """Split the demonstrations of `hdf5_path`, or of its filter key `filter_key`, with a share `val_ratio` for validation."""
    if filter_key is not None:
        print(f"using filter key: {filter_key}")
        demos = sorted(hdf5.get_demos_for_filter_key(hdf5_path, filter_key))
    else:
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"])

    num_demos = len(demos)
    num_val = int(val_ratio * num_demos)
    mask = np.zeros(num_demos)
    mask[:num_val] = 1.
    np.random.shuffle(mask)
    train_keys = [demos[i] for i in np.flatnonzero(1. - mask)]
    valid_keys = [demos[i] for i in np.flatnonzero(mask)]
    print(f"{num_val} validation demonstrations out of {num_demos} total demonstrations.")

    prefix = f"{filter_key}_" if filter_key is not None else ""
    train_lengths = hdf5.create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=train_keys, key_name=f"{prefix}train")
    valid_lengths = hdf5.create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=valid_keys, key_name=f"{prefix}valid")

    print(f"Total number of train samples: {np.sum(train_lengths)}")
    print(f"Average number of train samples {np.mean(train_lengths)}")

    print(f"Total number of valid samples: {np.sum(valid_lengths)}")
    print(f"Average number of valid samples {np.mean(valid_lengths)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to hdf5 dataset")
    parser.add_argument(
        "--filter_key",
        help="if provided, split the demonstrations of this filter key instead of all demonstrations",
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="validation ratio, in (0, 1)")
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    split_train_val_from_hdf5(args.dataset, val_ratio=args.ratio, filter_key=args.filter_key)


if __name__ == "__main__":
    main()
