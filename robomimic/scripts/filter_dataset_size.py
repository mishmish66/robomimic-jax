"""
Create filter keys of random subsets of a dataset's demonstrations: "<n>_demos" (or --output_filter_key) for each
--num_demos n, drawn from the demonstrations of --input_filter_key, and then prefixed by "<input_filter_key>_",
when given.

Example usage:

    python filter_dataset_size.py --dataset /path/to/demo.hdf5 --num_demos 10 20
"""
import argparse

import h5py
import numpy as np

from robomimic import datasets


def filter_dataset_size(hdf5_path, num_demos, input_filter_key=None, output_filter_key=None):
    if input_filter_key is not None:
        print(f"using filter key: {input_filter_key}")
        demos = sorted(datasets.get_demos_for_filter_key(hdf5_path, input_filter_key))
    else:
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"])

    mask = np.zeros(len(demos))
    mask[:num_demos] = 1.
    np.random.shuffle(mask)
    subset_keys = [demos[i] for i in np.flatnonzero(mask)]

    name = output_filter_key if output_filter_key is not None else f"{num_demos}_demos"
    if input_filter_key is not None:
        name = f"{input_filter_key}_{name}"
    subset_lengths = datasets.create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=subset_keys, key_name=name)

    print(f"Total number of subset samples: {np.sum(subset_lengths)}")
    print(f"Average number of subset samples {np.mean(subset_lengths)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to hdf5 dataset")
    parser.add_argument(
        "--input_filter_key",
        help="if provided, draw the subsets from the demonstrations of this filter key instead of all demonstrations",
    )
    parser.add_argument("--num_demos", type=int, nargs="+", required=True, help="number of demonstrations of each subset")
    parser.add_argument("--output_filter_key", help="name of the output filter key instead of <n>_demos")
    args = parser.parse_args()
    if args.output_filter_key is not None and len(args.num_demos) > 1:
        parser.error("--output_filter_key names a single subset, so it takes a single --num_demos")

    # seed to make sure results are consistent
    np.random.seed(0)

    for n in args.num_demos:
        filter_dataset_size(
            args.dataset, input_filter_key=args.input_filter_key, num_demos=n, output_filter_key=args.output_filter_key
        )


if __name__ == "__main__":
    main()
