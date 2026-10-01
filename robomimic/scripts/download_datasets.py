"""
Download robomimic datasets from Hugging Face into <download_dir>/<task>/<dataset_type>/. --tasks,
--dataset_types, and --hdf5_types each take a list of values or "all".

Example usage:

    # default: the lift proficient-human low-dim dataset
    python download_datasets.py

    # low-dim proficient-human datasets of all tasks (dry run first to list them)
    python download_datasets.py --tasks all --dataset_types ph --hdf5_types low_dim --dry_run
    python download_datasets.py --tasks all --dataset_types ph --hdf5_types low_dim

    # raw and low-dim multi-human datasets of the can and square tasks
    python download_datasets.py --tasks can square --dataset_types mh --hdf5_types raw low_dim

    # sparse-reward machine-generated low-dim datasets
    python download_datasets.py --tasks all --dataset_types mg --hdf5_types low_dim_sparse
"""
import argparse
from pathlib import Path

from robomimic import datasets

ALL_TASKS = list(dict.fromkeys(t for t, _, _ in datasets.DATASETS))
ALL_DATASET_TYPES = list(dict.fromkeys(dt for _, dt, _ in datasets.DATASETS))
ALL_HDF5_TYPES = list(dict.fromkeys(h for _, _, h in datasets.DATASETS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dir", type=Path, default=datasets.DATA_DIR,
        help="base download directory, created if missing; defaults to $ROBOMIMIC_DATA or ~/data/robomimic_v15",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["lift"], choices=[*ALL_TASKS, "all"],
        help="tasks to download datasets for; defaults to lift",
    )
    parser.add_argument(
        "--dataset_types", nargs="+", default=["ph"], choices=[*ALL_DATASET_TYPES, "all"],
        help="dataset types (e.g. ph, mh, mg) to download datasets for; defaults to ph",
    )
    parser.add_argument(
        "--hdf5_types", nargs="+", default=["low_dim"], choices=[*ALL_HDF5_TYPES, "all"],
        help="hdf5 types (e.g. raw, low_dim) to download datasets for; defaults to low_dim",
    )
    parser.add_argument("--dry_run", action="store_true", help="only print which datasets would be downloaded")
    args = parser.parse_args()

    chosen = lambda values, value: "all" in values or value in values
    for (task, dataset_type, hdf5_type), path in datasets.DATASETS.items():
        if not (chosen(args.tasks, task) and chosen(args.dataset_types, dataset_type) and chosen(args.hdf5_types, hdf5_type)):
            continue
        download_dir = (args.download_dir / task / dataset_type).absolute()
        print(
            f"\nDownloading dataset:\n    task: {task}\n    dataset type: {dataset_type}\n    hdf5 type: {hdf5_type}"
            f"\n    file: {path}\n    download path: {download_dir}"
        )
        if args.dry_run:
            print("dry run: skip download")
            continue
        download_dir.mkdir(parents=True, exist_ok=True)
        datasets.download_file_from_hf(datasets.HF_REPO_ID, path, download_dir)


if __name__ == "__main__":
    main()
