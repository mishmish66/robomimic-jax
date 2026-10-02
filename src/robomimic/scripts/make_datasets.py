"""
Build robomimic low_dim datasets of the transferred demonstrations: each packed file `<task>_<type>.npz` in
--packed becomes `<output>/<task>/<type>/low_dim_v15.hdf5`, with the POMDP's "obs" and "next_obs" of every
recorded state, ready for robomimic's dataset loaders.

Examples:
    python -m robomimic.scripts.make_datasets
    python -m robomimic.scripts.make_datasets --tasks lift_ph square_mh --output ~/data/robomimic_warp
"""
import argparse
import time
from pathlib import Path

from robomimic.data.hdf5 import unpack_demos
from robomimic.data.observations import add_observations


def main(args):
    packed = sorted(Path(args.packed).glob("*.npz"))
    if args.tasks:
        packed = [p for p in packed if p.stem in args.tasks]
        missing = set(args.tasks) - {p.stem for p in packed}
        if missing:
            raise SystemExit(f"no packed demos for {sorted(missing)} in {args.packed}")
    for path in packed:
        task, dataset_type = path.stem.rsplit("_", 1)
        dataset = Path(args.output).expanduser() / task / dataset_type / "low_dim_v15.hdf5"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        unpack_demos(path, dataset)
        add_observations(dataset, args.batch)
        print(f"{path} -> {dataset} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", default="datasets/warp", help="directory of the packed demos")
    parser.add_argument("--output", default="~/data/robomimic_warp", help="base directory of the datasets")
    parser.add_argument("--tasks", nargs="+", help="packed files to build, by name (e.g. lift_ph); defaults to all")
    parser.add_argument("--batch", type=int, default=256, help="states observed together")
    main(parser.parse_args())
