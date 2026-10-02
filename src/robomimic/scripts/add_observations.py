"""
Add the POMDP's observations of each recorded state to a robomimic dataset, as "obs" and "next_obs" (see
`robomimic.data.observations`). With --npz, the dataset is first written from demos packed by
`robomimic.data.hdf5.pack_demos`.

Examples:
    python -m robomimic.scripts.add_observations --dataset ~/data/robomimic_warp/lift/ph/demo_v15.hdf5
    python -m robomimic.scripts.add_observations --npz datasets/warp/lift_ph.npz --dataset lift_ph_warp.hdf5
"""
import argparse

from robomimic.data.hdf5 import unpack_demos
from robomimic.data.observations import add_observations


def main(args):
    if args.npz:
        unpack_demos(args.npz, args.dataset)
    add_observations(args.dataset, args.batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="hdf5 dataset with recorded states")
    parser.add_argument("--npz", help="packed demos to write to --dataset first")
    parser.add_argument("--batch", type=int, default=256, help="states observed together")
    main(parser.parse_args())
