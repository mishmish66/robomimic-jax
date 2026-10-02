"""
Transfer a robomimic dataset's demonstrations into the MuJoCo Warp POMDP (see `robomimic.data.transfer`).

Example:
    python -m robomimic.scripts.transfer_demos --dataset ~/data/robomimic_v15/square/mh/demo_v15.hdf5 --output square_mh_warp.hdf5
"""
import argparse
from pathlib import Path

from robomimic.data.transfer import merge, transfer_dataset


def main(args):
    if args.merge:
        return merge(Path(args.output).expanduser(), args.merge)
    options = vars(args)
    for name in ("merge",):
        options.pop(name)
    options["reuse"] = options["reuse"] or ()
    transfer_dataset(options.pop("dataset"), options.pop("output"), **options)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="robomimic hdf5 dataset with states and actions")
    parser.add_argument("--output", required=True, help="hdf5 file for the transferred demos")
    parser.add_argument("--samples", type=int, nargs="+", default=[32, 256],
                        help="action sequences sampled per chunk, at each stage of search")
    parser.add_argument("--horizon", type=int, default=10, help="steps each sampled sequence is rolled out")
    parser.add_argument("--commit", type=int, default=5, help="steps of the best sequence kept per chunk")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2],
                        help="sampling rounds per chunk, each at half the noise, at each stage of search")
    parser.add_argument("--noise", type=float, default=0.1, help="standard deviation of the arm action perturbation")
    parser.add_argument("--worlds", type=int, default=4096, help="worlds simulated together; each stage fills them with demos")
    parser.add_argument("--shard", default="0/1", help="i/n: transfer every n-th demo starting at the i-th")
    parser.add_argument("--limit", type=int, default=0, help="transfer at most this many demos")
    parser.add_argument("--demos", nargs="+", help="transfer only these demos")
    parser.add_argument("--retries", type=int, default=3,
                        help="reruns of unsuccessful demos at the last stage, each at 1.5 times the noise")
    parser.add_argument("--refine", type=int, default=0,
                        help="extra sampling rounds, each at 1.5 times the noise, for chunks that still drift")
    parser.add_argument("--tolerance", type=float, default=1.,
                        help="mean tracking cost per step (1 = 1 cm of object error) above which a chunk drifts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge", nargs="+", help="shard files to combine into --output instead of transferring")
    parser.add_argument("--reuse", nargs="+", help="earlier outputs whose successful demos are copied, not transferred")
    main(parser.parse_args())
