"""
Generate demonstrations of a MimicGen task variant in the MuJoCo Warp POMDP from transferred source demonstrations
(see `robomimic.data.generate`).

Example:
    python -m robomimic.scripts.generate_demos --task mug_cleanup_d1 --source mug_cleanup_source_warp.hdf5 --output mug_cleanup_d1_warp.hdf5
"""
import argparse

from robomimic.data.generate import generate_dataset


def main(args):
    generate_dataset(args.task, args.source, args.output, demos=args.demos, worlds=args.worlds, seed=args.seed,
                     max_rounds=args.max_rounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="MimicGen task variant to generate, e.g. mug_cleanup_d1")
    parser.add_argument("--source", required=True, help="hdf5 file of the task's source demos, transferred")
    parser.add_argument("--output", required=True, help="hdf5 file for the generated demos")
    parser.add_argument("--demos", type=int, default=1000, help="successful demos to generate")
    parser.add_argument("--worlds", type=int, default=1024, help="demos attempted together")
    parser.add_argument("--max-rounds", type=int, default=20, help="rounds of attempts at most")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
