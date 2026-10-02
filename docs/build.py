"""Render the documentation site with pdoc.

    JAX_PLATFORMS=cpu MUJOCO_GL=disable uv run --extra video --with pdoc python docs/build.py --out site

The site serves the task images that `docs/tasks.md` shows from `docs/images`, rendered by `docs/images.py`.
`.github/workflows/docs.yml` builds and publishes the site on every push to `master`.
"""

import argparse
import shutil
from pathlib import Path

import pdoc
import pdoc.docstrings
import pdoc.render

DOCS = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    # The site links the images rather than inlining them.
    pdoc.docstrings.embed_images = lambda docstring, source_file: docstring
    pdoc.render.configure(docformat="restructuredtext", search=True)
    pdoc.pdoc("robomimic", output_directory=args.out)
    shutil.copytree(DOCS / "images", args.out / "robomimic" / "images", dirs_exist_ok=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
