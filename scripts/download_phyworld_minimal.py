"""Download the minimal PhyWorld files used by the single-law sanity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download


FILES = (
    "id_ood_data/uniform_motion_30K.hdf5",
    "id_ood_data/uniform_motion_eval.hdf5",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=str, default="data/phyworld")
    args = parser.parse_args()
    for filename in FILES:
        path = hf_hub_download(
            repo_id="magicr/phyworld",
            repo_type="dataset",
            filename=filename,
            local_dir=args.local_dir,
        )
        print(path)


if __name__ == "__main__":
    main()
