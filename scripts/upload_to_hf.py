"""Mirror the OpenBiomechanics pitching data into a HuggingFace dataset.

This lets a deployed dashboard source all of its data directly from the Hub
(set ``OBP_HF_DATASET`` to the same repo id). It fetches the processed tables
and full-signal archives through the normal data-source layer (so they land in
the local cache), then uploads them to your dataset repo preserving the
``poi/``, ``full_sig/`` layout the dashboard expects.

Usage::

    huggingface-cli login            # or set HF_TOKEN
    python scripts/upload_to_hf.py --repo your-name/openbiomechanics-pitching

Optionally pass ``--with-c3d 1097_1,1031_2`` to also upload specific raw C3D
files (the full C3D set is large, so none are uploaded by default).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_sources  # noqa: E402

# (relative path under baseball_pitching/data, cached filename)
CORE_FILES = [
    ("poi/poi_metrics.csv", "poi_metrics.csv"),
    ("metadata.csv", "metadata.csv"),
    ("full_sig/force_plate.zip", "force_plate.zip"),
    ("full_sig/energy_flow.zip", "energy_flow.zip"),
    ("full_sig/landmarks.zip", "landmarks.zip"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True,
                    help="Target HF dataset repo id, e.g. you/obp-pitching")
    ap.add_argument("--with-c3d", default="",
                    help="Comma-separated session_pitch ids whose C3D to upload")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True,
                    private=args.private)

    def upload(local_path, path_in_repo):
        print(f"  uploading {path_in_repo} ...")
        api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                        repo_id=args.repo, repo_type="dataset")

    print(f"Mirroring core data into dataset '{args.repo}':")
    for rel, name in CORE_FILES:
        local = data_sources.get(rel, local_name=name)
        upload(local, rel)

    if args.with_c3d.strip():
        from velocity_model import load_dataset
        ds = load_dataset()
        for sp in [s.strip() for s in args.with_c3d.split(",") if s.strip()]:
            filename = ds.c3d_filename(sp)
            session = filename.split("_")[0]
            rel = f"c3d/{session}/{filename}"
            local = data_sources.get(rel, local_name=filename, dest_dir="c3d_data")
            upload(local, rel)

    print(f"\nDone. Point the dashboard at it with:\n"
          f"  export OBP_HF_DATASET={args.repo}")


if __name__ == "__main__":
    main()
