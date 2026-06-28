"""Central data sourcing for the dashboard.

Every table / C3D / signal file is fetched through :func:`get`, which resolves a
path *relative to* the OpenBiomechanics ``baseball_pitching/data`` directory and
returns a local file path, downloading and caching on first use.

Two backends are supported, chosen automatically:

* **HuggingFace dataset** — if the ``OBP_HF_DATASET`` environment variable is set
  (e.g. ``"your-name/openbiomechanics-pitching"``), files are pulled from that
  HF dataset repo with ``huggingface_hub``. This lets a deployed dashboard
  (e.g. a HuggingFace Space) source all of its data directly from the Hub.
* **OpenBiomechanics GitHub** (default) — files are downloaded from the public
  ``raw.githubusercontent.com`` mirror of the dataset.

Use ``scripts/upload_to_hf.py`` to mirror the processed data into a HF dataset.
"""

from __future__ import annotations

import os
import shutil
import urllib.request

OBP_BASE = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
    "main/baseball_pitching/data"
)

# Env vars (read lazily so they can be set before any fetch).
ENV_HF_DATASET = "OBP_HF_DATASET"
ENV_CACHE_DIR = "OBP_DATA_DIR"

DEFAULT_CACHE = "obp_data"


def cache_dir() -> str:
    return os.environ.get(ENV_CACHE_DIR, DEFAULT_CACHE)


def hf_dataset() -> str | None:
    return os.environ.get(ENV_HF_DATASET) or None


def _from_hf(rel: str, local: str) -> str | None:
    """Try to fetch ``rel`` from the configured HF dataset into ``local``."""
    repo = hf_dataset()
    if not repo:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        path = hf_hub_download(repo_id=repo, filename=rel, repo_type="dataset")
    except Exception:
        return None
    # Copy out of the HF cache so callers see a stable path under cache_dir.
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    if os.path.abspath(path) != os.path.abspath(local):
        shutil.copyfile(path, local)
    return local


def get(rel: str, local_name: str | None = None, dest_dir: str | None = None) -> str:
    """Return a local path to the data file ``rel`` (download + cache on miss).

    Parameters
    ----------
    rel:
        Path relative to ``baseball_pitching/data`` — e.g.
        ``"poi/poi_metrics.csv"``, ``"full_sig/force_plate.zip"``,
        ``"c3d/000774/000774_..._776.c3d"``.
    local_name:
        Filename to cache as (defaults to the basename of ``rel``).
    dest_dir:
        Cache directory (defaults to ``OBP_DATA_DIR`` or ``obp_data``).
    """
    rel = rel.lstrip("/")
    dest_dir = dest_dir or cache_dir()
    local = os.path.join(dest_dir, local_name or os.path.basename(rel))
    if os.path.exists(local):
        return local
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)

    if _from_hf(rel, local):
        return local

    urllib.request.urlretrieve(f"{OBP_BASE}/{rel}", local)
    return local


def describe() -> str:
    """Human-readable description of the active data source."""
    repo = hf_dataset()
    if repo:
        return f"HuggingFace dataset '{repo}' (cache: {cache_dir()})"
    return f"OpenBiomechanics GitHub mirror (cache: {cache_dir()})"
