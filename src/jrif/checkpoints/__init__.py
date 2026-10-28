"""Shipped model weights, bundled with the package.

Usage:
    from jrif import RTPredictor
    from jrif.checkpoints import checkpoint_path
    p = RTPredictor.from_checkpoint(checkpoint_path("rt_psmtag"))
"""

from importlib.resources import files
from pathlib import Path

_NAMES = {
    "rt_lf": "rt_lf.pt",          # RT, label-free chemistry
    "rt_psmtag": "rt_psmtag.pt",  # RT, PSMtag chemistry
    "im_psmtag": "im_psmtag.pt",  # IM, PSMtag chemistry
}


def checkpoint_path(name: str) -> Path:
    """Filesystem path of a shipped checkpoint: 'rt_lf', 'rt_psmtag', 'im_psmtag'."""
    if name not in _NAMES:
        raise KeyError(f"unknown checkpoint {name!r}; available: {sorted(_NAMES)}")
    return Path(str(files(__package__) / _NAMES[name]))
