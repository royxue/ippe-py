"""Infinitesimal Plane-based Pose Estimation (IPPE).

A Python port of Toby Collins' MATLAB implementation
(https://github.com/tobycollins/IPPE).

Typical usage::

    import ippe

    # U: 2xN model points on the z=0 plane, Q: 2xN normalised image points
    poses = ippe.mat_run(U, Q)
    R, t = poses["R1"], poses["t1"]
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .core import mat_run, run
from .homography import homography2d

try:
    __version__ = version("ippe")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = ["run", "mat_run", "homography2d", "__version__"]
