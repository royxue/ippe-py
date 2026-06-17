"""Direct Linear Transform (DLT) homography estimation.

Translated from Peter Kovesi's MATLAB implementation
(http://www.peterkovesi.com/matlabfns/).
"""

from __future__ import annotations

import numpy as np

__all__ = ["homography2d", "normalise2dpts"]


def homography2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Estimate the homography mapping ``x1`` to ``x2`` via the DLT.

    Parameters
    ----------
    x1, x2:
        ``3xN`` arrays of homogeneous 2D points such that ``x1 <-> x2``.

    Returns
    -------
    H:
        The ``3x3`` homography such that ``x2 = H @ x1``.
    """
    x1, T1 = normalise2dpts(x1)
    x2, T2 = normalise2dpts(x2)

    npts = x1.shape[1]
    A = np.zeros((3 * npts, 9))
    O = np.zeros(3)

    for i in range(npts):
        X = x1[:, i].T
        x, y, w = x2[0, i], x2[1, i], x2[2, i]
        A[3 * i] = np.concatenate((O, -w * X, y * X))
        A[3 * i + 1] = np.concatenate((w * X, O, -x * X))
        A[3 * i + 2] = np.concatenate((-y * X, x * X, O))

    _, _, Vh = np.linalg.svd(A)
    V = Vh.T

    H = V[:, 8].reshape(3, 3)

    # Denormalise: H = inv(T2) @ H @ T1
    H = np.linalg.solve(T2, H)
    H = H.dot(T1)

    return H


def normalise2dpts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate and scale 2D homogeneous points for better conditioning.

    The points are transformed so that their centroid is at the origin and
    their mean distance from the origin is ``sqrt(2)``. This typically
    improves the conditioning of equations used to solve for homographies,
    fundamental matrices, etc.

    Parameters
    ----------
    pts:
        ``3xN`` array of 2D homogeneous coordinates.

    Returns
    -------
    newpts:
        ``3xN`` array of transformed homogeneous coordinates.
    T:
        The ``3x3`` transformation matrix such that ``newpts = T @ pts``.
    """
    if pts.shape[0] != 3:
        raise ValueError("input points must be 3xN homogeneous coordinates")

    pts = np.array(pts, dtype=float)

    finite = np.flatnonzero(np.abs(pts[2, :]) > np.spacing(1))

    # Scale finite points to a unit homogeneous coordinate.
    pts[0, finite] /= pts[2, finite]
    pts[1, finite] /= pts[2, finite]
    pts[2, finite] = 1.0

    c = np.mean(pts[0:2, finite], axis=1)

    newp1 = pts[0, finite] - c[0]
    newp2 = pts[1, finite] - c[1]
    dist = np.sqrt(newp1 ** 2 + newp2 ** 2)
    meandist = np.mean(dist)

    scale = np.sqrt(2) / meandist

    T = np.array(
        [
            [scale, 0, -scale * c[0]],
            [0, scale, -scale * c[1]],
            [0, 0, 1],
        ]
    )

    newpts = T.dot(pts)

    return newpts, T
