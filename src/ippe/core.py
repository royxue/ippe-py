"""Infinitesimal Plane-based Pose Estimation (IPPE).

Python port of Toby Collins' MATLAB implementation
(https://github.com/tobycollins/IPPE), described in:

    Collins, T. & Bartoli, A. "Infinitesimal Plane-Based Pose Estimation".
    International Journal of Computer Vision 109(3), 252-286, 2014.
    https://doi.org/10.1007/s11263-014-0725-5
"""

from __future__ import annotations

import numpy as np

from .homography import homography2d

__all__ = ["run", "mat_run"]

_VALID_HOMOGRAPHY_METHODS = ("DLT", "Harker")


def run(U: np.ndarray, Q: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> dict:
    """Undistort image points and estimate pose with IPPE.

    This is a thin convenience wrapper around :func:`mat_run` that first
    removes the camera intrinsics and lens distortion from ``Q`` using
    OpenCV's ``undistortPoints``.

    Parameters
    ----------
    U:
        ``Nx2`` (or ``Nx3``) array of model points in world coordinates.
    Q:
        ``Nx2`` array of observed image points (in pixels).
    mtx:
        ``3x3`` camera intrinsic matrix.
    dist:
        Lens distortion coefficients accepted by OpenCV.

    Returns
    -------
    dict
        See :func:`mat_run`.

    Notes
    -----
    Requires the optional ``opencv-python`` dependency, imported lazily so
    that :func:`mat_run` can be used without OpenCV installed.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "ippe.run() requires opencv-python; install it with "
            "`pip install ippe[opencv]` or call ippe.mat_run() with "
            "pre-normalised points instead."
        ) from exc

    k = Q.shape[0]
    Q = Q.reshape(1, k, 2)
    Q = cv2.undistortPoints(Q, mtx, dist)
    Q = Q.reshape(k, 2)
    return mat_run(U.T, Q.T)


def mat_run(U: np.ndarray, Q: np.ndarray, hEstMethod: str = "DLT") -> dict:
    """Solve perspective IPPE from plane-to-image point correspondences.

    Parameters
    ----------
    U:
        ``2xN`` or ``3xN`` matrix holding the model points in world
        coordinates. If ``2xN`` the points lie on the plane ``z = 0``.
    Q:
        ``2xN`` matrix of normalised image points, i.e. the effects of the
        camera intrinsics and lens distortion have been removed so the points
        project with a perfect pinhole model.
    hEstMethod:
        Homography estimation method. Only ``"DLT"`` (Direct Linear
        Transform, the default) is currently implemented.

    Returns
    -------
    dict
        Dictionary with the two candidate solutions and their reprojection
        errors: ``R1``, ``t1``, ``R2``, ``t2``, ``reprojError1`` and
        ``reprojError2``. Solutions are ordered so that ``R1``/``t1`` has the
        smaller reprojection error.
    """
    if hEstMethod not in _VALID_HOMOGRAPHY_METHODS:
        raise ValueError(f"hEstMethod must be one of {_VALID_HOMOGRAPHY_METHODS}")
    if hEstMethod == "Harker":
        raise NotImplementedError("the Harker homography method is not yet supported")

    # Defensive copy: mat_run centres the model points in place below.
    U = np.array(U, dtype=float)
    Q = np.asarray(Q, dtype=float)

    assert U.shape[0] in (2, 3)
    assert U.shape[1] == Q.shape[1]
    assert Q.shape[0] == 2

    n = U.shape[1]
    modelDims = U.shape[0]

    Uuncentred = U

    if modelDims == 2:
        # Zero-centre the model points.
        Pbar = np.vstack((np.mean(U, axis=1, keepdims=True), 0))
        U[0, :] = U[0, :] - Pbar[0]
        U[1, :] = U[1, :] - Pbar[1]
    else:
        # Rotate the model points onto the plane z=0 and zero-centre them.
        Pbar = np.mean(U[:1])
        MCenter = np.eye(4)
        MCenter[0:3, -1] = -Pbar
        U_ = MCenter[0:3, :].dot(np.vstack((U, np.ones((1, U.shape[1])))))
        modelRotation, _sigs, _ = np.linalg.svd(U_.dot(U_.T))
        modelRotation = modelRotation.T

        modelRotation = np.hstack(
            (
                np.vstack((modelRotation, np.array([0, 0, 0]))),
                np.array([[0], [0], [0], [1]]),
            )
        )
        Mcorrective = modelRotation.dot(MCenter)
        U = Mcorrective[0:2, :].dot(np.vstack((U, np.ones((1, U.shape[1])))))

    # Compute the model-to-image homography.
    _U = np.vstack((U, np.ones((1, n))))
    _Q = np.vstack((Q, np.ones((1, n))))
    H = homography2d(_U, _Q)
    H = H / H[2, 2]

    # Jacobian J of the homography at (0,0).
    J = np.zeros((2, 2))
    J[0, 0] = H[0, 0] - H[2, 0] * H[0, 2]
    J[0, 1] = H[0, 1] - H[2, 1] * H[0, 2]
    J[1, 0] = H[1, 0] - H[2, 0] * H[1, 2]
    J[1, 1] = H[1, 1] - H[2, 1] * H[1, 2]

    # Rotation solutions.
    v = np.vstack((H[0, 2], H[1, 2]))
    R1, R2, _ = _ippe_dec(v, J)

    # Translation solutions.
    t1_ = _est_t(R1, U, Q)
    t2_ = _est_t(R2, U, Q)

    if modelDims == 2:
        t1 = np.hstack((R1, t1_)).dot(np.vstack((-Pbar, 1)))
        t2 = np.hstack((R2, t2_)).dot(np.vstack((-Pbar, 1)))
    else:
        M1 = np.vstack((np.hstack((R1, t1_)), np.array([0, 0, 0, 1])))
        M2 = np.vstack((np.hstack((R2, t2_)), np.array([0, 0, 0, 1])))
        M1 = M1.dot(Mcorrective)
        M2 = M2.dot(Mcorrective)
        R1 = M1[0:3, 0:3]
        R2 = M2[0:3, 0:3]
        t1 = M1[0:3, -1]
        t2 = M2[0:3, -1]

    reprojErr1, reprojErr2 = _compute_reproj_errs(R1, R2, t1, t2, Uuncentred, Q)

    if reprojErr1 > reprojErr2:
        R1, R2, t1, t2, reprojErr1, reprojErr2 = (
            R2,
            R1,
            t2,
            t1,
            reprojErr2,
            reprojErr1,
        )

    return {
        "R1": R1,
        "t1": t1,
        "R2": R2,
        "t2": t2,
        "reprojError1": reprojErr1,
        "reprojError2": reprojErr2,
    }


def _compute_reproj_errs(R1, R2, t1, t2, U, Q):
    """Reprojection errors for the two IPPE solutions."""
    if U.shape[0] == 2:
        PCam1 = R1[:, 0:2].dot(U)
        PCam2 = R2[:, 0:2].dot(U[0:2, :])
    else:
        PCam1 = R1.dot(U)
        PCam2 = R2.dot(U)

    PCam1[0, :] += t1[0]
    PCam1[1, :] += t1[1]
    PCam1[2, :] += t1[2]

    PCam2[0, :] += t2[0]
    PCam2[1, :] += t2[1]
    PCam2[2, :] += t2[2]

    Qest_1 = PCam1 / PCam1[2, :]
    Qest_2 = PCam2 / PCam2[2, :]

    reprojErr1 = np.linalg.norm(Qest_1[0:2, :] - Q)
    reprojErr2 = np.linalg.norm(Qest_2[0:2, :] - Q)

    return reprojErr1, reprojErr2


def _est_t(R, psPlane, Q):
    """Least-squares estimate of translation given a rotation solution."""
    if psPlane.shape[0] == 2:
        psPlane = np.vstack((psPlane, np.zeros((1, psPlane.shape[1]))))

    Ps = R.dot(psPlane)
    numPts = psPlane.shape[1]

    Ax = np.zeros((numPts, 3))
    bx = np.zeros((numPts, 1))
    Ay = np.zeros((numPts, 3))
    by = np.zeros((numPts, 1))

    Ax[:, 0] = 1
    Ax[:, 2] = -Q[0, :]
    bx[:] = (Q[0, :] * Ps[2, :] - Ps[0, :]).reshape(numPts, 1)

    Ay[:, 1] = 1
    Ay[:, 2] = -Q[1, :]
    by[:] = (Q[1, :] * Ps[2, :] - Ps[1, :]).reshape(numPts, 1)

    A = np.vstack((Ax, Ay))
    b = np.vstack((bx, by))

    AtA = A.conj().T.dot(A)
    Atb = A.conj().T.dot(b)

    return _inv33(AtA).dot(Atb)


def _inv33(A):
    """Closed-form inverse of a full-rank 3x3 matrix."""
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]

    Ainv = np.vstack(
        (
            np.array([a22 * a33 - a23 * a32, a13 * a32 - a12 * a33, a12 * a23 - a13 * a22]),
            np.array([a23 * a31 - a21 * a33, a11 * a33 - a13 * a31, a13 * a21 - a11 * a23]),
            np.array([a21 * a32 - a22 * a31, a12 * a31 - a11 * a32, a11 * a22 - a12 * a21]),
        )
    )
    det = (
        a11 * a22 * a33
        - a11 * a23 * a32
        - a12 * a21 * a33
        + a12 * a23 * a31
        + a13 * a21 * a32
        - a13 * a22 * a31
    )
    return Ainv / det


def _ippe_dec(v, J):
    """Two rotation solutions from the Jacobian of the model-to-plane homography.

    Parameters
    ----------
    v:
        ``2x1`` vector holding the point in normalised pixel coordinates that
        maps by ``H^-1`` to the point ``(0, 0, 0)`` in world coordinates.
    J:
        ``2x2`` Jacobian of ``H`` at ``(0, 0)``.

    Returns
    -------
    R1, R2:
        The two candidate ``3x3`` rotation matrices.
    gamma:
        The positive real-valued inverse-scale factor.
    """
    t = np.linalg.norm(v)

    if t < np.finfo(float).eps:
        # The plane is fronto-parallel to the camera; the corrective rotation
        # Rv is the identity and there is a single pose solution.
        Rv = np.eye(3)
    else:
        s = np.linalg.norm(np.vstack((v, 1)))
        costh = 1.0 / s
        sinth = np.sqrt(1 - 1.0 / s ** 2)
        Kcrs = (
            1.0
            / t
            * np.vstack(
                [
                    np.hstack([np.zeros((2, 2)), v]),
                    np.hstack([-v.T, np.zeros((1, 1))]),
                ]
            )
        )
        Rv = np.eye(3) + sinth * Kcrs + (1.0 - costh) * Kcrs.dot(Kcrs)

    # 2x2 SVD decomposition.
    B = np.hstack((np.eye(2), -v)).dot(Rv[:, 0:2])
    dt = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]
    Binv = np.vstack(
        [
            np.hstack([B[1, 1] / dt, -B[0, 1] / dt]),
            np.hstack([-B[1, 0] / dt, B[0, 0] / dt]),
        ]
    )
    A = Binv.dot(J)

    # Largest singular value of A.
    AAT = A.dot(A.T)
    gamma = np.sqrt(
        1.0
        / 2
        * (AAT[0, 0] + AAT[1, 1] + np.sqrt((AAT[0, 0] - AAT[1, 1]) ** 2 + 4 * AAT[0, 1] ** 2))
    )

    # Reconstruct the full rotation matrices.
    R22_tild = A / gamma

    h = np.eye(2) - R22_tild.T.dot(R22_tild)
    b = np.vstack((np.sqrt(h[0, 0]), np.sqrt(h[1, 1])))
    if h[0, 1] < 0:
        b[1] = -b[1]
    v1 = np.vstack((R22_tild[:, 0:1], np.array([b[0]])))
    v2 = np.vstack((R22_tild[:, 1:2], np.array([b[1]])))
    d = _crs(v1, v2)
    c = d[0:2]
    a = d[2]
    R1 = Rv.dot(np.vstack((np.hstack((R22_tild, c)), np.hstack((b.conj().T, np.array([a]))))))
    R2 = Rv.dot(np.vstack((np.hstack((R22_tild, -c)), np.hstack((-b.conj().T, np.array([a]))))))

    return R1, R2, gamma


def _crs(v1, v2):
    """3D cross product of column vectors ``v1`` and ``v2``."""
    v3 = np.zeros((3, 1))
    v3[0] = v1[1] * v2[2] - v1[2] * v2[1]
    v3[1] = v1[2] * v2[0] - v1[0] * v2[2]
    v3[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return v3
