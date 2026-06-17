"""Synthetic round-trip tests for the IPPE solver.

A planar object with a known pose is projected into a pinhole camera; IPPE
should recover that pose (up to the inherent two-fold ambiguity) with a
negligible reprojection error.
"""

import numpy as np
import pytest

import ippe


def _rotation_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K.dot(K)


def _make_scene(R, t, offset=(0.0, 0.0)):
    # A square planar object on the z=0 plane, optionally shifted off-origin.
    U = np.array(
        [
            [-1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    U = U + np.array(offset).reshape(2, 1)
    P = np.vstack((U, np.zeros((1, U.shape[1]))))  # 3xN on z=0
    Pcam = R.dot(P) + t.reshape(3, 1)
    Q = Pcam[0:2, :] / Pcam[2, :]  # normalised pinhole projection
    return U, Q


def _make_scene_3d(R, t):
    # A non-planar-looking 3D input: points on a tilted plane, given as 3xN.
    U2d = np.array(
        [
            [-1.0, 1.0, 1.0, -1.0, 0.3],
            [-1.0, -1.0, 1.0, 1.0, -0.4],
        ]
    )
    # Embed on a tilted plane through the origin so the model is genuinely 3xN.
    tilt = _rotation_from_axis_angle([1.0, 0.5, 0.0], 0.5)
    P_plane = np.vstack((U2d, np.zeros((1, U2d.shape[1]))))
    U = tilt.dot(P_plane)  # 3xN model points (not on z=0)
    Pcam = R.dot(U) + t.reshape(3, 1)
    Q = Pcam[0:2, :] / Pcam[2, :]
    return U, Q


def test_recovers_known_pose():
    R = _rotation_from_axis_angle([0.2, 1.0, 0.3], 0.4)
    t = np.array([0.05, -0.1, 6.0])
    U, Q = _make_scene(R, t)

    poses = ippe.mat_run(U, Q)

    # The better of the two solutions should reproject essentially perfectly.
    assert poses["reprojError1"] < 1e-9
    assert poses["reprojError1"] <= poses["reprojError2"]

    # And recover the true rotation/translation.
    assert np.allclose(poses["R1"], R, atol=1e-6)
    assert np.allclose(poses["t1"].reshape(3), t, atol=1e-6)


def test_recovers_known_pose_off_origin():
    # Exercises the 2D branch with a model whose centroid is not at the origin;
    # the reprojection error must still be ~0 (regression for the Uuncentred
    # aliasing bug).
    R = _rotation_from_axis_angle([0.2, 1.0, 0.3], 0.4)
    t = np.array([0.05, -0.1, 6.0])
    U, Q = _make_scene(R, t, offset=(5.0, -3.0))

    poses = ippe.mat_run(U, Q)

    assert poses["reprojError1"] < 1e-9
    assert np.allclose(poses["R1"], R, atol=1e-6)
    assert np.allclose(poses["t1"].reshape(3), t, atol=1e-6)


def test_recovers_known_pose_3d_input():
    # Exercises the 3xN model branch (regression for the Pbar centroid bug).
    R = _rotation_from_axis_angle([0.1, 0.8, 0.2], 0.35)
    t = np.array([0.2, 0.1, 7.0])
    U, Q = _make_scene_3d(R, t)

    poses = ippe.mat_run(U, Q)

    assert poses["reprojError1"] < 1e-9
    assert np.allclose(poses["R1"], R, atol=1e-6)
    assert np.allclose(poses["t1"].reshape(3), t, atol=1e-6)


def test_returns_two_distinct_solutions():
    R = _rotation_from_axis_angle([1.0, 0.0, 0.0], 0.6)
    t = np.array([0.0, 0.0, 5.0])
    U, Q = _make_scene(R, t)

    poses = ippe.mat_run(U, Q)

    for key in ("R1", "t1", "R2", "t2", "reprojError1", "reprojError2"):
        assert key in poses
    assert poses["R1"].shape == (3, 3)


def test_does_not_mutate_input():
    R = _rotation_from_axis_angle([0.0, 1.0, 0.0], 0.3)
    t = np.array([0.0, 0.0, 4.0])
    U, Q = _make_scene(R, t)
    U_before = U.copy()

    ippe.mat_run(U, Q)

    assert np.array_equal(U, U_before)


def test_rejects_unknown_homography_method():
    U, Q = _make_scene(np.eye(3), np.array([0.0, 0.0, 5.0]))
    with pytest.raises(ValueError):
        ippe.mat_run(U, Q, hEstMethod="nonsense")
