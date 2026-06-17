# ippe-py

Python implementation of **Infinitesimal Plane-based Pose Estimation (IPPE)** — a
fast, accurate analytical solution for the pose of a planar object from a single
image.

This is a port of Toby Collins' original MATLAB implementation,
[tobycollins/IPPE](https://github.com/tobycollins/IPPE).

## Installation

```bash
pip install ippe
```

The core solver depends only on NumPy. The convenience wrapper `ippe.run()`,
which undistorts image points first, additionally needs OpenCV:

```bash
pip install "ippe[opencv]"
```

## Usage

`mat_run` takes model points on the `z = 0` plane and **normalised** image
points (intrinsics and distortion already removed) and returns the two
candidate poses, ordered by reprojection error:

```python
import numpy as np
import ippe

# 2xN model points on the plane z=0 (world coordinates)
U = np.array([[-1.0, 1.0, 1.0, -1.0],
              [-1.0, -1.0, 1.0, 1.0]])

# 2xN normalised image points (pinhole projection of the object)
Q = ...

poses = ippe.mat_run(U, Q)

R, t = poses["R1"], poses["t1"]          # best solution
print(poses["reprojError1"], poses["reprojError2"])
```

The returned dictionary contains:

| Key            | Description                                  |
| -------------- | -------------------------------------------- |
| `R1`, `t1`     | First pose solution (lower reprojection error) |
| `R2`, `t2`     | Second (ambiguous) pose solution             |
| `reprojError1` | Reprojection error of the first solution     |
| `reprojError2` | Reprojection error of the second solution    |

If you have raw pixel coordinates and a calibrated camera, use `ippe.run()` to
undistort them first (requires OpenCV):

```python
poses = ippe.run(U, Q_pixels, camera_matrix, dist_coeffs)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{collins2014infinitesimal,
  title   = {Infinitesimal Plane-Based Pose Estimation},
  author  = {Collins, Toby and Bartoli, Adrien},
  journal = {International Journal of Computer Vision},
  volume  = {109},
  number  = {3},
  pages   = {252--286},
  year    = {2014},
  doi     = {10.1007/s11263-014-0725-5}
}
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
