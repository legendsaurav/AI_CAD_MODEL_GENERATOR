"""
tests/test_fitting.py — Tests for production primitive fitting algorithms.

Validates:
  - Levenberg-Marquardt convergence on synthetic point clouds
  - RANSAC robustness against outliers
  - Chamfer/Hausdorff distance metric correctness
  - Volume IoU computation
  - Primitive surface samplers
"""
import os
import sys
import math

import numpy as np
import pytest

# Ensure geometry-engine root is on path
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from primitive.fitting import (
    LevenbergMarquardtFitter,
    RANSACFitter,
    FitResult,
    _cylinder_residuals,
    _sphere_residuals,
    _plane_residuals,
)
from refinement.comparator import (
    chamfer_distance,
    hausdorff_distance,
    volume_iou,
    _sample_sphere,
    _sample_cylinder,
    _sample_box,
)


# ---------------------------------------------------------------------------
# Primitive residual tests
# ---------------------------------------------------------------------------

class TestResiduals:
    """Test analytic residual functions."""

    def test_sphere_residuals_on_surface(self):
        """Points exactly on sphere surface should have zero residuals."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 2.0
        # Generate points on sphere
        n = 100
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        points = np.stack([
            radius * sin_theta * np.cos(phi),
            radius * sin_theta * np.sin(phi),
            radius * cos_theta,
        ], axis=1)

        residuals = _sphere_residuals(points, center, radius)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_plane_residuals_on_surface(self):
        """Points on the plane should have zero residuals."""
        normal = np.array([0.0, 0.0, 1.0])
        distance = 5.0
        points = np.array([
            [1, 2, 5], [-3, 4, 5], [0, 0, 5], [10, -10, 5]
        ], dtype=float)
        residuals = _plane_residuals(points, normal, distance)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_cylinder_residuals_on_surface(self):
        """Points on cylinder surface should have near-zero residuals."""
        center = np.array([0, 0, 0], dtype=float)
        axis = np.array([0, 0, 1], dtype=float)
        radius = 3.0
        n = 100
        theta = np.linspace(0, 2 * np.pi, n)
        z = np.random.uniform(-5, 5, n)
        points = np.stack([
            radius * np.cos(theta),
            radius * np.sin(theta),
            z,
        ], axis=1)

        residuals = _cylinder_residuals(points, axis, center, radius)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Levenberg-Marquardt tests
# ---------------------------------------------------------------------------

class TestLevenbergMarquardt:
    """Test LM fitting on synthetic data."""

    def test_fit_sphere_converges(self):
        """LM should converge when given noisy sphere points."""
        np.random.seed(42)
        fitter = LevenbergMarquardtFitter(max_iterations=100, tolerance=1e-8)

        true_center = np.array([1.0, 2.0, 3.0])
        true_radius = 5.0

        # Generate noisy sphere points
        n = 200
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        noise = np.random.randn(n, 3) * 0.05

        points = true_center + true_radius * np.stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ], axis=1) + noise

        result = fitter.fit_sphere(points)
        assert result.converged
        assert abs(result.parameters["radius"] - true_radius) < 0.5
        assert result.residual_error < 0.1

    def test_fit_plane_converges(self):
        """Plane fitting via SVD should be exact for coplanar points."""
        fitter = LevenbergMarquardtFitter()

        # Points on z=3 plane
        n = 50
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.full(n, 3.0)
        points = np.stack([x, y, z], axis=1)

        result = fitter.fit_plane(points)
        assert result.converged
        # Normal should be approximately [0, 0, ±1]
        nz = abs(result.parameters["normal_z"])
        assert nz > 0.99

    def test_fit_cylinder_from_init(self):
        """LM should converge for cylinder with good initialization."""
        np.random.seed(123)
        fitter = LevenbergMarquardtFitter(max_iterations=200)

        true_radius = 2.0
        n = 300
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.uniform(-3, 3, n)
        noise = np.random.randn(n, 3) * 0.02

        points = np.stack([
            true_radius * np.cos(theta),
            true_radius * np.sin(theta),
            z,
        ], axis=1) + noise

        result = fitter.fit_cylinder(points, {
            "center_x": 0, "center_y": 0, "center_z": 0,
            "axis_x": 0, "axis_y": 0, "axis_z": 1,
            "radius": 1.5,  # deliberately wrong
        })
        assert abs(result.parameters["radius"] - true_radius) < 0.5


# ---------------------------------------------------------------------------
# RANSAC tests
# ---------------------------------------------------------------------------

class TestRANSAC:
    """Test RANSAC fitting with outliers."""

    def test_sphere_ransac_with_outliers(self):
        """RANSAC should fit sphere correctly despite 30% outliers."""
        np.random.seed(42)
        fitter = RANSACFitter(max_trials=500, residual_threshold=0.2)

        true_radius = 3.0
        n_inliers = 200
        n_outliers = 80

        # Generate sphere inliers
        phi = np.random.uniform(0, 2 * np.pi, n_inliers)
        cos_theta = np.random.uniform(-1, 1, n_inliers)
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        inliers = true_radius * np.stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ], axis=1) + np.random.randn(n_inliers, 3) * 0.05

        outliers = np.random.uniform(-10, 10, (n_outliers, 3))
        points = np.vstack([inliers, outliers])

        result = fitter.fit(points, "Sphere")
        assert result.inlier_ratio > 0.5
        assert abs(result.parameters["radius"] - true_radius) < 1.0


# ---------------------------------------------------------------------------
# Distance metric tests
# ---------------------------------------------------------------------------

class TestDistanceMetrics:
    """Test Chamfer, Hausdorff, and IoU computations."""

    def test_chamfer_identical_clouds(self):
        """Chamfer distance of identical clouds should be 0."""
        pts = np.random.randn(100, 3)
        cd = chamfer_distance(pts, pts.copy())
        assert cd < 1e-10

    def test_chamfer_offset_clouds(self):
        """Chamfer distance should scale with offset magnitude."""
        pts = np.random.randn(100, 3)
        cd_small = chamfer_distance(pts, pts + 0.1)
        cd_large = chamfer_distance(pts, pts + 1.0)
        assert cd_small < cd_large

    def test_hausdorff_identical(self):
        pts = np.random.randn(100, 3)
        hd = hausdorff_distance(pts, pts.copy())
        assert hd < 1e-10

    def test_volume_iou_identical(self):
        pts = np.random.randn(200, 3)
        iou = volume_iou(pts, pts.copy(), voxel_size=0.5)
        assert iou == 1.0

    def test_volume_iou_disjoint(self):
        a = np.random.randn(100, 3) + np.array([100, 0, 0])
        b = np.random.randn(100, 3) - np.array([100, 0, 0])
        iou = volume_iou(a, b, voxel_size=0.5)
        assert iou == 0.0


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------

class TestSamplers:
    def test_sphere_sampler_radius(self):
        pts = _sample_sphere({"radius": 5.0, "center_x": 0, "center_y": 0, "center_z": 0}, 1000)
        dists = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(dists, 5.0, atol=0.01)

    def test_cylinder_sampler_radius(self):
        pts = _sample_cylinder({"radius": 3.0, "height": 10, "center_x": 0, "center_y": 0, "center_z": 0}, 1000)
        xy_dists = np.linalg.norm(pts[:, :2], axis=1)
        np.testing.assert_allclose(xy_dists, 3.0, atol=0.01)

    def test_box_sampler_bounds(self):
        pts = _sample_box({"width": 4, "height": 4, "depth": 4, "center_x": 0, "center_y": 0, "center_z": 0}, 5000)
        assert pts[:, 0].max() <= 2.01
        assert pts[:, 0].min() >= -2.01
