"""
TriContext dataset — RECIST-guided three-center patch sampling for 3D CT pretraining.

Three spatially distinct 50×50×50 patches per lesion (no segmentation mask required):

  1. Tumor center        — RECIST midpoint (coordX, coordY, coordZ)
  2. Peritumoral center  — just outside lesion boundary along chosen direction
  3. Parenchymal center  — farther away, validated to lie in lung parenchyma

Direction selection:
  - Candidate directions: N evenly spaced angles in the axial plane
  - For each candidate: check (a) all patches within image volume,
    (b) parenchymal HU consistent with lung tissue (−950 to −300 HU)
  - Pick the direction with HU closest to −600 (lung parenchyma centre)
  - Falls back to next in-bounds direction if no lung-HU direction found

Expected CSV columns (same as SSLRadiomicsDataset):
  image_path, coordX, coordY, coordZ         — physical coords of tumor centre (mm)
  lesion_diameters_x, lesion_diameters_y     — RECIST diameters (mm)
  spacing_x, spacing_y, spacing_z            — voxel spacing (mm)
  [optional] bbox                            — "[x1,y1,x2,y2]" pixel bbox
  [optional] centroid_x, centroid_y          — pixel centroid of lesion
"""

import ast
import numpy as np
import SimpleITK as sitk
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from .utils import resample_image_to_spacing, slice_image

# HU range considered lung parenchyma
LUNG_HU_MIN    = -950
LUNG_HU_MAX    = -300
LUNG_HU_CENTER = -600


class TriContextSSLDataset(Dataset):
    """
    Returns three positive pairs per lesion:
        (tumor_v1,  tumor_v2)
        (peri_v1,   peri_v2)
        (par_v1,    par_v2)
    where each pair is two independently augmented views of the same patch.
    """

    def __init__(
        self,
        path,
        radius: int = 25,
        m1: float = 4.0,
        m2: float = 20.0,
        n_dir_candidates: int = 8,
        orient: bool = True,
        resample_spacing=None,
        transform=None,
    ):
        super().__init__()
        self._path = Path(path)
        self.radius = radius
        self.m1 = m1
        self.m2 = m2
        self.n_dir_candidates = n_dir_candidates
        self.orient = orient
        self.resample_spacing = resample_spacing
        self.transform = transform
        self.annotations = pd.read_csv(self._path)
        self._num_samples = len(self.annotations)

    def __len__(self):
        return self._num_samples

    # ── RECIST geometry ──────────────────────────────────────────────────────

    def _recist_long_axis(self, row) -> np.ndarray:
        """
        Unit vector of RECIST long axis in physical space (axial plane only).
        Uses the bounding box diagonal if available, otherwise falls back to
        the larger of lesion_diameters_x / lesion_diameters_y.
        """
        if "bbox" in row and not pd.isna(row.get("bbox", None)):
            try:
                x1, y1, x2, y2 = ast.literal_eval(str(row["bbox"]))
                sx = float(row.get("spacing_x", 1.0))
                sy = float(row.get("spacing_y", 1.0))
                dx = (x2 - x1) * sx
                dy = (y2 - y1) * sy
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 1e-6:
                    return np.array([dx / norm, dy / norm, 0.0])
            except Exception:
                pass

        # Fallback: axis of larger diameter
        dx = float(row.get("lesion_diameters_x", 1.0))
        dy = float(row.get("lesion_diameters_y", 1.0))
        return np.array([1.0, 0.0, 0.0]) if dx >= dy else np.array([0.0, 1.0, 0.0])

    def _candidate_directions(self, long_axis: np.ndarray) -> list:
        """
        Generate n_dir_candidates unit vectors in the axial plane.
        First candidate = perpendicular to long axis; subsequent candidates
        are rotated by 360/n steps.
        """
        # Perpendicular to long axis in axial plane
        perp = np.array([-long_axis[1], long_axis[0], 0.0])
        norm = np.linalg.norm(perp)
        if norm < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp /= norm

        base_angle = np.arctan2(perp[1], perp[0])
        directions = []
        for i in range(self.n_dir_candidates):
            angle = base_angle + 2 * np.pi * i / self.n_dir_candidates
            directions.append(np.array([np.cos(angle), np.sin(angle), 0.0]))
        return directions

    def _center_in_bounds(self, image, center_phys) -> bool:
        """True if a patch of size radius can be extracted around center_phys."""
        size = image.GetSize()  # (X, Y, Z)
        try:
            ci = image.TransformPhysicalPointToContinuousIndex(
                [float(c) for c in center_phys]
            )
            return all(
                self.radius <= ci[d] < size[d] - self.radius for d in range(3)
            )
        except Exception:
            return False

    def _sample_hu(self, image, center_phys) -> float:
        """Sample mean HU in a small 5×5×1 neighbourhood at center_phys."""
        try:
            ci = image.TransformPhysicalPointToContinuousIndex(
                [float(c) for c in center_phys]
            )
            x, y, z = [int(round(c)) for c in ci]
            size = image.GetSize()
            r = 2  # 5×5 neighbourhood
            xs = range(max(0, x-r), min(size[0], x+r+1))
            ys = range(max(0, y-r), min(size[1], y+r+1))
            z  = int(np.clip(z, 0, size[2]-1))
            vals = [image.GetPixel(int(xi), int(yi), z)
                    for xi in xs for yi in ys]
            return float(np.mean(vals))
        except Exception:
            return 0.0  # neutral score

    def _select_direction(self, image, ct, half_L, long_axis) -> np.ndarray:
        """
        From n_dir_candidates axial directions, return the one where:
          1. Both cperi and cpar are within image bounds
          2. Parenchymal HU is closest to LUNG_HU_CENTER (−600)
        Falls back to first in-bounds direction if none passes the HU check.
        """
        candidates = self._candidate_directions(long_axis)

        in_bounds_fallback = None
        best_dir   = None
        best_score = float("inf")

        for v in candidates:
            cperi = ct + (half_L + self.m1) * v
            cpar  = ct + (half_L + self.m2) * v

            if not (self._center_in_bounds(image, cperi) and
                    self._center_in_bounds(image, cpar)):
                continue

            if in_bounds_fallback is None:
                in_bounds_fallback = v

            hu = self._sample_hu(image, cpar)
            if LUNG_HU_MIN <= hu <= LUNG_HU_MAX:
                score = abs(hu - LUNG_HU_CENTER)
                if score < best_score:
                    best_score = score
                    best_dir = v

        # If no direction has lung-range HU, use first in-bounds direction
        if best_dir is None:
            best_dir = in_bounds_fallback

        # Last resort: perpendicular to long axis (original behaviour)
        if best_dir is None:
            best_dir = candidates[0]

        return best_dir

    # ── Patch extraction ─────────────────────────────────────────────────────

    def _extract_patch(self, image, center_phys):
        ci = image.TransformPhysicalPointToContinuousIndex(
            [float(c) for c in center_phys]
        )
        ci = [int(round(c)) for c in ci]
        patch_idx = [(c - self.radius, c + self.radius) for c in ci]
        patch = slice_image(image, patch_idx)
        patch = sitk.DICOMOrient(patch, "LPS")
        return sitk.GetArrayFromImage(patch)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_three_centers(self, row) -> tuple:
        """
        Compute (ct, cperi, cpar, direction) for a CSV row.
        Useful for visualisation without loading the image.
        Returns physical-space coordinates (mm).
        """
        ct = np.array([float(row["coordX"]),
                       float(row["coordY"]),
                       float(row["coordZ"])])
        L = max(float(row.get("lesion_diameters_x", 10.0)),
                float(row.get("lesion_diameters_y", 10.0)))
        long_axis = self._recist_long_axis(row)
        # Default direction (perpendicular to long axis) — no image needed
        perp = np.array([-long_axis[1], long_axis[0], 0.0])
        norm = np.linalg.norm(perp[:2])
        if norm > 1e-6:
            perp = perp / norm
        cperi = ct + (L / 2 + self.m1) * perp
        cpar  = ct + (L / 2 + self.m2) * perp
        return ct, cperi, cpar, long_axis, perp

    def get_three_centers_validated(self, image, row) -> tuple:
        """
        Like get_three_centers but selects direction using image HU validation.
        Returns (ct, cperi, cpar, long_axis, chosen_direction).
        """
        ct = np.array([float(row["coordX"]),
                       float(row["coordY"]),
                       float(row["coordZ"])])
        L = max(float(row.get("lesion_diameters_x", 10.0)),
                float(row.get("lesion_diameters_y", 10.0)))
        long_axis = self._recist_long_axis(row)
        v = self._select_direction(image, ct, L / 2, long_axis)
        cperi = ct + (L / 2 + self.m1) * v
        cpar  = ct + (L / 2 + self.m2) * v
        return ct, cperi, cpar, long_axis, v

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        image = sitk.ReadImage(str(row["image_path"]))
        if self.resample_spacing is not None:
            image = resample_image_to_spacing(image, self.resample_spacing, -1024)
        if self.orient:
            image = sitk.DICOMOrient(image, "LPI")

        ct, cperi, cpar, long_axis, _ = self.get_three_centers_validated(image, row)

        patches = []
        for center in [ct, cperi, cpar]:
            arr = self._extract_patch(image, center)
            pair = self.transform(arr) if self.transform is not None else (arr, arr)
            patches.append(pair)

        return patches
