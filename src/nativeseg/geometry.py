"""Canonical RAS native-slice CT & NIfTI geometry utilities."""

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@dataclass
class Volume:
    data: np.ndarray
    affine: np.ndarray
    original_affine: np.ndarray
    original_shape: tuple
    to_canonical: np.ndarray
    to_original: np.ndarray
    header: nib.Nifti1Header

    @property
    def spacing(self) -> tuple:
        return tuple(
            float(x) * self.unit_scale_mm for x in nib.affines.voxel_sizes(self.affine)
        )

    @property
    def unit_scale_mm(self) -> float:
        return {
            "unknown": 1.0,
            "mm": 1.0,
            "meter": 1000.0,
            "micron": 0.001,
        }[self.header.get_xyzt_units()[0]]

    @property
    def affine_mm(self) -> np.ndarray:
        affine = self.affine.copy()
        affine[:3] *= self.unit_scale_mm
        return affine

    def restore(self, canonical_array: np.ndarray) -> np.ndarray:
        """Restore a canonical RAS (X, Y, Z) array back to the original NIfTI voxel layout."""
        if canonical_array.shape != self.data.shape:
            raise ValueError("Prediction does not match the full canonical image grid")
        return np.ascontiguousarray(apply_orientation(canonical_array, self.to_original))

    def save_prediction(self, canonical_array: np.ndarray, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = self.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(1, 0)
        img = nib.Nifti1Image(
            self.restore(canonical_array).astype(np.uint8),
            self.original_affine,
            header=header,
        )
        nib.save(img, path)


def affine_mm(image):
    affine = image.affine.copy()
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.array_equal(affine[3], (0, 0, 0, 1))
        or np.linalg.matrix_rank(affine[:3, :3]) != 3
    ):
        raise ValueError("Expected a finite nonsingular physical affine")
    scale = {
        "unknown": 1.0,
        "mm": 1.0,
        "meter": 1000.0,
        "micron": 0.001,
    }[image.header.get_xyzt_units()[0]]
    affine[:3] *= scale
    return affine


def grid_displacement_mm(image, volume: Volume) -> float:
    if not np.isfinite(image.affine).all() or not np.isfinite(volume.original_affine).all():
        raise ValueError("Nonfinite physical affine")
    scale = {
        "unknown": 1.0,
        "mm": 1.0,
        "meter": 1000.0,
        "micron": 0.001,
    }[image.header.get_xyzt_units()[0]]
    a = image.affine.copy()
    a[:3] *= scale
    b = volume.original_affine.copy()
    b[:3] *= volume.unit_scale_mm
    corners = np.array(
        [(*point, 1) for point in product(*[(0, n - 1) for n in volume.original_shape])]
    )
    displacement = float(np.linalg.norm((corners @ (a - b).T)[:, :3], axis=1).max())
    if not np.isfinite(displacement):
        raise ValueError("Nonfinite physical displacement")
    return displacement


def load_native_image(path) -> Volume:
    """Load a 3D axial CT NIfTI volume into canonical RAS orientation without resampling."""
    image = nib.load(path)
    if len(image.shape) != 3 or min(image.shape) < 1:
        raise ValueError("Expected a nonempty 3D container of 2D slices")
    affine_mm(image)
    spacing = nib.affines.voxel_sizes(image.affine)
    directions = image.affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError("Expected near-orthogonal CT; shear is unsupported")
    orientation = io_orientation(image.affine)
    ras = axcodes2ornt(("R", "A", "S"))
    forward = ornt_transform(orientation, ras)
    reverse = ornt_transform(ras, orientation)
    if forward[2, 0] != 2:
        raise ValueError("RAS reorientation would change the source axial slice plane")
    original = image.get_fdata(dtype=np.float32)
    if not np.isfinite(original).all():
        raise ValueError("Nonfinite CT values")
    canonical = np.ascontiguousarray(apply_orientation(original, forward))
    return Volume(
        canonical,
        image.affine @ inv_ornt_aff(forward, original.shape),
        image.affine.copy(),
        original.shape,
        forward,
        reverse,
        image.header.copy(),
    )


def nearest_identity(image, volume: Volume):
    if image.shape != volume.original_shape:
        raise ValueError("Different grid shapes cannot have identity index mapping")
    source = affine_mm(image)
    target = volume.original_affine.copy()
    target[:3] *= volume.unit_scale_mm
    corners = np.array(
        [(*point, 1.0) for point in product(*[(0, n - 1) for n in volume.original_shape])]
    )
    forward = np.linalg.solve(source, target) - np.eye(4)
    reverse = np.linalg.solve(target, source) - np.eye(4)
    bounds = [
        np.abs((corners @ matrix.T)[:, :3]).max(axis=0) for matrix in (forward, reverse)
    ]
    if any(not np.isfinite(b).all() or np.max(b) >= 0.5 - 1e-6 for b in bounds):
        raise ValueError("Nearest-neighbor bidirectional index identity is not proven")
    return {
        "bidirectional_identity": True,
        "forward_max_index_shift": float(bounds[0].max()),
        "reverse_max_index_shift": float(bounds[1].max()),
        "physical_corner_displacement_mm": grid_displacement_mm(image, volume),
    }


def load_native_mask(path, volume: Volume) -> np.ndarray:
    """Load a binary NIfTI mask verified to match `volume`'s voxel grid in canonical RAS."""
    image = nib.load(path)
    nearest_identity(image, volume)
    values = image.get_fdata(dtype=np.float32)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Expected a finite binary mask")
    return np.ascontiguousarray(apply_orientation(values.astype(bool), volume.to_canonical))
