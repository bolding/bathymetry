"""Conservative regridding via xESMF with weight-file caching.

Uses xESMF (the xarray wrapper for ESMF), available in the stats conda
environment.  The caching strategy mirrors stats/lib/regridding.py
(RegridManager): weight files are stored on disk, keyed by a hash of the
source and destination grid geometry, so repeated runs on the same grids skip
the expensive weight-computation step.

Two fields are produced from a single regridder:
- depth      – area-weighted mean over ocean source cells (source NaN on land
               → FRACAREA normalisation)
- wet_fraction – fraction of each destination cell covered by ocean source
               cells (source 1/0 no NaN → DSTAREA normalisation)

Both fields share exactly the same conservative weights because xESMF applies
NaN handling at field-application time, not at weight-computation time.

Cell-corner bounds are always passed explicitly so that rotated and Cartesian
destination grids are handled correctly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from grid import BaseGrid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def regrid(
    src: xr.Dataset,
    dst_grid: BaseGrid,
    min_depth: float = 0.0,
    min_wet_fraction: float = 0.0,
    cache_dir: str = "./regrid_weights",
) -> xr.Dataset:
    """Regrid fine-resolution bathymetry onto *dst_grid* conservatively.

    Parameters
    ----------
    src : xr.Dataset
        Source bathymetry as returned by :func:`reader.read_source`.
        Must have ``depth`` and ``land`` on a regular lon/lat grid.
    dst_grid : BaseGrid
        Target grid with corner and centre coordinates.
    min_depth : float
        Minimum ocean depth after regridding (m).
    min_wet_fraction : float
        Coarse cells with wet_fraction below this threshold are flagged in
        the summary but not forced to land here.
    cache_dir : str
        Directory for cached xESMF weight files.

    Returns
    -------
    xr.Dataset
        Variables: ``depth``, ``wet_fraction``, ``mask`` on the destination
        grid, with 2-D ``lon`` / ``lat`` centre-coordinate arrays.
    """
    # ------------------------------------------------------------------
    # Environment: must be set before xESMF / esmpy are imported.
    # UCX on some HPC systems aborts MPI init when it reads invalid network
    # config (e.g. LAT=1.8us).  Locking UCX to shared-memory transport
    # avoids the RDMA/InfiniBand code path that reads those variables.
    # ------------------------------------------------------------------
    os.environ.setdefault("UCX_TLS", "self,sm")
    os.environ.setdefault("UCX_POSIX_USE_PROC_LINK", "no")

    import xesmf as xe  # noqa: PLC0415

    # ------------------------------------------------------------------
    # Build xESMF-compatible source and destination Datasets with bounds
    # ------------------------------------------------------------------
    src_ds = _source_ds(src)
    dst_ds = _dst_ds(dst_grid)

    # ------------------------------------------------------------------
    # Retrieve or compute conservative regrid weights (cached on disk)
    # ------------------------------------------------------------------
    regridder = _get_regridder(xe, src_ds, dst_ds, dst_grid, cache_dir)

    # ------------------------------------------------------------------
    # Apply regridder to depth (NaN on land → FRACAREA normalisation)
    # ------------------------------------------------------------------
    depth_src = xr.DataArray(
        np.where(src["land"].values, np.nan, src["depth"].values),
        dims=["lat", "lon"],
        coords={"lat": src.lat, "lon": src.lon},
    )
    depth_out_da = regridder(depth_src)

    # ------------------------------------------------------------------
    # Apply same regridder to binary ocean field (1=ocean, 0=land, no NaN)
    # → DSTAREA normalisation = fraction of destination cell that is ocean
    # ------------------------------------------------------------------
    ocean_src = xr.DataArray(
        (~src["land"].values).astype(np.float64),
        dims=["lat", "lon"],
        coords={"lat": src.lat, "lon": src.lon},
    )
    wetfrac_out_da = regridder(ocean_src)

    # ------------------------------------------------------------------
    # Extract 2-D arrays and post-process
    # ------------------------------------------------------------------
    depth_out = np.asarray(depth_out_da).reshape(dst_grid.ny, dst_grid.nx)
    wetfrac_out = np.asarray(wetfrac_out_da).reshape(dst_grid.ny, dst_grid.nx)

    # Cells with no ocean coverage are land
    ocean_mask = wetfrac_out > 0.0
    depth_out = np.where(ocean_mask, depth_out, 0.0)

    # Replace NaN depths that slipped through (unmapped cells)
    depth_out = np.where(np.isnan(depth_out), 0.0, depth_out)

    # Clamp shallow cells
    if min_depth > 0.0:
        depth_out = np.where(ocean_mask, np.maximum(depth_out, min_depth), depth_out)

    mask_out = ocean_mask.astype(np.int8)

    return xr.Dataset(
        {
            "depth": (["lat", "lon"], np.where(ocean_mask, depth_out, np.nan)),
            "wet_fraction": (["lat", "lon"], wetfrac_out),
            "mask": (["lat", "lon"], mask_out),
        },
        coords={
            "lon": (["lat", "lon"], dst_grid.center_lon),
            "lat": (["lat", "lon"], dst_grid.center_lat),
        },
        attrs={
            "min_depth": min_depth,
            "min_wet_fraction_flag": min_wet_fraction,
        },
    )


def compute_cgrid_depth(
    depth_t: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Derive Arakawa C-grid face depths from T-point depths.

    Both outputs have shape ``[ny, nx]`` — the same as ``depth_t``.

    ``depth_u[i, j]`` — depth at the **eastern** face of T-cell (i, j):

    .. code-block:: python

        depth_u[:, :-1] = minimum(depth_t[:, :-1], depth_t[:, 1:])
        depth_u[:, -1]  = depth_t[:, -1]   # eastern boundary

    ``depth_v[i, j]`` — depth at the **northern** face of T-cell (i, j):

    .. code-block:: python

        depth_v[:-1, :] = minimum(depth_t[:-1, :], depth_t[1:, :])
        depth_v[-1, :]  = depth_t[-1, :]   # northern boundary

    The minimum convention ensures zero transport across a face that borders
    land on either side.  NaN propagates: land (NaN) on either adjacent T-cell
    makes the face NaN (land) too.
    """
    depth_u = depth_t.copy()
    depth_u[:, :-1] = np.minimum(depth_t[:, :-1], depth_t[:, 1:])

    depth_v = depth_t.copy()
    depth_v[:-1, :] = np.minimum(depth_t[:-1, :], depth_t[1:, :])

    return depth_u, depth_v


def regrid_summary(dst: xr.Dataset, dst_grid: BaseGrid) -> dict:
    """Return a summary dict for the regridded destination dataset."""
    depth = dst["depth"].values
    mask = dst["mask"].values.astype(bool)
    return {
        "nx × ny": f"{dst_grid.nx} × {dst_grid.ny}",
        "total cells": dst_grid.nx * dst_grid.ny,
        "ocean cells": int(mask.sum()),
        "land cells": int((~mask).sum()),
        "min depth (m)": f"{float(np.nanmin(depth)):.1f}" if mask.any() else "n/a",
        "max depth (m)": f"{float(np.nanmax(depth)):.1f}" if mask.any() else "n/a",
        "mean depth (m)": f"{float(np.nanmean(depth)):.1f}" if mask.any() else "n/a",
        "min wet_fraction": (
            f"{float(dst['wet_fraction'].values[mask].min()):.3f}"
            if mask.any() else "n/a"
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_ds(src: xr.Dataset) -> xr.Dataset:
    """Build xESMF-compatible Dataset from the source (1-D regular grid)."""
    lon = src.lon.values
    lat = src.lat.values
    dlon = float(np.diff(lon).mean())
    dlat = float(np.diff(lat).mean())
    lon_b = np.append(lon - dlon / 2.0, lon[-1] + dlon / 2.0)
    lat_b = np.append(lat - dlat / 2.0, lat[-1] + dlat / 2.0)
    return xr.Dataset(
        {
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "lat_b": ("lat_b", lat_b),
            "lon_b": ("lon_b", lon_b),
        }
    )


def _dst_ds(dst_grid: BaseGrid) -> xr.Dataset:
    """Build xESMF-compatible Dataset from BaseGrid (always 2-D bounds).

    Using 2-D corner arrays works for regular, rotated, and Cartesian grids
    without any special-casing.
    """
    return xr.Dataset(
        {
            "lat": (["y", "x"], dst_grid.center_lat),
            "lon": (["y", "x"], dst_grid.center_lon),
            "lat_b": (["y_b", "x_b"], dst_grid.corner_lat),
            "lon_b": (["y_b", "x_b"], dst_grid.corner_lon),
        }
    )


def _cache_key(src_ds: xr.Dataset, dst_grid: BaseGrid) -> str:
    """MD5 hash of source shape + extents + destination grid identity."""
    info = (
        f"src={src_ds.sizes['lat']}x{src_ds.sizes['lon']}"
        f"_slat={float(src_ds.lat.min()):.4f}_{float(src_ds.lat.max()):.4f}"
        f"_slon={float(src_ds.lon.min()):.4f}_{float(src_ds.lon.max()):.4f}"
        f"_dst={type(dst_grid).__name__}_{dst_grid.ny}x{dst_grid.nx}"
        f"_dlat={dst_grid.lat_bounds[0]:.4f}_{dst_grid.lat_bounds[1]:.4f}"
        f"_dlon={dst_grid.lon_bounds[0]:.4f}_{dst_grid.lon_bounds[1]:.4f}"
    )
    return hashlib.md5(info.encode()).hexdigest()[:12]


def _get_regridder(xe, src_ds: xr.Dataset, dst_ds: xr.Dataset,
                   dst_grid: BaseGrid, cache_dir: str):
    """Return a conservative xe.Regridder, loading cached weights if available.

    Mirrors the caching logic in stats/lib/regridding.py::RegridManager.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _cache_key(src_ds, dst_grid)
    weight_file = Path(cache_dir) / f"weights_conservative_{key}.nc"

    if weight_file.exists():
        print(f"  Loading cached weights: {weight_file.name}")
        regridder = xe.Regridder(
            src_ds, dst_ds, "conservative",
            reuse_weights=True,
            filename=str(weight_file),
            unmapped_to_nan=True,
        )
    else:
        print("  Computing conservative weights (first run only) …")
        regridder = xe.Regridder(src_ds, dst_ds, "conservative", unmapped_to_nan=True)
        regridder.to_netcdf(str(weight_file))
        print(f"  Weights saved: {weight_file.name}")

    return regridder
