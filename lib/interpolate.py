"""Conservative regridding via xESMF with weight-file caching.

Uses xESMF (the xarray wrapper for ESMF), available in the stats conda
environment.  The caching strategy mirrors stats/lib/regridding.py
(RegridManager): weight files are stored on disk, keyed by a hash of the
source and destination tile geometry, so repeated runs on the same grids skip
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

Memory strategy — optional tiled regridding
-------------------------------------------
The default ``regrid()`` call uses a single-pass weight matrix over the full
source and destination grids.  For very high-resolution sources (e.g. native
EMODnet at 1/480° ≈ 230 m) a padded North Sea domain contains ~47 M source
cells.  Pass ``tile_cells > 0`` to enable tiled regridding: the destination
grid is split into tiles of *tile_cells* × *tile_cells* grid cells; for each
tile the source is subsetted to the tile's bounding box plus a *tile_buf_deg*
margin.  At 50-cell tiles and 0.5° buffer a tile at 0.05° covers a ~3.5°×3.5°
source area (≈ 2.8 M cells) and needs only ~50 MB for the weight matrix and
data arrays.  Weight files are cached per tile so subsequent runs skip the
computation.
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
    tile_cells: int = 0,
    tile_buf_deg: float = 0.5,
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
    tile_cells : int
        Maximum destination grid cells per tile in each dimension.
        ``0`` (default) — single-pass regridding (no tiling).
        Positive value — tiled regridding: splits the destination grid into
        tiles of at most *tile_cells* × *tile_cells* cells, subsetting the
        source per tile.  Use when the full source dataset is too large to
        fit in memory.
    tile_buf_deg : float
        Source-side buffer (degrees) added around each destination tile when
        tiling is enabled.  Prevents edge artefacts.

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
    # Single-pass or tiled conservative regridding.
    # ------------------------------------------------------------------
    if tile_cells > 0:
        depth_out, wetfrac_out = _regrid_tiled(
            xe, src, dst_grid, cache_dir,
            tile_cells=tile_cells,
            buf_deg=tile_buf_deg,
        )
    else:
        depth_out, wetfrac_out = _regrid_single(xe, src, dst_grid, cache_dir)

    # ------------------------------------------------------------------
    # Post-process assembled arrays
    # ------------------------------------------------------------------
    ocean_mask = wetfrac_out > 0.0

    # Force cells with insufficient ocean coverage to land.
    # These are typically marginal coastal/tidal cells where the conservative
    # average picked up a tiny sliver of ocean.  Removing them before basin
    # detection and strait analysis avoids large numbers of spurious flags.
    if min_wet_fraction > 0.0:
        ocean_mask = ocean_mask & (wetfrac_out >= min_wet_fraction)

    depth_out = np.where(ocean_mask, depth_out, 0.0)

    # Replace NaN depths that slipped through (unmapped cells)
    depth_out = np.where(np.isnan(depth_out), 0.0, depth_out)

    # Clamp shallow cells
    if min_depth > 0.0:
        depth_out = np.where(ocean_mask, np.maximum(depth_out, min_depth), depth_out)

    mask_out = ocean_mask.astype(np.int8)
    n_dropped = int((wetfrac_out > 0.0).sum()) - int(ocean_mask.sum())
    if min_wet_fraction > 0.0 and n_dropped:
        print(f"  min_wet_fraction={min_wet_fraction}: {n_dropped} marginal cells forced to land")

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
            "min_wet_fraction": min_wet_fraction,
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

def _dst_resolution(dst_grid: BaseGrid) -> float:
    """Estimate the destination grid cell size in degrees (smaller of lon/lat)."""
    try:
        return min(float(dst_grid.dlon), float(dst_grid.dlat))  # type: ignore[attr-defined]
    except AttributeError:
        pass
    # General case: central-difference estimate from centre-coordinate arrays
    clon = dst_grid.center_lon
    clat = dst_grid.center_lat
    mid_j = clon.shape[0] // 2
    mid_i = clon.shape[1] // 2
    d_lon = float(abs(np.diff(clon[mid_j, :]).mean()))
    d_lat = float(abs(np.diff(clat[:, mid_i]).mean()))
    return min(d_lon, d_lat)


def _tile_cache_key(src_ds: xr.Dataset, dst_ds: xr.Dataset) -> str:
    """MD5 hash uniquely identifying a (source tile, destination tile) pair."""
    info = (
        f"src={src_ds.sizes['lat']}x{src_ds.sizes['lon']}"
        f"_slat={float(src_ds.lat.min()):.5f}_{float(src_ds.lat.max()):.5f}"
        f"_slon={float(src_ds.lon.min()):.5f}_{float(src_ds.lon.max()):.5f}"
        f"_dst={dst_ds.sizes['y']}x{dst_ds.sizes['x']}"
        f"_dlat={float(dst_ds.lat.min()):.5f}_{float(dst_ds.lat.max()):.5f}"
        f"_dlon={float(dst_ds.lon.min()):.5f}_{float(dst_ds.lon.max()):.5f}"
    )
    return hashlib.md5(info.encode()).hexdigest()[:12]


def _get_tile_regridder(xe, src_ds: xr.Dataset, dst_ds: xr.Dataset, cache_dir: str):
    """Return a conservative xe.Regridder, loading cached weights when available.

    Always passes *filename* to the constructor so xESMF writes the weights on
    first use and reads them on subsequent calls.  ``reuse_weights`` is set from
    whether the file already exists — this is the canonical xESMF cache pattern.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _tile_cache_key(src_ds, dst_ds)
    weight_file = Path(cache_dir) / f"weights_conservative_{key}.nc"
    cached = weight_file.exists()
    if cached:
        print(f"    (weights cached: {weight_file.name})", end=" ", flush=True)
    return xe.Regridder(
        src_ds, dst_ds, "conservative",
        filename=str(weight_file),
        reuse_weights=cached,
        unmapped_to_nan=True,
    )


def _regrid_single(
    xe,
    src: xr.Dataset,
    dst_grid: BaseGrid,
    cache_dir: str,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Single-pass conservative regridding (no tiling).

    Builds one weight matrix that covers the entire source and destination
    grids.  Suitable when the source dataset fits comfortably in memory.
    """
    ny, nx = dst_grid.ny, dst_grid.nx

    # Source DS with explicit 1-D bounds
    src_dlon = float(np.diff(src.lon.values[:2]).mean())
    src_dlat = float(np.diff(src.lat.values[:2]).mean())
    src_lon_b = np.append(src.lon.values - src_dlon / 2,
                          src.lon.values[-1] + src_dlon / 2)
    src_lat_b = np.append(src.lat.values - src_dlat / 2,
                          src.lat.values[-1] + src_dlat / 2)
    src_ds = xr.Dataset({
        "lat":   ("lat",   src.lat.values),
        "lon":   ("lon",   src.lon.values),
        "lat_b": ("lat_b", src_lat_b),
        "lon_b": ("lon_b", src_lon_b),
    })

    # Destination DS with explicit 2-D corner bounds
    dst_ds = xr.Dataset({
        "lat":   (["y",   "x"  ], dst_grid.center_lat),
        "lon":   (["y",   "x"  ], dst_grid.center_lon),
        "lat_b": (["y_b", "x_b"], dst_grid.corner_lat),
        "lon_b": (["y_b", "x_b"], dst_grid.corner_lon),
    })

    n_src = src.sizes["lat"] * src.sizes["lon"]
    print(f"  Single-pass regridding ({n_src / 1e6:.1f} M source cells) …")

    regridder = _get_tile_regridder(xe, src_ds, dst_ds, cache_dir)

    depth_src = xr.DataArray(
        np.where(src["land"].values, np.nan, src["depth"].values.astype(np.float64)),
        dims=["lat", "lon"],
        coords={"lat": src.lat.values, "lon": src.lon.values},
    )
    ocean_src = xr.DataArray(
        (~src["land"].values).astype(np.float64),
        dims=["lat", "lon"],
        coords={"lat": src.lat.values, "lon": src.lon.values},
    )

    depth_out   = np.asarray(regridder(depth_src)).reshape(ny, nx)
    wetfrac_out = np.asarray(regridder(ocean_src)).reshape(ny, nx)

    return depth_out, wetfrac_out


def _regrid_tiled(
    xe,
    src: xr.Dataset,
    dst_grid: BaseGrid,
    cache_dir: str,
    tile_cells: int = 50,
    buf_deg: float = 0.5,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Tile-by-tile conservative regridding.

    Splits the destination grid into rectangular tiles of at most
    *tile_cells* × *tile_cells* grid cells.  For each tile the source
    dataset is subsetted to the tile bounding box plus *buf_deg* on all
    sides — only locally relevant source cells are loaded into the weight
    computation.

    Returns ``(depth_out, wetfrac_out)`` as 2-D arrays shaped ``[ny, nx]``.
    """
    ny, nx = dst_grid.ny, dst_grid.nx
    n_tlon = max(1, int(np.ceil(nx / tile_cells)))
    n_tlat = max(1, int(np.ceil(ny / tile_cells)))
    n_total = n_tlon * n_tlat

    src_dlon = float(abs(np.diff(src.lon.values[:2]).mean()))
    dst_res  = _dst_resolution(dst_grid)

    if n_total == 1:
        n_src = int((tile_cells * dst_res + 2 * buf_deg) / src_dlon) ** 2
        print(f"  Single tile  (≈ {n_src / 1e6:.1f} M source cells) …")
    else:
        n_src = int((tile_cells * dst_res + 2 * buf_deg) / src_dlon) ** 2
        print(
            f"  Tiled regridding: {n_tlon}×{n_tlat} destination tiles "
            f"(≈ {n_src / 1e6:.1f} M source cells per tile) …"
        )

    depth_out   = np.full((ny, nx), np.nan)
    wetfrac_out = np.zeros((ny, nx))

    # Pre-convert full source arrays (subsetting is done per tile via index arrays)
    depth_src_full = np.where(src["land"].values, np.nan, src["depth"].values).astype(np.float64)
    ocean_src_full = (~src["land"].values).astype(np.float64)
    src_lon = src.lon.values
    src_lat = src.lat.values

    for jt in range(n_tlat):
        j0 = jt * tile_cells
        j1 = min(j0 + tile_cells, ny)

        for it in range(n_tlon):
            i0 = it * tile_cells
            i1 = min(i0 + tile_cells, nx)
            k  = jt * n_tlon + it + 1

            if n_total > 1:
                print(f"    Tile {k}/{n_total} …", end=" ", flush=True)

            # ---- Destination tile (2-D centre + corner coords) ----
            tile_center_lon = dst_grid.center_lon[j0:j1, i0:i1]
            tile_center_lat = dst_grid.center_lat[j0:j1, i0:i1]
            tile_corner_lon = dst_grid.corner_lon[j0:j1 + 1, i0:i1 + 1]
            tile_corner_lat = dst_grid.corner_lat[j0:j1 + 1, i0:i1 + 1]

            tile_dst_ds = xr.Dataset({
                "lat":   (["y",   "x"  ], tile_center_lat),
                "lon":   (["y",   "x"  ], tile_center_lon),
                "lat_b": (["y_b", "x_b"], tile_corner_lat),
                "lon_b": (["y_b", "x_b"], tile_corner_lon),
            })

            # ---- Source subset (bbox of corner coords + buffer) ----
            slon_min = float(tile_corner_lon.min()) - buf_deg
            slon_max = float(tile_corner_lon.max()) + buf_deg
            slat_min = float(tile_corner_lat.min()) - buf_deg
            slat_max = float(tile_corner_lat.max()) + buf_deg

            lon_idx = np.where((src_lon >= slon_min) & (src_lon <= slon_max))[0]
            lat_idx = np.where((src_lat >= slat_min) & (src_lat <= slat_max))[0]

            if len(lon_idx) < 2 or len(lat_idx) < 2:
                if n_total > 1:
                    print("(no source data)")
                continue

            tile_lon = src_lon[lon_idx]
            tile_lat = src_lat[lat_idx]
            tile_depth_data = depth_src_full[np.ix_(lat_idx, lon_idx)]
            tile_ocean_data = ocean_src_full[np.ix_(lat_idx, lon_idx)]

            # xESMF source DS with explicit 1-D bounds
            tile_dlon = float(np.diff(tile_lon).mean())
            tile_dlat = float(np.diff(tile_lat).mean())
            tile_lon_b = np.append(tile_lon - tile_dlon / 2, tile_lon[-1] + tile_dlon / 2)
            tile_lat_b = np.append(tile_lat - tile_dlat / 2, tile_lat[-1] + tile_dlat / 2)

            tile_src_ds = xr.Dataset({
                "lat":   ("lat",   tile_lat),
                "lon":   ("lon",   tile_lon),
                "lat_b": ("lat_b", tile_lat_b),
                "lon_b": ("lon_b", tile_lon_b),
            })

            # ---- Regrid tile ----
            regridder = _get_tile_regridder(xe, tile_src_ds, tile_dst_ds, cache_dir)

            depth_da = xr.DataArray(
                tile_depth_data, dims=["lat", "lon"],
                coords={"lat": tile_lat, "lon": tile_lon},
            )
            ocean_da = xr.DataArray(
                tile_ocean_data, dims=["lat", "lon"],
                coords={"lat": tile_lat, "lon": tile_lon},
            )

            tile_d = np.asarray(regridder(depth_da)).reshape(j1 - j0, i1 - i0)
            tile_w = np.asarray(regridder(ocean_da)).reshape(j1 - j0, i1 - i0)

            depth_out[j0:j1, i0:i1]   = tile_d
            wetfrac_out[j0:j1, i0:i1] = tile_w

            if n_total > 1:
                n_wet = int(np.sum(tile_w > 0))
                print(f"({n_wet} wet cells)")

    return depth_out, wetfrac_out
