"""Conservative regridding via xESMF with weight-file caching.

Uses xESMF (the xarray wrapper for ESMF), available in the stats conda
environment.  The caching strategy mirrors stats/lib/regridding.py
(RegridManager): weight files are stored on disk, keyed by a hash of the
source and destination tile geometry, so repeated runs on the same grids skip
the expensive weight-computation step.

Two fields are produced from a single regridder:
- depth      – area-weighted mean over ocean source cells only.
               Land cells are set to 0 in the source; after regridding both
               depth and the ocean-fraction field, depth is divided by the
               ocean fraction to recover the ocean-only average.  This avoids
               NaN propagation through the sparse weight multiply (NaN-for-land
               would contaminate any coarse cell that overlaps even one land
               fine cell, yielding the erroneous min_depth everywhere).
- wet_fraction – fraction of each destination cell covered by ocean source
               cells (source 1/0 → conservative weighted sum)

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
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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

    # Clamp shallow cells
    if min_depth > 0.0:
        depth_out = np.where(ocean_mask, np.maximum(depth_out, min_depth), depth_out)

    mask_out = ocean_mask.astype(np.int8)
    n_dropped = int((wetfrac_out > 0.0).sum()) - int(ocean_mask.sum())
    if min_wet_fraction > 0.0 and n_dropped:
        logger.info("  min_wet_fraction=%s: %d marginal cells forced to land",
                    min_wet_fraction, n_dropped)

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


def _tile_cache_key(src_ds: xr.Dataset, dst_ds: xr.Dataset) -> tuple[str, str]:
    """Return (hash, description) uniquely identifying a (source, destination) pair.

    The description is a human-readable plain-text summary of the inputs that
    produced the hash, written as a sidecar file alongside the weight NetCDF.
    """
    src_ny = src_ds.sizes['lat']
    src_nx = src_ds.sizes['lon']
    src_lat_min = float(src_ds.lat.min())
    src_lat_max = float(src_ds.lat.max())
    src_lon_min = float(src_ds.lon.min())
    src_lon_max = float(src_ds.lon.max())
    dst_ny = dst_ds.sizes['y']
    dst_nx = dst_ds.sizes['x']
    dst_lat_min = float(dst_ds.lat.min())
    dst_lat_max = float(dst_ds.lat.max())
    dst_lon_min = float(dst_ds.lon.min())
    dst_lon_max = float(dst_ds.lon.max())

    info = (
        f"src={src_ny}x{src_nx}"
        f"_slat={src_lat_min:.5f}_{src_lat_max:.5f}"
        f"_slon={src_lon_min:.5f}_{src_lon_max:.5f}"
        f"_dst={dst_ny}x{dst_nx}"
        f"_dlat={dst_lat_min:.5f}_{dst_lat_max:.5f}"
        f"_dlon={dst_lon_min:.5f}_{dst_lon_max:.5f}"
    )
    key = hashlib.md5(info.encode()).hexdigest()[:12]

    desc = (
        f"xESMF conservative weight file\n"
        f"hash: {key}\n"
        f"\n"
        f"source grid\n"
        f"  size : {src_ny} lat × {src_nx} lon\n"
        f"  lat  : {src_lat_min:.5f} – {src_lat_max:.5f}\n"
        f"  lon  : {src_lon_min:.5f} – {src_lon_max:.5f}\n"
        f"\n"
        f"destination grid\n"
        f"  size : {dst_ny} y × {dst_nx} x\n"
        f"  lat  : {dst_lat_min:.5f} – {dst_lat_max:.5f}\n"
        f"  lon  : {dst_lon_min:.5f} – {dst_lon_max:.5f}\n"
    )
    return key, desc


def _get_tile_regridder(xe, src_ds: xr.Dataset, dst_ds: xr.Dataset, cache_dir: str):
    """Return a conservative xe.Regridder, loading cached weights when available.

    Always passes *filename* to the constructor so xESMF writes the weights on
    first use and reads them on subsequent calls.  ``reuse_weights`` is set from
    whether the file already exists — this is the canonical xESMF cache pattern.
    A plain-text sidecar ``*.txt`` is written alongside each weight file listing
    the grid parameters that produced the hash.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key, desc = _tile_cache_key(src_ds, dst_ds)
    weight_file = Path(cache_dir) / f"weights_conservative_{key}.nc"
    cached = weight_file.exists()
    if cached:
        logger.debug("    (weights cached: %s)", weight_file.name)
    regridder = xe.Regridder(
        src_ds, dst_ds, "conservative",
        filename=str(weight_file),
        reuse_weights=cached,
        unmapped_to_nan=True,
    )
    if not cached:
        txt_file = weight_file.with_suffix(".txt")
        txt_file.write_text(desc)
    return regridder


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
    logger.info("  Single-pass regridding (%.1f M source cells) …", n_src / 1e6)

    regridder = _get_tile_regridder(xe, src_ds, dst_ds, cache_dir)

    # Land cells are set to 0 (not NaN) so NaN does not propagate through the
    # sparse weight matrix.  After regridding both fields we recover the
    # ocean-only area-weighted average by dividing by the wet fraction:
    #
    #   depth_num  = W @ depth_zeros  = sum_{ocean s}(W_{ds} * depth_s)
    #   wetfrac    = W @ ocean_flag   = sum_{ocean s}(W_{ds})
    #   depth_avg  = depth_num / wetfrac   ← area-weighted mean of ocean cells
    #
    # With FRACAREA weights (sum_s W_{ds} = 1) this is the correct exclusive
    # ocean average.  NaN-for-land would propagate through the sparse multiply
    # and make any coarse cell that overlaps even a single land fine cell NaN,
    # which is then replaced by 0 and clamped to min_depth — too shallow.
    depth_src = xr.DataArray(
        np.where(src["land"].values, 0.0, src["depth"].values.astype(np.float64)),
        dims=["lat", "lon"],
        coords={"lat": src.lat.values, "lon": src.lon.values},
    )
    ocean_src = xr.DataArray(
        (~src["land"].values).astype(np.float64),
        dims=["lat", "lon"],
        coords={"lat": src.lat.values, "lon": src.lon.values},
    )

    depth_num   = np.asarray(regridder(depth_src)).reshape(ny, nx)
    wetfrac_out = np.asarray(regridder(ocean_src)).reshape(ny, nx)

    # Ocean-only average; cells with no ocean source remain 0 / NaN-free
    depth_out = np.where(wetfrac_out > 0, depth_num / wetfrac_out, 0.0)

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

    n_src = int((tile_cells * dst_res + 2 * buf_deg) / src_dlon) ** 2
    if n_total == 1:
        logger.info("  Single tile  (≈ %.1f M source cells) …", n_src / 1e6)
    else:
        logger.info("  Tiled regridding: %d×%d destination tiles "
                    "(≈ %.1f M source cells per tile) …",
                    n_tlon, n_tlat, n_src / 1e6)

    depth_out   = np.full((ny, nx), np.nan)
    wetfrac_out = np.zeros((ny, nx))

    # Pre-convert full source arrays (subsetting is done per tile via index arrays).
    # Land cells are filled with 0 (not NaN) — see _regrid_single for the rationale.
    depth_src_full = np.where(src["land"].values, 0.0, src["depth"].values).astype(np.float64)
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
                logger.debug("    Tile %d/%d …", k, n_total)

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
                    logger.debug("      (no source data)")
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

            tile_num = np.asarray(regridder(depth_da)).reshape(j1 - j0, i1 - i0)
            tile_w   = np.asarray(regridder(ocean_da)).reshape(j1 - j0, i1 - i0)

            # Ocean-only average depth (divide by wet fraction)
            tile_d = np.where(tile_w > 0, tile_num / tile_w, 0.0)

            depth_out[j0:j1, i0:i1]   = tile_d
            wetfrac_out[j0:j1, i0:i1] = tile_w

            if n_total > 1:
                n_wet = int(np.sum(tile_w > 0))
                logger.debug("      (%d wet cells)", n_wet)

    return depth_out, wetfrac_out


# ---------------------------------------------------------------------------
# Bbox-restricted percentile post-pass
# ---------------------------------------------------------------------------

def apply_bbox_percentile(
    depth_out: npt.NDArray,
    wet_fraction: npt.NDArray,
    src: xr.Dataset,
    dst_grid: BaseGrid,
    bboxes: list[tuple[float, float, float, float]],
    percentile: float = 75,
    min_wet_fraction: float = 0.3,
) -> npt.NDArray:
    """Replace area-weighted mean depth with a percentile inside bbox regions.

    Only deepens cells — never makes a cell shallower than the conservative mean.
    Cells with ``wet_fraction`` below *min_wet_fraction* are left unchanged.

    The result is baked into the raw-regrid cache so ``--skip-regrid`` runs
    automatically use the adjusted depths.

    Parameters
    ----------
    depth_out : [ny, nx] array
        Conservative mean depths as returned by ``regrid()``.  NaN for land.
    wet_fraction : [ny, nx] array
        Ocean fraction per coarse cell.
    src : xr.Dataset
        Fine-resolution source (``depth``, ``land``, 1-D ``lat``/``lon``).
    dst_grid : BaseGrid
        Coarse destination grid with ``corner_lat``/``corner_lon`` [ny+1, nx+1].
    bboxes : list of (lon_min, lon_max, lat_min, lat_max)
        Geographic boxes where the percentile is applied.
    percentile : float
        Depth percentile to use (default 75 — biased toward deeper values).
    min_wet_fraction : float
        Minimum ocean fraction to apply the percentile (default 0.3).

    Returns
    -------
    [ny, nx] array — depth_out with percentile applied inside bboxes.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("apply_bbox_percentile: pandas not available — skipping")
        return depth_out

    if not bboxes:
        return depth_out

    ny, nx = dst_grid.ny, dst_grid.nx
    depth_out = depth_out.copy()

    # ---- Which coarse cells are inside any bbox? -------------------------
    c_lat = dst_grid.center_lat   # [ny, nx]
    c_lon = dst_grid.center_lon
    coarse_in_bbox = np.zeros((ny, nx), dtype=bool)
    for (lo0, lo1, la0, la1) in bboxes:
        coarse_in_bbox |= (
            (c_lon >= lo0) & (c_lon <= lo1) &
            (c_lat >= la0) & (c_lat <= la1)
        )

    n_target = int(coarse_in_bbox.sum())
    if n_target == 0:
        logger.info("  bbox_depth_percentile: no coarse cells inside any bbox — skipped")
        return depth_out

    # ---- Fine-grid arrays ------------------------------------------------
    src_lat = src.lat.values   # 1-D [n_fine_lat], increasing
    src_lon = src.lon.values   # 1-D [n_fine_lon], increasing
    depth_src = np.where(src["land"].values, np.nan,
                         src["depth"].values.astype(float))

    # ---- Coarse cell lat/lon edges (works for regular spherical grids) ---
    # corner_lat[:, 0] gives monotone lat edges; corner_lon[0, :] lon edges.
    lat_edges = dst_grid.corner_lat[:, 0]   # [ny+1]
    lon_edges = dst_grid.corner_lon[0, :]   # [nx+1]

    # ---- Restrict fine-grid to bbox union extent (+ 1 fine cell margin) -
    rows_b, cols_b = np.where(coarse_in_bbox)
    dlat_f = float(abs(np.diff(src_lat).mean()))
    dlon_f = float(abs(np.diff(src_lon).mean()))
    lat_min_b = float(lat_edges[rows_b.min()]) - dlat_f
    lat_max_b = float(lat_edges[rows_b.max() + 1]) + dlat_f
    lon_min_b = float(lon_edges[cols_b.min()]) - dlon_f
    lon_max_b = float(lon_edges[cols_b.max() + 1]) + dlon_f

    lat_mask = (src_lat >= lat_min_b) & (src_lat <= lat_max_b)
    lon_mask = (src_lon >= lon_min_b) & (src_lon <= lon_max_b)
    sub_lat   = src_lat[lat_mask]
    sub_lon   = src_lon[lon_mask]
    if sub_lat.size == 0 or sub_lon.size == 0:
        logger.info("  bbox_depth_percentile: no fine-grid data in bbox extent — skipped")
        return depth_out
    sub_depth = depth_src[np.ix_(lat_mask, lon_mask)]   # [n_sub_lat, n_sub_lon]

    # ---- Bin fine cells into coarse cells --------------------------------
    j_bins = np.digitize(sub_lat, lat_edges) - 1   # 0-based coarse row
    i_bins = np.digitize(sub_lon, lon_edges) - 1   # 0-based coarse col

    # Broadcast to [n_sub_lat, n_sub_lon]
    j_2d = np.broadcast_to(j_bins[:, None], sub_depth.shape)
    i_2d = np.broadcast_to(i_bins[None, :], sub_depth.shape)

    flat_j = j_2d.ravel()
    flat_i = i_2d.ravel()
    flat_d = sub_depth.ravel()

    # Safe index for coarse_in_bbox lookup (clamped to valid range)
    j_clip = np.clip(flat_j, 0, ny - 1)
    i_clip = np.clip(flat_i, 0, nx - 1)

    valid = (
        np.isfinite(flat_d) &
        (flat_j >= 0) & (flat_j < ny) &
        (flat_i >= 0) & (flat_i < nx) &
        coarse_in_bbox[j_clip, i_clip]
    )

    if not valid.any():
        logger.info("  bbox_depth_percentile: no valid fine cells in bbox region — skipped")
        return depth_out

    # ---- Groupby (j, i) → percentile ------------------------------------
    df = pd.DataFrame({
        'j': flat_j[valid].astype(np.int32),
        'i': flat_i[valid].astype(np.int32),
        'depth': flat_d[valid],
    })
    pct_series = df.groupby(['j', 'i'])['depth'].quantile(percentile / 100.0)

    # ---- Apply: only deepen; skip marginal and land cells ---------------
    n_deepened = 0
    for (j, i), pct_val in pct_series.items():  # type: ignore[misc]
        cur = depth_out[j, i]
        if not np.isfinite(cur):                   # land → skip
            continue
        if wet_fraction[j, i] < min_wet_fraction:
            continue
        if pct_val > cur:
            depth_out[j, i] = float(pct_val)
            n_deepened += 1

    logger.info(
        "  bbox_depth_percentile (%.0f%%): %d / %d bbox cell(s) deepened",
        percentile, n_deepened, n_target,
    )
    return depth_out
