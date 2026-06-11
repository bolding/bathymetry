"""Bathymetry quality analysis: strait detection and isolated-cell masking.

Strait / connectivity detection
--------------------------------
For every interface between adjacent wet coarse cells the algorithm extracts
the corresponding fine-resolution cross-section and computes:

* ``width_km``      – minimum wet width at the interface (km)
* ``sill_depth``    – deepest point in the fine cross-section (m)
* ``section_area``  – integral of depth across wet cells (m²), proportional to
                      transport capacity
* ``sill_ratio``    – sill_depth_fine / sill_depth_coarse
* ``area_ratio``    – section_area_fine / section_area_coarse

Interfaces are classified into:

AREA_DEFICIT
    area_ratio < area_ratio_threshold (transport capacity under-represented)
SILL_DEFICIT
    sill_ratio < sill_ratio_threshold (dense bottom-water inflow blocked)
BLOCKED
    No continuous fine-resolution wet path exists between the two sub-tiles.
LAND_BRIDGE
    A land cell (wet_fraction > 0, forced below ``min_wet_fraction``) that
    sits between two otherwise-disconnected wet basins.  Opening it would
    reconnect them.

Works for any grid type (SphericalGrid, RotatedPoleGrid, CartesianGrid).
Cell extents are estimated from geographic centre-coordinate neighbour distances.

Isolated-cell masking
---------------------
Uses scipy.ndimage flood-fill labelling to identify connected wet regions.
The *nkeep* largest are retained; all others are set to land.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray as xr

from grid import BaseGrid, SphericalGrid  # SphericalGrid kept for fast-path check

# Earth radius used for width estimates
_R_EARTH_KM = 6371.0


# ---------------------------------------------------------------------------
# Isolated-cell masking
# ---------------------------------------------------------------------------

def mask_isolated(
    dst: xr.Dataset,
    nkeep: int = 1,
    keep_basins: list[int] | None = None,
) -> tuple[xr.Dataset, list[dict]]:
    """Remove disconnected wet regions.

    Basins are numbered 1, 2, 3 … in descending size order (1 = largest).

    By default the *nkeep* largest basins are retained.  Pass *keep_basins*
    to select specific basins by their size-rank number instead; *nkeep* is
    ignored when *keep_basins* is given.

    Adapted from pygetm.domain.Domain.mask_subbasins() (GETM, line 1312).

    Parameters
    ----------
    dst : xr.Dataset
        Regridded destination dataset with ``depth``, ``wet_fraction``, and
        ``mask`` variables.
    nkeep : int
        Number of connected ocean basins to retain (largest first).
        Ignored when *keep_basins* is provided.
    keep_basins : list[int] | None
        Explicit list of size-rank basin numbers to retain (1 = largest).
        Overrides *nkeep* when given.

    Returns
    -------
    dst_masked : xr.Dataset
        Updated dataset with isolated regions set to land.
    basin_records : list[dict]
        Summary of each identified basin (id, size, retained).
    """
    from scipy.ndimage import label

    ocean = dst["mask"].values.astype(bool)
    labelled, n_features = label(ocean)  # 4-connectivity

    basin_sizes: dict[int, int] = {}
    for bid in range(1, n_features + 1):
        basin_sizes[bid] = int((labelled == bid).sum())

    ordered = sorted(basin_sizes, key=lambda b: basin_sizes[b], reverse=True)

    if keep_basins is not None:
        # keep_basins contains 1-based size-rank indices
        keep_ids = {ordered[r - 1] for r in keep_basins if 1 <= r <= len(ordered)}
    else:
        keep_ids = set(ordered[:nkeep])

    new_ocean = np.isin(labelled, list(keep_ids))
    new_mask = new_ocean.astype(np.int8)
    new_depth = np.where(new_ocean, dst["depth"].values, np.nan)
    new_wf = np.where(new_ocean, dst["wet_fraction"].values, 0.0)

    basin_records = [
        {
            "basin_id": bid,
            "size_cells": basin_sizes[bid],
            "retained": bid in keep_ids,
        }
        for bid in ordered
    ]

    dst_masked = xr.Dataset(
        {
            "depth": (["lat", "lon"], new_depth),
            "wet_fraction": (["lat", "lon"], new_wf),
            "mask": (["lat", "lon"], new_mask),
            "basin_labels": (["lat", "lon"], labelled.astype(np.int32)),
        },
        coords=dst.coords,
        attrs=dst.attrs,
    )
    return dst_masked, basin_records


def isolation_summary(basin_records: list[dict]) -> dict:
    total = len(basin_records)
    removed = [r for r in basin_records if not r["retained"]]
    removed_cells = sum(r["size_cells"] for r in removed)
    return {
        "connected regions found": total,
        "regions removed": len(removed),
        "cells masked as isolated": removed_cells,
        "regions retained": total - len(removed),
    }


# ---------------------------------------------------------------------------
# Strait / connectivity detection
# ---------------------------------------------------------------------------

def find_straits(
    src: xr.Dataset,
    dst: xr.Dataset,
    dst_grid: BaseGrid,
    wet_frac_threshold: float = 0.3,
    width_threshold_cells: float = 2.0,
    sill_ratio_threshold: float = 0.7,
    area_ratio_threshold: float = 0.5,
) -> list[dict]:
    """Identify narrow or blocked interfaces between adjacent wet coarse cells.

    Parameters
    ----------
    src : xr.Dataset
        Fine-resolution source bathymetry (from reader.read_source).
    dst : xr.Dataset
        Coarse regridded bathymetry (from interpolate.regrid).
    dst_grid : BaseGrid
        Destination grid object.  Works for any grid type — SphericalGrid,
        RotatedPoleGrid, CartesianGrid.
    wet_frac_threshold : float
        Only inspect interfaces where at least one adjacent cell has
        wet_fraction < this value.
    width_threshold_cells : float
        Flag as narrow if the minimum cross-section width (in fine cells) is
        less than this many coarse-cell widths.
    sill_ratio_threshold : float
        Flag as SILL_DEFICIT if sill_depth_fine / sill_depth_coarse < threshold.
    area_ratio_threshold : float
        Flag as AREA_DEFICIT if section_area_fine / section_area_coarse < threshold.

    Returns
    -------
    records : list[dict]
        One record per flagged interface.  Fields: ``lon``, ``lat``,
        ``direction``, ``category``, ``width_km``, ``sill_depth_fine``,
        ``sill_depth_coarse``, ``sill_ratio``, ``section_area_fine``,
        ``section_area_coarse``, ``area_ratio``, ``connected``,
        ``suggested_fix``.
    """
    mask = dst["mask"].values.astype(bool)        # [ny_dst, nx_dst]
    depth_dst = np.where(mask, dst["depth"].values, 0.0)
    wf = dst["wet_fraction"].values               # [ny_dst, nx_dst]

    src_lon = src.lon.values
    src_lat = src.lat.values
    src_depth = np.where(src["land"].values, 0.0, src["depth"].values)

    dlon_src = float(np.diff(src_lon).mean())
    dlat_src = float(np.diff(src_lat).mean())

    # Per-cell geographic extent in degrees — works for axis-aligned and rotated grids.
    # For a SphericalGrid with no rotation the fast-path returns constant arrays.
    dlon_cell, dlat_cell = _cell_size_arrays(dst_grid)

    ny_dst, nx_dst = mask.shape
    records: list[dict] = []

    # Check every U interface (between columns j and j+1 in the same row i)
    for i in range(ny_dst):
        for j in range(nx_dst - 1):
            if not (mask[i, j] and mask[i, j + 1]):
                continue
            if wf[i, j] >= wet_frac_threshold and wf[i, j + 1] >= wet_frac_threshold:
                continue
            rec = _analyse_u_interface(
                i, j, src_lon, src_lat, src_depth, dlon_src, dlat_src,
                dst_grid, depth_dst, dlon_cell, dlat_cell,
                width_threshold_cells, sill_ratio_threshold, area_ratio_threshold,
            )
            if rec is not None:
                records.append(rec)

    # Check every V interface (between rows i and i+1 in the same column j)
    for i in range(ny_dst - 1):
        for j in range(nx_dst):
            if not (mask[i, j] and mask[i + 1, j]):
                continue
            if wf[i, j] >= wet_frac_threshold and wf[i + 1, j] >= wet_frac_threshold:
                continue
            rec = _analyse_v_interface(
                i, j, src_lon, src_lat, src_depth, dlon_src, dlat_src,
                dst_grid, depth_dst, dlon_cell, dlat_cell,
                width_threshold_cells, sill_ratio_threshold, area_ratio_threshold,
            )
            if rec is not None:
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cell_size_arrays(
    dst_grid: BaseGrid,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return per-cell geographic extents (degrees) for any grid type.

    For a non-rotated SphericalGrid the result is constant (fast path).
    For rotated or Cartesian grids, cell extents are estimated from the
    geographic distances between adjacent cell centres — the only information
    available from ``BaseGrid`` without knowing the concrete subclass.

    ``dlon_cell[i, j]`` ≈ geographic longitude span of cell (i, j).
    ``dlat_cell[i, j]`` ≈ geographic latitude span of cell (i, j).
    """
    if isinstance(dst_grid, SphericalGrid) and dst_grid.rotation_deg == 0.0:
        dlon_cell = np.full(dst_grid.center_lon.shape, dst_grid.dlon)
        dlat_cell = np.full(dst_grid.center_lat.shape, dst_grid.dlat)
        return dlon_cell, dlat_cell

    clon = dst_grid.center_lon   # [ny, nx]
    clat = dst_grid.center_lat   # [ny, nx]

    # Longitude span: central differences; forward/backward at boundaries.
    dlon = np.empty_like(clon)
    dlon[:, 1:-1] = np.abs(clon[:, 2:] - clon[:, :-2]) / 2.0
    dlon[:, 0]    = np.abs(clon[:, 1]  - clon[:, 0])
    dlon[:, -1]   = np.abs(clon[:, -1] - clon[:, -2])

    # Latitude span: central differences; forward/backward at boundaries.
    dlat = np.empty_like(clat)
    dlat[1:-1, :] = np.abs(clat[2:, :] - clat[:-2, :]) / 2.0
    dlat[0, :]    = np.abs(clat[1, :]  - clat[0, :])
    dlat[-1, :]   = np.abs(clat[-1, :] - clat[-2, :])

    return dlon, dlat


def _fine_indices(
    center_val: float, half_width: float, coords_1d: npt.NDArray
) -> tuple[int, int]:
    """Return (i_start, i_end) slice indices in *coords_1d* for the window
    [center_val - half_width, center_val + half_width]."""
    idx_start = int(np.searchsorted(coords_1d, center_val - half_width))
    idx_end = int(np.searchsorted(coords_1d, center_val + half_width, side="right"))
    idx_start = max(0, idx_start)
    idx_end = min(len(coords_1d), idx_end)
    return idx_start, idx_end


def _section_metrics(
    depth_section: npt.NDArray,
    cell_size_km: float,
) -> tuple[float, float, float]:
    """From a 1-D depth profile at a cross-section, return
    (width_km, sill_depth, section_area_m2)."""
    wet = depth_section > 0.0
    n_wet = int(wet.sum())
    width_km = n_wet * cell_size_km
    if n_wet == 0:
        return 0.0, 0.0, 0.0
    sill_depth = float(depth_section[wet].max())
    section_area = float(depth_section[wet].sum()) * cell_size_km * 1000.0  # m²
    return width_km, sill_depth, section_area


def _connectivity_ok(fine_sub: npt.NDArray, split_col: int) -> bool:
    """Return True if the left and right halves of *fine_sub* are connected."""
    from scipy.ndimage import label

    ocean = fine_sub > 0.0
    if not ocean[:, :split_col].any() or not ocean[:, split_col:].any():
        return False
    labelled, _ = label(ocean)
    left_ids = set(labelled[:, split_col - 1][ocean[:, split_col - 1]].tolist())
    right_ids = set(labelled[:, split_col][ocean[:, split_col]].tolist())
    return bool(left_ids & right_ids)


def _classify(
    sill_ratio: float,
    area_ratio: float,
    connected: bool,
    sill_thr: float,
    area_thr: float,
) -> tuple[str, str]:
    if not connected:
        return "BLOCKED", "Open blocking fine cells and re-interpolate"
    if sill_ratio < sill_thr:
        return (
            "SILL_DEFICIT",
            f"Override coarse depth to fine sill depth ({sill_ratio:.2f} ratio)",
        )
    if area_ratio < area_thr:
        return (
            "AREA_DEFICIT",
            "Open neighbouring dry cells or deepen to restore cross-section area",
        )
    return "OK", ""


def _coarse_corners_for_interface(
    dst_grid: BaseGrid,
    i: int,
    j: int,
    direction: str,
    pad: int = 1,
) -> tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
    """Return coarse corner arrays for the neighbourhood of interface (i,j).

    For a U-interface the two cells are (i,j) and (i,j+1); for a V-interface
    they are (i,j) and (i+1,j).  *pad* extra cells are included on every side
    so the two interface cells are always shown in full even when they sit at
    the edge of the analysis window.

    Returns ``(corner_lons_2d, corner_lats_2d)`` or ``(None, None)``.
    """
    ny, nx = dst_grid.center_lon.shape
    if direction == "U":
        i_min = max(0, i - pad)
        i_max = min(ny - 1, i + pad)
        j_min = max(0, j - pad)
        j_max = min(nx - 1, j + 1 + pad)
    else:  # V
        i_min = max(0, i - pad)
        i_max = min(ny - 1, i + 1 + pad)
        j_min = max(0, j - pad)
        j_max = min(nx - 1, j + pad)
    return (
        dst_grid.corner_lon[i_min:i_max + 2, j_min:j_max + 2],
        dst_grid.corner_lat[i_min:i_max + 2, j_min:j_max + 2],
    )


def _analyse_u_interface(
    i: int, j: int,
    src_lon, src_lat, src_depth,
    dlon_src, dlat_src,
    dst_grid: BaseGrid,
    depth_dst,
    dlon_cell: npt.NDArray,
    dlat_cell: npt.NDArray,
    width_thr, sill_thr, area_thr,
) -> Optional[dict]:
    """Analyse the interface between coarse cells (i,j) and (i,j+1)."""
    clon_j  = dst_grid.center_lon[i, j]
    clon_j1 = dst_grid.center_lon[i, j + 1]
    clat_j  = dst_grid.center_lat[i, j]
    clat_j1 = dst_grid.center_lat[i, j + 1]

    # Geographic midpoint of the interface (works for rotated grids too)
    lon_interface = (clon_j + clon_j1) / 2.0
    lat_interface = (clat_j + clat_j1) / 2.0

    # Window: one average cell wide in each axis so we capture both sides
    lon_half = (dlon_cell[i, j] + dlon_cell[i, j + 1]) / 2.0
    lat_half = (dlat_cell[i, j] + dlat_cell[i, j + 1]) / 4.0  # half cell tall

    il_start, il_end = _fine_indices(lon_interface, lon_half, src_lon)
    ia_start, ia_end = _fine_indices(lat_interface, lat_half, src_lat)

    if il_end <= il_start or ia_end <= ia_start:
        return None

    fine_sub = src_depth[ia_start:ia_end, il_start:il_end]  # [ny_f, nx_f]
    split_col = fine_sub.shape[1] // 2

    # Cross-section: column nearest to the interface longitude
    cs_col = np.argmin(np.abs(src_lon[il_start:il_end] - lon_interface))
    depth_section = fine_sub[:, cs_col]

    dlat_km = dlat_src * _R_EARTH_KM * np.pi / 180.0
    width_km, sill_fine, area_fine = _section_metrics(depth_section, dlat_km)

    # Coarse equivalent: depth × geographic height of cell
    cell_dlat = (dlat_cell[i, j] + dlat_cell[i, j + 1]) / 2.0
    clat_height_km = cell_dlat * _R_EARTH_KM * np.pi / 180.0
    depth_coarse_avg = (depth_dst[i, j] + depth_dst[i, j + 1]) / 2.0
    area_coarse = depth_coarse_avg * clat_height_km * 1000.0
    sill_coarse = depth_coarse_avg

    sill_ratio = sill_fine / sill_coarse if sill_coarse > 0 else 0.0
    area_ratio = area_fine / area_coarse if area_coarse > 0 else 0.0

    cell_dlon = (dlon_cell[i, j] + dlon_cell[i, j + 1]) / 2.0
    width_thr_km = width_thr * cell_dlon * _R_EARTH_KM * np.pi / 180.0 * np.cos(
        np.radians(lat_interface)
    )

    connected = _connectivity_ok(fine_sub, split_col)
    category, fix = _classify(sill_ratio, area_ratio, connected, sill_thr, area_thr)

    if category == "OK" and width_km >= width_thr_km:
        return None

    return {
        "lon": lon_interface,
        "lat": lat_interface,
        "direction": "U",
        "category": category,
        "width_km": round(width_km, 2),
        "sill_depth_fine": round(sill_fine, 1),
        "sill_depth_coarse": round(sill_coarse, 1),
        "sill_ratio": round(sill_ratio, 3),
        "section_area_fine": round(area_fine, 0),
        "section_area_coarse": round(area_coarse, 0),
        "area_ratio": round(area_ratio, 3),
        "connected": connected,
        "suggested_fix": fix,
        "_depth_section": depth_section,
        "_dlat_km": dlat_km,
        "_fine_sub": fine_sub,
        "_fine_lons": src_lon[il_start:il_end],
        "_fine_lats": src_lat[ia_start:ia_end],
        "_cs_col": cs_col,
        **dict(zip(
            ("_coarse_corner_lons", "_coarse_corner_lats"),
            _coarse_corners_for_interface(dst_grid, i, j, "U"),
        )),
    }


def _analyse_v_interface(
    i: int, j: int,
    src_lon, src_lat, src_depth,
    dlon_src, dlat_src,
    dst_grid: BaseGrid,
    depth_dst,
    dlon_cell: npt.NDArray,
    dlat_cell: npt.NDArray,
    width_thr, sill_thr, area_thr,
) -> Optional[dict]:
    """Analyse the interface between coarse cells (i,j) and (i+1,j)."""
    clon_i  = dst_grid.center_lon[i, j]
    clon_i1 = dst_grid.center_lon[i + 1, j]
    clat_i  = dst_grid.center_lat[i, j]
    clat_i1 = dst_grid.center_lat[i + 1, j]

    lon_interface = (clon_i + clon_i1) / 2.0
    lat_interface = (clat_i + clat_i1) / 2.0

    lat_half = (dlat_cell[i, j] + dlat_cell[i + 1, j]) / 2.0   # one average cell tall
    lon_half = (dlon_cell[i, j] + dlon_cell[i + 1, j]) / 4.0   # half cell wide

    il_start, il_end = _fine_indices(lon_interface, lon_half, src_lon)
    ia_start, ia_end = _fine_indices(lat_interface, lat_half, src_lat)

    if il_end <= il_start or ia_end <= ia_start:
        return None

    fine_sub = src_depth[ia_start:ia_end, il_start:il_end]
    split_row = fine_sub.shape[0] // 2

    cs_row = np.argmin(np.abs(src_lat[ia_start:ia_end] - lat_interface))
    depth_section = fine_sub[cs_row, :]

    cos_lat = np.cos(np.radians(lat_interface))
    dlon_km = dlon_src * _R_EARTH_KM * np.pi / 180.0 * cos_lat
    width_km, sill_fine, area_fine = _section_metrics(depth_section, dlon_km)

    cell_dlon = (dlon_cell[i, j] + dlon_cell[i + 1, j]) / 2.0
    clon_width_km = cell_dlon * _R_EARTH_KM * np.pi / 180.0 * cos_lat
    depth_coarse_avg = (depth_dst[i, j] + depth_dst[i + 1, j]) / 2.0
    area_coarse = depth_coarse_avg * clon_width_km * 1000.0
    sill_coarse = depth_coarse_avg

    sill_ratio = sill_fine / sill_coarse if sill_coarse > 0 else 0.0
    area_ratio = area_fine / area_coarse if area_coarse > 0 else 0.0

    cell_dlat = (dlat_cell[i, j] + dlat_cell[i + 1, j]) / 2.0
    width_thr_km = width_thr * cell_dlat * _R_EARTH_KM * np.pi / 180.0

    connected = _connectivity_ok(fine_sub, split_row)  # type: ignore[arg-type]
    category, fix = _classify(sill_ratio, area_ratio, connected, sill_thr, area_thr)

    if category == "OK" and width_km >= width_thr_km:
        return None

    return {
        "lon": lon_interface,
        "lat": lat_interface,
        "direction": "V",
        "category": category,
        "width_km": round(width_km, 2),
        "sill_depth_fine": round(sill_fine, 1),
        "sill_depth_coarse": round(sill_coarse, 1),
        "sill_ratio": round(sill_ratio, 3),
        "section_area_fine": round(area_fine, 0),
        "section_area_coarse": round(area_coarse, 0),
        "area_ratio": round(area_ratio, 3),
        "connected": connected,
        "suggested_fix": fix,
        "_depth_section": depth_section,
        "_dlon_km": dlon_km,
        "_fine_sub": fine_sub,
        "_fine_lons": src_lon[il_start:il_end],
        "_fine_lats": src_lat[ia_start:ia_end],
        "_cs_row": cs_row,
        **dict(zip(
            ("_coarse_corner_lons", "_coarse_corner_lats"),
            _coarse_corners_for_interface(dst_grid, i, j, "V"),
        )),
    }


_CAT_RANK = {"BLOCKED": 0, "SILL_DEFICIT": 1, "AREA_DEFICIT": 2, "OK": 3}


def sort_straits(records: list[dict]) -> list[dict]:
    """Return records sorted by physical importance, worst first.

    Primary key: category severity (BLOCKED → SILL_DEFICIT → AREA_DEFICIT).

    Secondary key within each category: absolute sill deficit in metres
    (descending).  For BLOCKED (sill_ratio = 0) this equals the coarse cell
    depth, so deep blockages rank above shallow tidal-flat cases.  For
    SILL_DEFICIT it is the actual depth underestimate::

        sill_deficit_m = sill_depth_coarse × (1 − sill_ratio)

    This ensures a 2 m BLOCKED coastal cell sorts far below a 100 m BLOCKED
    deep-water strait.
    """
    def _key(r: dict):
        coarse = r.get("sill_depth_coarse", 0.0)
        deficit = coarse * (1.0 - r.get("sill_ratio", 0.0))
        return (
            _CAT_RANK.get(r.get("category", "OK"), 9),
            -deficit,                       # descending: larger deficit first
            r.get("area_ratio", 1.0),       # lower area ratio = worse tiebreaker
        )
    return sorted(records, key=_key)


def strait_summary(records: list[dict]) -> dict:
    categories = [r["category"] for r in records]
    n_blocked  = categories.count("BLOCKED")
    n_sill     = categories.count("SILL_DEFICIT")
    n_area     = categories.count("AREA_DEFICIT")
    n_ok       = categories.count("OK")
    return {
        "flagged interfaces total": len(records),
        "BLOCKED":      n_blocked,
        "SILL_DEFICIT": n_sill,
        "AREA_DEFICIT": n_area,
        "OK (narrow)":  n_ok,
    }


# ---------------------------------------------------------------------------
# Phantom island detection
# ---------------------------------------------------------------------------

def detect_phantom_islands(
    dst: xr.Dataset,
    max_wet_fraction: float = 0.5,
    search_radius: int = 2,
    max_cluster_size: int = 4,
) -> list[dict]:
    """Detect ocean cells that are mostly land in the fine-resolution source.

    A *phantom island* is a coarse ocean cell (mask=1) whose fine-grid
    wet-fraction is below *max_wet_fraction*, AND that has no coarse land
    cell (mask=0) within *search_radius* cells — meaning the cell is
    isolated from the main land mass and is likely a real island that the
    conservative regrid failed to preserve as land.

    Connected groups of such cells up to *max_cluster_size* are returned as
    individual records (one per cell).  Larger connected regions are skipped
    because they are more likely to be coastal features than isolated islands.

    Parameters
    ----------
    dst : xr.Dataset
        Regridded dataset with ``mask``, ``wet_fraction``, ``depth``, and
        2-D ``lon``/``lat`` coordinate arrays.
    max_wet_fraction : float
        Cells with wet_fraction below this are considered phantom-island
        candidates (default 0.5 — cell is majority land in fine grid).
    search_radius : int
        Square neighbourhood half-width in cells.  All cells within this
        radius must be ocean (mask=1) for the candidate to qualify.
    max_cluster_size : int
        Connected components (4-connectivity) larger than this are not
        flagged — they are more likely poorly-resolved coast than islands.

    Returns
    -------
    list[dict]
        One record per phantom-island cell with keys:
        ``lon``, ``lat``, ``wet_fraction``, ``depth``, ``cluster_id``,
        ``cluster_size``.
    """
    from scipy.ndimage import label, binary_dilation

    mask = dst["mask"].values.astype(bool)          # True = ocean
    wf   = dst["wet_fraction"].values
    depth_arr = dst["depth"].values
    lon_2d = dst.lon.values
    lat_2d = dst.lat.values
    ny, nx = mask.shape

    # --- candidates: ocean cells with low wet_fraction ----------------------
    candidates = mask & (wf < max_wet_fraction)

    # --- neighbourhood check: no land within search_radius ------------------
    # Dilate the LAND mask by search_radius; any candidate that overlaps the
    # dilated land mask is adjacent to the coast and is therefore not an island.
    struct = np.ones((2 * search_radius + 1, 2 * search_radius + 1), dtype=bool)
    land_dilated = binary_dilation(~mask, structure=struct)
    isolated = candidates & ~land_dilated

    if not isolated.any():
        return []

    # --- find connected clusters of isolated candidates ---------------------
    labelled, n_clusters = label(isolated)

    records = []
    for cid in range(1, n_clusters + 1):
        cells = np.argwhere(labelled == cid)
        if len(cells) > max_cluster_size:
            continue
        for iy, ix in cells:
            records.append({
                "lon":          float(lon_2d[iy, ix]),
                "lat":          float(lat_2d[iy, ix]),
                "wet_fraction": float(wf[iy, ix]),
                "depth":        float(depth_arr[iy, ix])
                                if np.isfinite(depth_arr[iy, ix]) else 0.0,
                "cluster_id":   int(cid),
                "cluster_size": int(len(cells)),
            })

    return records


# ---------------------------------------------------------------------------
# Fix application
# ---------------------------------------------------------------------------

def apply_fixes(
    dst: "xr.Dataset",
    fixes: list[dict],
) -> tuple["xr.Dataset", list[dict]]:
    """Apply user-specified bathymetry fixes to the destination dataset.

    Each fix is a dict with at least ``lon``, ``lat``, and ``action``.
    The nearest coarse cell is found by Euclidean distance in lon/lat space.

    Actions
    -------
    set_depth
        Force the cell depth to ``value`` (m) and mark it as ocean.
    open_cell
        Same as set_depth but communicates intent to open a blocked passage.
        Uses ``depth`` key (alias for ``value``).
    close_cell
        Force the cell to land (mask=0, depth=NaN).

    Parameters
    ----------
    dst : xr.Dataset
        Regridded destination dataset.
    fixes : list[dict]
        List of fix dicts from the YAML ``fixes:`` section.

    Returns
    -------
    dst_fixed : xr.Dataset
        Updated dataset.
    applied : list[dict]
        Log of applied fixes (includes actual lon/lat snapped to grid).
    """
    import xarray as xr

    if not fixes:
        return dst, []

    lon_2d = dst.lon.values
    lat_2d = dst.lat.values
    depth = dst["depth"].values.copy()
    mask = dst["mask"].values.copy()
    wf = dst["wet_fraction"].values.copy()

    applied = []
    for fix in fixes:
        flon = float(fix["lon"])
        flat = float(fix["lat"])
        action = fix.get("action", "set_depth")

        dist = (lon_2d - flon) ** 2 + (lat_2d - flat) ** 2
        iy, ix = int(np.unravel_index(int(dist.argmin()), dist.shape)[0]), \
                 int(np.unravel_index(int(dist.argmin()), dist.shape)[1])

        if action in ("set_depth", "open_cell"):
            new_depth = float(fix.get("value", fix.get("depth", 10.0)))
            depth[iy, ix] = new_depth
            mask[iy, ix] = 1
            wf[iy, ix] = max(float(wf[iy, ix]), 0.01)
        elif action == "deepen_by":
            delta = float(fix.get("value", 0.0))
            depth[iy, ix] = max(float(depth[iy, ix]) + delta, 0.0)
            mask[iy, ix] = 1
            wf[iy, ix] = max(float(wf[iy, ix]), 0.01)
        elif action in ("close_cell", "mask_cell"):
            depth[iy, ix] = np.nan
            mask[iy, ix] = 0
            wf[iy, ix] = 0.0
        else:
            raise ValueError(f"Unknown fix action {action!r}")

        applied.append({
            "action": action,
            "requested_lon": flon,
            "requested_lat": flat,
            "actual_lon": float(lon_2d[iy, ix]),
            "actual_lat": float(lat_2d[iy, ix]),
            "distance_deg": float(np.sqrt(dist[iy, ix])),
            "row": iy,
            "col": ix,
            "value": float(fix.get("value", fix.get("depth", 10.0)))
                     if action in ("set_depth", "open_cell") else float("nan"),
        })

    dst_fixed = xr.Dataset(
        {
            "depth": (["lat", "lon"], depth),
            "wet_fraction": (["lat", "lon"], wf),
            "mask": (["lat", "lon"], mask),
            **{k: dst[k] for k in dst.data_vars
               if k not in ("depth", "wet_fraction", "mask")},
        },
        coords=dst.coords,
        attrs=dst.attrs,
    )
    return dst_fixed, applied


# ---------------------------------------------------------------------------
# Explicit mask regions
# ---------------------------------------------------------------------------

def apply_mask_regions(
    dst: "xr.Dataset",
    regions: list[dict],
) -> tuple["xr.Dataset", list[dict]]:
    """Force specific geographic areas to land regardless of bathymetry.

    Useful for removing water bodies that should not be part of the model
    domain (e.g. enclosed lagoons, river estuaries, small inland seas).

    Each region dict must have a ``type`` key.  Supported types:

    ``rectangle`` (aliases: ``rect``, ``box``)
        Keys: ``lon_min``, ``lon_max``, ``lat_min``, ``lat_max``.

    ``polygon``
        Key: ``vertices`` — list of ``[lon, lat]`` pairs forming a closed
        polygon.  Uses ``matplotlib.path.Path`` for the point-in-polygon test.

    ``point``
        Keys: ``lon``, ``lat``.  Masks the single nearest wet cell.

    ``ij_rectangle`` (aliases: ``ij_rect``, ``ij_box``)
        Keys: ``i_min``, ``i_max``, ``j_min``, ``j_max`` — 0-based grid indices
        as shown in ncview: i = x-direction (longitude column), j = y-direction
        (latitude row).  Useful when the exact indices are known from inspecting
        the NetCDF output.

    ``ij_point`` (alias: ``index_point``)
        Keys: ``i``, ``j`` — single cell: i = x (longitude column),
        j = y (latitude row).

    All types accept an optional ``name`` key used only in the report.

    Returns
    -------
    dst_masked : xr.Dataset
        Updated dataset with the selected cells set to land.
    applied : list[dict]
        Log entry for each region: name, type, number of cells masked.
    """
    import xarray as xr

    if not regions:
        return dst, []

    lon_2d = dst.lon.values
    lat_2d = dst.lat.values
    depth = dst["depth"].values.copy()
    mask  = dst["mask"].values.copy().astype(bool)
    wf    = dst["wet_fraction"].values.copy()

    applied = []
    for region in regions:
        rtype = str(region.get("type", "rectangle")).lower()
        label = region.get("name", rtype)

        if rtype in ("rectangle", "rect", "box"):
            sel = (
                (lon_2d >= float(region["lon_min"])) &
                (lon_2d <= float(region["lon_max"])) &
                (lat_2d >= float(region["lat_min"])) &
                (lat_2d <= float(region["lat_max"])) &
                mask
            )
        elif rtype == "polygon":
            import warnings as _warnings
            import matplotlib.path as mpath
            verts = region["vertices"]   # [[lon, lat], ...]
            if len(verts) < 3:
                # A polygon with fewer than 3 vertices has zero area and
                # contains_points will never match any cell centre.
                # Fall back to masking the nearest wet cell to each vertex.
                _warnings.warn(
                    f"Mask region '{label}': polygon has only {len(verts)} vertex/vertices "
                    f"(need ≥ 3 for an area).  Falling back to nearest-cell masking "
                    f"for each vertex.",
                    stacklevel=2,
                )
                sel = np.zeros_like(mask)
                for v in verts:
                    flon, flat = float(v[0]), float(v[1])
                    dist = (lon_2d - flon) ** 2 + (lat_2d - flat) ** 2
                    iy, ix = np.unravel_index(int(dist.argmin()), dist.shape)
                    sel[iy, ix] = mask[iy, ix]
            else:
                path = mpath.Path(verts)
                pts  = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
                sel  = path.contains_points(pts).reshape(lon_2d.shape) & mask
        elif rtype == "point":
            flon, flat = float(region["lon"]), float(region["lat"])
            dist = (lon_2d - flon) ** 2 + (lat_2d - flat) ** 2
            iy, ix = np.unravel_index(int(dist.argmin()), dist.shape)
            sel = np.zeros_like(mask)
            sel[iy, ix] = mask[iy, ix]   # only if the cell is wet
        elif rtype in ("ij_rectangle", "ij_rect", "ij_box"):
            ny, nx = mask.shape  # array shape is [lat(j), lon(i)]
            i_min = int(region.get("i_min", 0))
            i_max = int(region.get("i_max", nx - 1))
            j_min = int(region.get("j_min", 0))
            j_max = int(region.get("j_max", ny - 1))
            sel_ij = np.zeros_like(mask)
            sel_ij[j_min:j_max + 1, i_min:i_max + 1] = True
            sel = sel_ij & mask
        elif rtype in ("ij_point", "index_point"):
            i = int(region["i"])
            j = int(region["j"])
            sel = np.zeros_like(mask)
            if 0 <= j < mask.shape[0] and 0 <= i < mask.shape[1]:
                sel[j, i] = mask[j, i]
        else:
            raise ValueError(
                f"Unknown mask region type {rtype!r}. "
                "Supported: rectangle, polygon, point, ij_rectangle, ij_point."
            )

        n = int(sel.sum())
        depth[sel] = np.nan
        mask[sel]  = False
        wf[sel]    = 0.0
        applied.append({"name": label, "type": rtype, "cells_masked": n})

    dst_masked = xr.Dataset(
        {
            "depth":        (["lat", "lon"], depth),
            "wet_fraction": (["lat", "lon"], wf),
            "mask":         (["lat", "lon"], mask.astype(np.int8)),
            **{k: dst[k] for k in dst.data_vars
               if k not in ("depth", "wet_fraction", "mask")},
        },
        coords=dst.coords,
        attrs=dst.attrs,
    )
    return dst_masked, applied


def detect_land_bridges(
    mask: np.ndarray,
    wet_fraction: np.ndarray,
    lon_2d: np.ndarray,
    lat_2d: np.ndarray,
    depth: Optional[np.ndarray] = None,
) -> list[dict]:
    """Find land cells that separate two or more disconnected wet basins.

    After regridding, some cells with non-zero wet_fraction may have been
    forced to land by the ``min_wet_fraction`` threshold.  If such a cell
    sits between two otherwise-disconnected wet basins, opening it (via an
    ``open_cell`` fix) would restore the connection.

    Parameters
    ----------
    mask : ndarray [ny, nx]
        Final ocean mask (1=ocean, 0=land) after isolation masking.
    wet_fraction : ndarray [ny, nx]
        Wet fraction from the raw conservative regrid, **before** isolation
        masking zeroed it.  Non-zero values on land cells identify candidates.
    lon_2d, lat_2d : ndarray [ny, nx]
        T-point geographic coordinates.
    depth : ndarray [ny, nx] or None
        Final T-point depth (NaN for land).  Used to estimate a suggested
        depth for the ``open_cell`` fix (mean of adjacent wet cells).

    Returns
    -------
    list[dict]
        One record per LAND_BRIDGE cell.  Fields: ``i``, ``j``, ``lon``,
        ``lat``, ``wet_fraction``, ``n_components``, ``adjacent_components``,
        ``estimated_depth``, ``category``, ``suggested_fix``.
    """
    from scipy.ndimage import label

    ocean = mask.astype(bool)
    labelled, _ = label(ocean)

    land_with_wf = (~ocean) & (wet_fraction > 0.0)
    js, is_ = np.where(land_with_wf)

    ny, nx = mask.shape
    records: list[dict] = []

    for j, i in zip(js.tolist(), is_.tolist()):
        adjacent: set[int] = set()
        adj_depths: list[float] = []
        for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx:
                comp = int(labelled[nj, ni])
                if comp > 0:
                    adjacent.add(comp)
                    if depth is not None and np.isfinite(depth[nj, ni]):
                        adj_depths.append(float(depth[nj, ni]))

        if len(adjacent) < 2:
            continue

        est_depth = round(float(np.mean(adj_depths)), 1) if adj_depths else 5.0
        records.append({
            "i": int(i),
            "j": int(j),
            "lon": float(lon_2d[j, i]),
            "lat": float(lat_2d[j, i]),
            "wet_fraction": round(float(wet_fraction[j, i]), 4),
            "n_components": len(adjacent),
            "adjacent_components": sorted(adjacent),
            "category": "LAND_BRIDGE",
            "estimated_depth": est_depth,
            "suggested_fix": f"open_cell — bridges {len(adjacent)} disconnected basin(s)",
        })

    return records


def land_bridge_summary(records: list[dict]) -> dict:
    return {
        "land-bridge cells found": len(records),
        "cells connecting ≥3 basins": sum(1 for r in records if r["n_components"] >= 3),
    }


def mask_regions_summary(applied: list[dict]) -> dict:
    return {
        "regions applied": len(applied),
        "total cells masked": sum(r["cells_masked"] for r in applied),
    }
