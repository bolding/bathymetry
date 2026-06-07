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

Note: strait detection is implemented for spherical (lon/lat) grids only.
Rotated or Cartesian grids emit a warning and return no flagged straits.

Isolated-cell masking
---------------------
Uses scipy.ndimage flood-fill labelling to identify connected wet regions.
The *nkeep* largest are retained; all others are set to land.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray as xr

from grid import BaseGrid, SphericalGrid

# Earth radius used for width estimates
_R_EARTH_KM = 6371.0


# ---------------------------------------------------------------------------
# Isolated-cell masking
# ---------------------------------------------------------------------------

def mask_isolated(
    dst: xr.Dataset,
    nkeep: int = 1,
) -> tuple[xr.Dataset, list[dict]]:
    """Remove disconnected wet regions, keeping the *nkeep* largest.

    Adapted from pygetm.domain.Domain.mask_subbasins() (GETM, line 1312).

    Parameters
    ----------
    dst : xr.Dataset
        Regridded destination dataset with ``depth``, ``wet_fraction``, and
        ``mask`` variables.
    nkeep : int
        Number of connected ocean basins to retain (largest first).

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
        Destination grid object; must be a SphericalGrid for analysis to run.
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
    if not isinstance(dst_grid, SphericalGrid):
        warnings.warn(
            "Strait detection is only implemented for SphericalGrid. "
            "Skipping analysis for this grid type.",
            stacklevel=2,
        )
        return []

    if dst_grid.rotation_deg != 0.0:
        warnings.warn(
            "Strait detection is not supported for rotated grids. "
            "Skipping.",
            stacklevel=2,
        )
        return []

    mask = dst["mask"].values.astype(bool)        # [ny_dst, nx_dst]
    depth_dst = np.where(mask, dst["depth"].values, 0.0)
    wf = dst["wet_fraction"].values               # [ny_dst, nx_dst]

    src_lon = src.lon.values
    src_lat = src.lat.values
    src_depth = np.where(src["land"].values, 0.0, src["depth"].values)

    dlon_src = float(np.diff(src_lon).mean())
    dlat_src = float(np.diff(src_lat).mean())
    dlon_dst = dst_grid.dlon
    dlat_dst = dst_grid.dlat

    # Fine cells per coarse cell (approximate)
    ratio_lon = dlon_dst / dlon_src
    ratio_lat = dlat_dst / dlat_src

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
                dst_grid, depth_dst, ratio_lon, ratio_lat,
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
                dst_grid, depth_dst, ratio_lon, ratio_lat,
                width_threshold_cells, sill_ratio_threshold, area_ratio_threshold,
            )
            if rec is not None:
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _analyse_u_interface(
    i: int, j: int,
    src_lon, src_lat, src_depth,
    dlon_src, dlat_src,
    dst_grid: SphericalGrid,
    depth_dst,
    ratio_lon, ratio_lat,
    width_thr, sill_thr, area_thr,
) -> Optional[dict]:
    """Analyse the interface between coarse cells (i,j) and (i,j+1)."""
    # Coarse cell centres and half-extents
    clon_j = dst_grid.center_lon[i, j]
    clon_j1 = dst_grid.center_lon[i, j + 1]
    clat_i = dst_grid.center_lat[i, j]

    lon_interface = (clon_j + clon_j1) / 2.0
    lon_half = dst_grid.dlon                      # one coarse cell wide each side
    lat_half = dst_grid.dlat / 2.0

    il_start, il_end = _fine_indices(lon_interface, lon_half, src_lon)
    ia_start, ia_end = _fine_indices(clat_i, lat_half, src_lat)

    if il_end <= il_start or ia_end <= ia_start:
        return None

    fine_sub = src_depth[ia_start:ia_end, il_start:il_end]  # [ny_f, nx_f]
    split_col = fine_sub.shape[1] // 2

    # Cross-section: column nearest to the interface
    cs_col = np.argmin(np.abs(src_lon[il_start:il_end] - lon_interface))
    depth_section = fine_sub[:, cs_col]

    dlat_km = dlat_src * _R_EARTH_KM * np.pi / 180.0
    width_km, sill_fine, area_fine = _section_metrics(depth_section, dlat_km)

    # Coarse equivalent: depth × height of cell
    clat_height_km = dst_grid.dlat * _R_EARTH_KM * np.pi / 180.0
    depth_coarse_avg = (depth_dst[i, j] + depth_dst[i, j + 1]) / 2.0
    area_coarse = depth_coarse_avg * clat_height_km * 1000.0
    sill_coarse = depth_coarse_avg

    sill_ratio = sill_fine / sill_coarse if sill_coarse > 0 else 0.0
    area_ratio = area_fine / area_coarse if area_coarse > 0 else 0.0

    # Width threshold: min-width in fine cells vs coarse cell width in fine cells
    width_thr_km = width_thr * dst_grid.dlon * _R_EARTH_KM * np.pi / 180.0 * np.cos(
        np.radians(clat_i)
    )

    connected = _connectivity_ok(fine_sub, split_col)
    category, fix = _classify(sill_ratio, area_ratio, connected, sill_thr, area_thr)

    if category == "OK" and width_km >= width_thr_km:
        return None

    return {
        "lon": lon_interface,
        "lat": clat_i,
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
        # Store for profile plots
        "_depth_section": depth_section,
        "_dlat_km": dlat_km,
    }


def _analyse_v_interface(
    i: int, j: int,
    src_lon, src_lat, src_depth,
    dlon_src, dlat_src,
    dst_grid: SphericalGrid,
    depth_dst,
    ratio_lon, ratio_lat,
    width_thr, sill_thr, area_thr,
) -> Optional[dict]:
    """Analyse the interface between coarse cells (i,j) and (i+1,j)."""
    clon_j = dst_grid.center_lon[i, j]
    clat_i = dst_grid.center_lat[i, j]
    clat_i1 = dst_grid.center_lat[i + 1, j]

    lat_interface = (clat_i + clat_i1) / 2.0
    lat_half = dst_grid.dlat
    lon_half = dst_grid.dlon / 2.0

    il_start, il_end = _fine_indices(clon_j, lon_half, src_lon)
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

    clon_width_km = dst_grid.dlon * _R_EARTH_KM * np.pi / 180.0 * cos_lat
    depth_coarse_avg = (depth_dst[i, j] + depth_dst[i + 1, j]) / 2.0
    area_coarse = depth_coarse_avg * clon_width_km * 1000.0
    sill_coarse = depth_coarse_avg

    sill_ratio = sill_fine / sill_coarse if sill_coarse > 0 else 0.0
    area_ratio = area_fine / area_coarse if area_coarse > 0 else 0.0
    width_thr_km = width_thr * dst_grid.dlat * _R_EARTH_KM * np.pi / 180.0

    connected = _connectivity_ok(fine_sub, split_row)  # type: ignore[arg-type]
    category, fix = _classify(sill_ratio, area_ratio, connected, sill_thr, area_thr)

    if category == "OK" and width_km >= width_thr_km:
        return None

    return {
        "lon": clon_j,
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
    }


def strait_summary(records: list[dict]) -> dict:
    categories = [r["category"] for r in records]
    return {
        "flagged interfaces total": len(records),
        "BLOCKED": categories.count("BLOCKED"),
        "SILL_DEFICIT": categories.count("SILL_DEFICIT"),
        "AREA_DEFICIT": categories.count("AREA_DEFICIT"),
    }


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
        elif action == "close_cell":
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
            import matplotlib.path as mpath
            verts = region["vertices"]   # [[lon, lat], ...]
            path  = mpath.Path(verts)
            pts   = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
            sel   = path.contains_points(pts).reshape(lon_2d.shape) & mask
        elif rtype == "point":
            flon, flat = float(region["lon"]), float(region["lat"])
            dist = (lon_2d - flon) ** 2 + (lat_2d - flat) ** 2
            iy, ix = np.unravel_index(int(dist.argmin()), dist.shape)
            sel = np.zeros_like(mask)
            sel[iy, ix] = mask[iy, ix]   # only if the cell is wet
        else:
            raise ValueError(
                f"Unknown mask region type {rtype!r}. "
                "Supported: rectangle, polygon, point."
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


def mask_regions_summary(applied: list[dict]) -> dict:
    return {
        "regions applied": len(applied),
        "total cells masked": sum(r["cells_masked"] for r in applied),
    }
