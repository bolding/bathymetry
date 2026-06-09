"""Thalweg extraction and fine-vs-coarse comparison for strait quality assessment.

A *thalweg* is the longitudinal profile of deepest depths along a channel.
At each cross-section perpendicular to the flow direction the thalweg point
is the deepest wet cell.  Comparing the fine-resolution and coarse-resolution
thalweg reveals how much depth information is lost after regridding.

This module is called after :func:`analysis.find_straits`.  For each flagged
strait interface it:

1. Extracts the fine-resolution thalweg depth profile through the strait (N
   cross-sections along the flow axis, deepest wet cell per cross-section).
2. Samples the coarse-resolution depth at the same geographic positions
   (nearest wet coarse cell via KD-tree).
3. Reports the sill depth (profile minimum) for both resolutions and the
   deficit in metres.

Typical call
------------
::

    import thalweg as thalwegmod

    thalweg_records = thalwegmod.compute_strait_thalwegs(
        src, dst, strait_records, n_straits=8
    )
    for i, tw in enumerate(thalweg_records):
        report.plot_thalweg_comparison(tw, src, path=f"thalweg_{i:02d}.png")
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_R_EARTH_KM = 6371.0


def _great_circle_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2.0 * _R_EARTH_KM * math.asin(math.sqrt(min(1.0, a)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_coarse_tree(
    lon2d: npt.NDArray,
    lat2d: npt.NDArray,
    depth2d: npt.NDArray,
    mask2d: npt.NDArray,
):
    """Return a (KDTree, depth_values) pair for fast coarse nearest-cell lookup."""
    from scipy.spatial import cKDTree

    wet = mask2d.astype(bool)
    lons = lon2d[wet].ravel()
    lats = lat2d[wet].ravel()
    deps = depth2d[wet].ravel()
    if len(lons) == 0:
        return None, None
    tree = cKDTree(np.column_stack([lons, lats]))
    return tree, deps


def _sample_coarse(
    lons: npt.NDArray,
    lats: npt.NDArray,
    tree,
    depths: npt.NDArray,
    max_dist_deg: float = 1.0,
) -> npt.NDArray:
    """Return coarse depths at each (lon, lat) point via nearest-cell lookup.

    Points whose nearest coarse cell is farther than *max_dist_deg* in lon/lat
    space are returned as NaN.
    """
    if tree is None:
        return np.full(len(lons), np.nan)
    pts = np.column_stack([lons, lats])
    dist, idx = tree.query(pts)
    result = np.where(dist <= max_dist_deg, depths[idx], np.nan)
    return result.astype(float)


def _extract_fine_profile(
    src_lon: npt.NDArray,
    src_lat: npt.NDArray,
    src_depth: npt.NDArray,
    src_land: npt.NDArray,
    center_lon: float,
    center_lat: float,
    direction: str,
    half_width_deg: float,
    cross_half_deg: float,
) -> dict | None:
    """Extract along-channel thalweg depth profile from the fine-resolution source.

    For direction ``"U"`` (flow is E–W, interface is N–S):
    - Sample at each longitude in [center_lon ± half_width_deg].
    - At each longitude, find the deepest wet cell within the latitude band
      [center_lat ± cross_half_deg].
    - Depth profile: depth vs longitude.

    For direction ``"V"`` (flow is N–S, interface is E–W):
    - Sample at each latitude; band is longitude.

    Returns
    -------
    dict or None
        ``lon``, ``lat`` : thalweg geographic path (1-D arrays).
        ``dist_km``      : along-channel signed distance (km), zero at centre.
        ``depth``        : thalweg depth at each sample (NaN = no wet cell).
        ``sill_depth``   : minimum depth along the profile.
        ``sill_dist_km`` : signed distance of the sill from the interface.
    """
    if direction == "U":
        # along-channel = longitude
        alo_idx = np.where(
            (src_lon >= center_lon - half_width_deg) &
            (src_lon <= center_lon + half_width_deg)
        )[0]
        crs_idx = np.where(
            (src_lat >= center_lat - cross_half_deg) &
            (src_lat <= center_lat + cross_half_deg)
        )[0]
        if len(alo_idx) < 3 or len(crs_idx) < 1:
            return None

        alo_coords = src_lon[alo_idx]    # shape (n_along,)
        crs_coords = src_lat[crs_idx]    # shape (n_cross,)

        sub_depth = src_depth[np.ix_(crs_idx, alo_idx)]   # [n_cross, n_along]
        sub_land  = src_land [np.ix_(crs_idx, alo_idx)]

        thal_depth = np.full(len(alo_idx), np.nan)
        thal_lat   = np.full(len(alo_idx), center_lat)

        for k in range(len(alo_idx)):
            wet = ~sub_land[:, k]
            if wet.any():
                vals = np.where(wet, sub_depth[:, k], -np.inf)
                best = int(np.argmax(vals))
                thal_depth[k] = sub_depth[best, k]
                thal_lat[k]   = crs_coords[best]

        thal_lon = alo_coords
        cos_lat = math.cos(math.radians(center_lat))
        dist_km = (thal_lon - center_lon) * (math.pi / 180.0) * _R_EARTH_KM * cos_lat

    else:  # "V": along-channel = latitude
        alo_idx = np.where(
            (src_lat >= center_lat - half_width_deg) &
            (src_lat <= center_lat + half_width_deg)
        )[0]
        crs_idx = np.where(
            (src_lon >= center_lon - cross_half_deg) &
            (src_lon <= center_lon + cross_half_deg)
        )[0]
        if len(alo_idx) < 3 or len(crs_idx) < 1:
            return None

        alo_coords = src_lat[alo_idx]    # shape (n_along,)
        crs_coords = src_lon[crs_idx]    # shape (n_cross,)

        sub_depth = src_depth[np.ix_(alo_idx, crs_idx)]   # [n_along, n_cross]
        sub_land  = src_land [np.ix_(alo_idx, crs_idx)]

        thal_depth = np.full(len(alo_idx), np.nan)
        thal_lon   = np.full(len(alo_idx), center_lon)

        for k in range(len(alo_idx)):
            wet = ~sub_land[k, :]
            if wet.any():
                vals = np.where(wet, sub_depth[k, :], -np.inf)
                best = int(np.argmax(vals))
                thal_depth[k] = sub_depth[k, best]
                thal_lon[k]   = crs_coords[best]

        thal_lat = alo_coords
        dist_km = (thal_lat - center_lat) * (math.pi / 180.0) * _R_EARTH_KM

    valid = np.isfinite(thal_depth)
    if valid.sum() < 3:
        return None

    sill_idx   = int(np.where(valid, thal_depth, np.inf).argmin())
    sill_depth = float(thal_depth[sill_idx])

    return {
        "lon":          thal_lon,
        "lat":          thal_lat,
        "dist_km":      dist_km,
        "depth":        thal_depth,
        "sill_depth":   sill_depth,
        "sill_dist_km": float(dist_km[sill_idx]),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_strait_thalwegs(
    src,
    dst,
    strait_records: list[dict],
    n_straits: int = 8,
    extend_coarse_cells: float = 5.0,
    cross_coarse_cells: float = 2.0,
) -> list[dict]:
    """Compute fine-vs-coarse thalweg profiles for the top *n_straits* straits.

    Parameters
    ----------
    src : xr.Dataset
        Fine-resolution source bathymetry (``lon``, ``lat``, ``depth``, ``land``).
    dst : xr.Dataset
        Coarse regridded bathymetry (``lon``, ``lat``, ``depth``, ``mask``).
    strait_records : list[dict]
        From :func:`analysis.find_straits` — sorted worst-first.  Records with
        ``_fine_lons`` / ``_fine_lats`` private keys are used to estimate the
        coarse cell size.
    n_straits : int
        Process only the first *n_straits* records (worst first).
    extend_coarse_cells : float
        Half-width of the along-channel window expressed in coarse-cell widths.
    cross_coarse_cells : float
        Half-width of the cross-channel band in coarse-cell widths.

    Returns
    -------
    list[dict]
        One entry per successfully processed strait.  Each dict has:

        ``lon``, ``lat``         – interface centre coordinates.
        ``direction``            – ``"U"`` or ``"V"``.
        ``category``             – strait category from find_straits.
        ``fine``                 – fine-resolution profile dict (lon, lat,
                                   dist_km, depth, sill_depth, sill_dist_km).
        ``coarse``               – coarse profile dict (dist_km, depth,
                                   sill_depth).
        ``sill_deficit_m``       – fine_sill − coarse_sill (positive = model
                                   too shallow).
    """
    import xarray as xr  # noqa: F401 (checked by caller)

    src_lon   = src.lon.values      # 1-D
    src_lat   = src.lat.values      # 1-D
    src_depth = src["depth"].values
    src_land  = src["land"].values

    dst_lon2d = dst.lon.values      # 2-D
    dst_lat2d = dst.lat.values
    dst_depth2d = np.where(dst["mask"].values.astype(bool),
                           dst["depth"].values, np.nan)
    dst_mask2d  = dst["mask"].values.astype(bool)

    coarse_tree, coarse_depths = _build_coarse_tree(
        dst_lon2d, dst_lat2d, dst_depth2d, dst_mask2d
    )

    # Fine-source cell sizes (for window sizing fallback)
    dlon_src = float(abs(np.diff(src_lon).mean()))
    dlat_src = float(abs(np.diff(src_lat).mean()))

    results: list[dict] = []

    for rec in strait_records[:n_straits]:
        lon_c     = float(rec["lon"])
        lat_c     = float(rec["lat"])
        direction = rec["direction"]

        # Estimate coarse cell size in degrees from the private fine-window keys
        if direction == "U":
            fine_lons = rec.get("_fine_lons")
            if fine_lons is not None and len(fine_lons) >= 2:
                cell_deg = float(fine_lons[-1] - fine_lons[0])
            else:
                cell_deg = dlon_src * 20        # rough fallback
            half_width_deg = cell_deg * extend_coarse_cells
            cross_half_deg = float(abs(np.diff(src_lat).mean())) * 20 * cross_coarse_cells
        else:
            fine_lats = rec.get("_fine_lats")
            if fine_lats is not None and len(fine_lats) >= 2:
                cell_deg = float(fine_lats[-1] - fine_lats[0])
            else:
                cell_deg = dlat_src * 20
            half_width_deg = cell_deg * extend_coarse_cells
            cross_half_deg = float(abs(np.diff(src_lon).mean())) * 20 * cross_coarse_cells

        fine = _extract_fine_profile(
            src_lon, src_lat, src_depth, src_land,
            lon_c, lat_c, direction,
            half_width_deg=half_width_deg,
            cross_half_deg=cross_half_deg,
        )
        if fine is None:
            continue

        # Sample coarse grid at thalweg positions
        coarse_dep = _sample_coarse(
            fine["lon"], fine["lat"], coarse_tree, coarse_depths,
            max_dist_deg=max(dlon_src * 20, dlat_src * 20),
        )
        c_valid = np.isfinite(coarse_dep)
        coarse_sill = float(np.nanmin(coarse_dep[c_valid])) if c_valid.any() else np.nan
        coarse_sill_dist = (
            float(fine["dist_km"][int(np.where(c_valid, coarse_dep, np.inf).argmin())])
            if c_valid.any() else np.nan
        )

        coarse = {
            "dist_km":      fine["dist_km"],
            "depth":        coarse_dep,
            "sill_depth":   coarse_sill,
            "sill_dist_km": coarse_sill_dist,
        }

        deficit = (fine["sill_depth"] - coarse_sill
                   if np.isfinite(coarse_sill) else np.nan)

        results.append({
            "lon":            lon_c,
            "lat":            lat_c,
            "direction":      direction,
            "category":       rec.get("category", ""),
            "fine":           fine,
            "coarse":         coarse,
            "sill_deficit_m": deficit,
        })

    return results


_MODE_LABEL = {
    "AUTO":     "boundary",
    "WAYPOINT": "waypoint",
}


def _mode_label(record: dict) -> str:
    cat = record.get("category", "")
    if cat in _MODE_LABEL:
        return _MODE_LABEL[cat]
    direction = record.get("direction", "")
    if direction in ("U", "V"):
        return "strait"
    return cat.lower() or "?"


def print_thalweg_table(records: list[dict], title: str = "Thalweg analysis") -> None:
    """Print a per-thalweg ASCII table to stdout.

    Columns: index, mode, name, fine sill (m), coarse sill (m), deficit (m).
    """
    if not records:
        print(f"\n{title}\n" + "-" * max(len(title), 40))
        print("  no thalwegs computed")
        print()
        return

    # Column widths
    names   = [r.get("name", f"#{i}") for i, r in enumerate(records)]
    modes   = [_mode_label(r) for r in records]
    w_name  = max(len(n) for n in names)
    w_mode  = max(len(m) for m in modes)
    w_name  = max(w_name, 4)
    w_mode  = max(w_mode, 4)

    hdr = (f"  {'#':>3}  {'mode':<{w_mode}}  {'name':<{w_name}}"
           f"  {'fine sill':>10}  {'coarse sill':>11}  {'deficit':>8}")
    sep = "  " + "-" * (len(hdr) - 2)

    print(f"\n{title}")
    print("-" * max(len(title), len(hdr)))
    print(hdr)
    print(sep)

    for i, (rec, name, mode) in enumerate(zip(records, names, modes)):
        fine_sill   = rec["fine"]["sill_depth"]
        coarse_sill = rec["coarse"]["sill_depth"]
        deficit     = rec.get("sill_deficit_m", float("nan"))

        fs  = f"{fine_sill:10.1f}" if np.isfinite(fine_sill)   else f"{'n/a':>10}"
        cs  = f"{coarse_sill:11.1f}" if np.isfinite(coarse_sill) else f"{'n/a':>11}"
        dft = f"{deficit:8.1f}"    if np.isfinite(deficit)     else f"{'n/a':>8}"

        print(f"  {i:>3}  {mode:<{w_mode}}  {name:<{w_name}}  {fs}  {cs}  {dft}")

    print()


def thalweg_summary(records: list[dict]) -> dict:
    """Return an aggregate summary dict for a list of thalweg records."""
    if not records:
        return {"thalwegs computed": 0}
    deficits = [r["sill_deficit_m"] for r in records if np.isfinite(r.get("sill_deficit_m", float("nan")))]
    return {
        "thalwegs computed":       len(records),
        "max sill deficit (m)":    f"{max(deficits):.1f}" if deficits else "n/a",
        "mean sill deficit (m)":   f"{np.mean(deficits):.1f}" if deficits else "n/a",
        "with coarse > fine sill": sum(1 for d in deficits if d < 0),
    }


def write_thalweg_csv(record: dict, csv_path: str | None = None) -> str:
    """Write thalweg path coordinates to a CSV file and return the path used.

    Columns: ``index``, ``lon``, ``lat``, ``dist_km``, ``fine_depth``,
    ``coarse_depth``.  The sill row is marked with an extra ``is_sill``
    column (1/0).  If *csv_path* is None the path is derived from the record's
    ``name`` field (spaces → underscores, → → dash).
    """
    import csv
    from pathlib import Path

    fine   = record["fine"]
    coarse = record["coarse"]
    name   = record.get("name", "thalweg")

    if csv_path is None:
        safe = (name.replace(" ", "_").replace("→", "-")
                .replace("[", "").replace("]", ""))
        csv_path = f"{safe}.csv"

    sill_dist = fine.get("sill_dist_km", float("nan"))
    sill_idx  = int(np.argmin(np.abs(fine["dist_km"] - sill_dist))) if np.isfinite(sill_dist) else -1

    n = len(fine["lon"])
    coarse_depth = coarse["depth"] if len(coarse["depth"]) == n else np.full(n, float("nan"))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "lon", "lat", "dist_km", "fine_depth", "coarse_depth", "is_sill"])
        for k in range(n):
            writer.writerow([
                k,
                f"{fine['lon'][k]:.6f}",
                f"{fine['lat'][k]:.6f}",
                f"{fine['dist_km'][k]:.3f}",
                f"{fine['depth'][k]:.2f}",
                f"{coarse_depth[k]:.2f}" if np.isfinite(coarse_depth[k]) else "",
                1 if k == sill_idx else 0,
            ])
    return csv_path


# ---------------------------------------------------------------------------
# Maximum spanning tree — max-bottleneck path (build once, query many)
# ---------------------------------------------------------------------------

def _build_bottleneck_mst(
    depth: npt.NDArray,
    mask: npt.NDArray,
) -> tuple:
    """Build a maximum spanning tree for bottleneck path queries.

    The edge weight between two adjacent wet cells is min(depth_u, depth_v) —
    the shallowest point you must pass through.  The maximum spanning tree
    (MST of negated weights) connects every wet cell such that the path
    between any two nodes in the tree is the max-bottleneck path: the route
    that maximises the minimum depth along it.

    This is O(E log E) in C via scipy and is built once per dataset, after
    which each path query is a cheap BFS on the sparse tree.

    Parameters
    ----------
    depth : ndarray [ny, nx]
        Ocean depth; 0 for land cells.
    mask : ndarray [ny, nx] bool
        Ocean mask; only True cells are included as nodes.

    Returns
    -------
    mst : scipy.sparse.csr_matrix
        Symmetric undirected MST with positive bottleneck edge weights.
    node_id : ndarray [ny, nx] int32
        Maps (row, col) → node index; -1 for land cells.
    wet_rc : ndarray [n_nodes, 2]
        Maps node index → (row, col).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    wet = mask.astype(bool)
    wet_rc = np.argwhere(wet)                                   # (n_nodes, 2)
    n_nodes = len(wet_rc)

    node_id = np.full(depth.shape, -1, dtype=np.int32)
    node_id[wet_rc[:, 0], wet_rc[:, 1]] = np.arange(n_nodes, dtype=np.int32)

    # Vectorised edge construction for 4-connected grid
    row_lists: list[npt.NDArray] = []
    col_lists: list[npt.NDArray] = []
    w_lists:   list[npt.NDArray] = []

    # Horizontal edges: (r, c) ↔ (r, c+1)
    r, c = np.where(wet[:, :-1] & wet[:, 1:])
    if len(r):
        u = node_id[r, c]
        v = node_id[r, c + 1]
        w = np.minimum(depth[r, c], depth[r, c + 1])
        row_lists += [u, v];  col_lists += [v, u];  w_lists += [w, w]

    # Vertical edges: (r, c) ↔ (r+1, c)
    r, c = np.where(wet[:-1, :] & wet[1:, :])
    if len(r):
        u = node_id[r, c]
        v = node_id[r + 1, c]
        w = np.minimum(depth[r, c], depth[r + 1, c])
        row_lists += [u, v];  col_lists += [v, u];  w_lists += [w, w]

    if not row_lists:
        from scipy.sparse import csr_matrix as _csr
        return _csr((n_nodes, n_nodes), dtype=float), node_id, wet_rc

    rows_arr = np.concatenate(row_lists)
    cols_arr = np.concatenate(col_lists)
    data_arr = np.concatenate(w_lists)

    # Negate weights so minimum_spanning_tree gives the maximum spanning tree
    adj = csr_matrix((-data_arr, (rows_arr, cols_arr)), shape=(n_nodes, n_nodes))
    mst_neg = minimum_spanning_tree(adj)        # lower-triangular CSR, weights = -w
    mst_sym = -(mst_neg + mst_neg.T)            # symmetric, weights = +w
    return mst_sym, node_id, wet_rc


def _mst_path(
    mst,
    node_id: npt.NDArray,
    wet_rc: npt.NDArray,
    start_ij: tuple[int, int],
    end_ij: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Return the max-bottleneck path between two cells using a prebuilt MST.

    The path is found by a single BFS from *start_ij* in the sparse MST,
    which is O(n_nodes) worst case but typically much cheaper.

    Parameters
    ----------
    mst : scipy.sparse.csr_matrix
        Symmetric MST from :func:`_build_bottleneck_mst`.
    node_id : ndarray [ny, nx] int32
    wet_rc  : ndarray [n_nodes, 2]
    start_ij, end_ij : (row, col)

    Returns
    -------
    list of (row, col) or None
    """
    from scipy.sparse.csgraph import breadth_first_order

    s = int(node_id[start_ij])
    e = int(node_id[end_ij])
    if s < 0 or e < 0:
        return None
    if s == e:
        return [start_ij]

    _, predecessors = breadth_first_order(
        mst, i_start=s, directed=False, return_predecessors=True,
    )

    if predecessors[e] < 0:    # -9999 sentinel: e not reachable from s
        return None

    path: list[tuple[int, int]] = []
    cur = e
    while cur != s:
        path.append((int(wet_rc[cur, 0]), int(wet_rc[cur, 1])))
        cur = int(predecessors[cur])
        if cur < 0:
            return None
    path.append((int(wet_rc[s, 0]), int(wet_rc[s, 1])))
    path.reverse()
    return path


def _path_to_profile(
    path: list[tuple[int, int]],
    depth: npt.NDArray,
    lon2d: npt.NDArray,
    lat2d: npt.NDArray,
) -> dict:
    """Convert a grid-index path to geographic thalweg profile arrays."""
    rows, cols = zip(*path)
    lons   = lon2d[rows, cols]
    lats   = lat2d[rows, cols]
    depths = depth[rows, cols]

    # Distance along path (cumulative km)
    dist_km = np.zeros(len(path))
    for k in range(1, len(path)):
        dlat = lats[k] - lats[k - 1]
        dlon = lons[k] - lons[k - 1]
        cos_lat = math.cos(math.radians(0.5 * (lats[k] + lats[k - 1])))
        dx = dlon * math.pi / 180.0 * _R_EARTH_KM * cos_lat
        dy = dlat * math.pi / 180.0 * _R_EARTH_KM
        dist_km[k] = dist_km[k - 1] + math.hypot(dx, dy)

    valid = np.isfinite(depths) & (depths > 0)
    sill = float(np.nanmin(depths[valid])) if valid.any() else np.nan
    sill_dist = float(dist_km[int(np.where(valid, depths, np.inf).argmin())]) if valid.any() else np.nan

    return {
        "lon":          np.array(lons,    dtype=float),
        "lat":          np.array(lats,    dtype=float),
        "dist_km":      dist_km,
        "depth":        np.array(depths,  dtype=float),
        "sill_depth":   sill,
        "sill_dist_km": sill_dist,
    }


def _clip_profile_to_domain(
    fine: dict,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
) -> dict:
    """Return the longest contiguous sub-path that lies within the domain box.

    The fine source is padded beyond the coarse domain so boundary-start paths
    may begin outside the domain.  Taking the longest contiguous in-domain run
    keeps the relevant portion and ensures sill statistics are computed inside
    the model domain, even if the path briefly exits and re-enters.
    """
    in_domain = (
        (fine["lon"] >= lon_min) & (fine["lon"] <= lon_max) &
        (fine["lat"] >= lat_min) & (fine["lat"] <= lat_max)
    )
    if not in_domain.any():
        return fine  # nothing inside domain — return unchanged

    # Find the longest contiguous run of in-domain points
    best_start = best_len = cur_start = cur_len = 0
    for k, inside in enumerate(in_domain):
        if inside:
            if cur_len == 0:
                cur_start = k
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    i0, i1 = best_start, best_start + best_len

    depths_c = fine["depth"][i0:i1]
    dist_c   = fine["dist_km"][i0:i1] - fine["dist_km"][i0]
    valid    = np.isfinite(depths_c) & (depths_c > 0)
    sill     = float(np.nanmin(depths_c[valid])) if valid.any() else np.nan
    sill_d   = float(dist_c[int(np.where(valid, depths_c, np.inf).argmin())]) if valid.any() else np.nan

    return {
        "lon":          fine["lon"][i0:i1],
        "lat":          fine["lat"][i0:i1],
        "dist_km":      dist_c,
        "depth":        depths_c,
        "sill_depth":   sill,
        "sill_dist_km": sill_d,
    }


# ---------------------------------------------------------------------------
# Mode B: boundary auto-detection
# ---------------------------------------------------------------------------

def _boundary_starts(
    depth: npt.NDArray,
    mask: npt.NDArray,
    lon2d: npt.NDArray,
    lat2d: npt.NDArray,
    max_inward: int = 20,
    min_segment_cells: int = 10,
) -> list[dict]:
    """Return the deepest wet cell in each open-boundary segment on each domain edge.

    Every open boundary in a structured grid is a straight line: constant *j*
    for south/north edges, constant *i* for west/east edges.  The algorithm
    exploits this constraint via a histogram:

    1. For each column *i* (south/north) or row *j* (west/east) scan inward
       from the physical edge up to *max_inward* steps and record the index of
       the first wet cell — this is the *perpendicular index* (row for S/N,
       column for W/E).

    2. Build a histogram of those perpendicular indices.  A genuine open
       boundary produces a tall bar: many columns/rows share the same
       perpendicular index.  Fjords, inlets and coastal notches scatter across
       many low-count bins.

    3. Any bin with count ≥ *min_segment_cells* is a candidate boundary line.
       Within that line collect the contributing traversal indices and split
       into contiguous runs (gaps > 1 start a new segment).

    4. For each contiguous segment return the deepest wet cell as the start.

    *max_inward* limits how far inward the scan goes: mask_region closures of
    N rows shift the effective boundary by N (e.g. skamix Kattegat south sits
    at j≈16 after j=0–15 are closed), while Norwegian fjords and other interior
    features extend 50–200 rows inward and are automatically excluded.

    Parameters
    ----------
    depth, mask, lon2d, lat2d : ndarray [ny, nx]
    max_inward : int
        Maximum rows/columns to scan inward from each physical edge.
    min_segment_cells : int
        Minimum number of columns/rows sharing the same perpendicular index
        for a boundary line to be accepted.

    Returns
    -------
    list of dicts: {edge, segment, ij, lon, lat, depth}
    """
    ny, nx = depth.shape
    starts: list[dict] = []

    # For each direction: scan inward and record (traversal_idx, perp_idx) pairs.
    # south/north: traversal = i (column), perp = j (row)
    # west/east:   traversal = j (row),    perp = i (column)
    for direction in ("west", "east", "south", "north"):
        # Build list of (traversal_idx, perp_idx) for every column/row that
        # has a wet cell within max_inward.
        hits: list[tuple[int, int]] = []   # (traversal, perp)
        if direction == "south":
            for i in range(nx):
                for j in range(min(ny, max_inward)):
                    if mask[j, i]:
                        hits.append((i, j))
                        break
        elif direction == "north":
            for i in range(nx):
                for j in range(ny - 1, max(ny - 1 - max_inward, -1), -1):
                    if mask[j, i]:
                        hits.append((i, j))
                        break
        elif direction == "west":
            for j in range(ny):
                for i in range(min(nx, max_inward)):
                    if mask[j, i]:
                        hits.append((j, i))
                        break
        else:  # east
            for j in range(ny):
                for i in range(nx - 1, max(nx - 1 - max_inward, -1), -1):
                    if mask[j, i]:
                        hits.append((j, i))
                        break

        if not hits:
            continue

        # Histogram over perpendicular index — each bin is a candidate boundary line.
        perp_to_trav: dict[int, list[int]] = defaultdict(list)
        for trav, perp in hits:
            perp_to_trav[perp].append(trav)

        seg_idx = 0
        for perp, trav_list in sorted(perp_to_trav.items()):
            if len(trav_list) < min_segment_cells:
                continue   # too few columns/rows — not a real boundary

            # Split into contiguous traversal runs (gaps > 1 → new segment).
            trav_sorted = sorted(trav_list)
            runs: list[list[int]] = [[trav_sorted[0]]]
            for t in trav_sorted[1:]:
                if t == runs[-1][-1] + 1:
                    runs[-1].append(t)
                else:
                    runs.append([t])

            for run in runs:
                if len(run) < min_segment_cells:
                    continue
                # Pick the deepest cell in this segment.
                if direction in ("south", "north"):
                    # perp=j, trav=i → cell is (j, i)
                    cells_rc = [(perp, i) for i in run]
                else:
                    # perp=i, trav=j → cell is (j, i)
                    cells_rc = [(j, perp) for j in run]
                d_vals = np.array([depth[r, c] for r, c in cells_rc])
                best   = int(np.argmax(d_vals))
                r, c   = cells_rc[best]
                starts.append({
                    "edge":    direction,
                    "segment": seg_idx,
                    "ij":      (r, c),
                    "lon":     float(lon2d[r, c]),
                    "lat":     float(lat2d[r, c]),
                    "depth":   float(depth[r, c]),
                })
                logger.debug(
                    "      boundary_starts: %s seg%d perp=%d trav=%d..%d "
                    "n=%d depth=%.1f m",
                    direction, seg_idx, perp, run[0], run[-1], len(run),
                    float(depth[r, c]),
                )
                seg_idx += 1
    return starts


def boundary_starts_from_csv(
    csv_path: str,
    depth: npt.NDArray,
    mask: npt.NDArray,
    lon2d: npt.NDArray,
    lat2d: npt.NDArray,
) -> list[dict]:
    """Detect open-boundary start cells from a lon/lat CSV file.

    The CSV defines the boundary cells of a model domain (e.g. exported from
    GETM/NEMO boundary tooling).  Each row is one boundary cell; physically
    disconnected boundaries appear as jumps in the sequence.

    Algorithm
    ---------
    1. Parse CSV, skipping header lines (any row that contains non-numeric
       tokens in the first two columns).
    2. Split into *CSV segments* wherever the spacing between consecutive
       rows exceeds 5 × the median row spacing (gap-detection).
    3. For each CSV segment, map every lon/lat to the nearest fine-grid cell
       index.
    4. Within the mapped fine-grid indices, find *wet sub-segments*: groups of
       consecutive cells where |Δrow| ≤ 2 and |Δcol| ≤ 2 (cells that are
       adjacent in index space, allowing diagonal steps) and the cell is wet.
    5. Return the deepest wet cell of each sub-segment as a start dict
       (same schema as :func:`_boundary_starts`).

    The ``edge`` label is ``bdy{i}`` for segment *i*; when a CSV segment
    splits into multiple wet sub-segments the label is ``bdy{i}_{j}``.

    Parameters
    ----------
    csv_path : str
        Path to the boundary CSV.  Expected columns: ``lon``, ``lat``
        (first two numeric columns after any header).
    depth, mask, lon2d, lat2d : ndarray
        Fine-grid arrays — same as those passed to :func:`_boundary_starts`.
    """
    import csv as _csv

    # --- 1. Parse CSV ---
    raw_lons: list[float] = []
    raw_lats: list[float] = []
    with open(csv_path, newline="") as fh:
        for row in _csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                lo = float(row[0])
                la = float(row[1])
            except ValueError:
                continue  # header or non-numeric line
            raw_lons.append(lo)
            raw_lats.append(la)

    if len(raw_lons) < 2:
        logger.warning("boundary_starts_from_csv: no data rows in %s", csv_path)
        return []

    lons_arr = np.array(raw_lons, dtype=float)
    lats_arr = np.array(raw_lats, dtype=float)

    # --- 2. Detect CSV-level segments by gap in row spacing ---
    dists = np.sqrt(np.diff(lons_arr) ** 2 + np.diff(lats_arr) ** 2)
    median_d = float(np.median(dists))
    gap_thresh = 5.0 * median_d if median_d > 0 else 1e-3
    gap_idx = np.where(dists > gap_thresh)[0]  # indices *before* the gap

    seg_bounds: list[tuple[int, int]] = []
    prev = 0
    for gi in gap_idx:
        seg_bounds.append((prev, int(gi) + 1))
        prev = int(gi) + 1
    seg_bounds.append((prev, len(lons_arr)))

    # Unique lon/lat arrays for the fine source (1-D)
    src_lon_1d = lon2d[0, :]  # first row gives unique lons (regular grid)
    src_lat_1d = lat2d[:, 0]  # first col gives unique lats

    def _nearest_ij(lo: float, la: float) -> tuple[int, int]:
        ci = int(np.argmin(np.abs(src_lon_1d - lo)))
        ri = int(np.argmin(np.abs(src_lat_1d - la)))
        return (ri, ci)

    starts: list[dict] = []

    for seg_i, (s0, s1) in enumerate(seg_bounds):
        seg_lons = lons_arr[s0:s1]
        seg_lats = lats_arr[s0:s1]
        n_pts = s1 - s0

        # --- 3. Map lon/lat → fine-grid (row, col) ---
        ijs = [_nearest_ij(seg_lons[k], seg_lats[k]) for k in range(n_pts)]

        # --- 4. Contiguous wet sub-segments on the fine grid ---
        sub_groups: list[list[int]] = []
        cur_group: list[int] = []
        for k, (ri, ci) in enumerate(ijs):
            if mask[ri, ci]:
                if not cur_group:
                    cur_group.append(k)
                else:
                    prev_ri, prev_ci = ijs[cur_group[-1]]
                    if abs(ri - prev_ri) <= 2 and abs(ci - prev_ci) <= 2:
                        cur_group.append(k)
                    else:
                        sub_groups.append(cur_group)
                        cur_group = [k]
            else:
                if cur_group:
                    sub_groups.append(cur_group)
                    cur_group = []
        if cur_group:
            sub_groups.append(cur_group)

        n_subs = len(sub_groups)
        if n_subs == 0:
            logger.debug("boundary_starts_from_csv: seg %d has no wet cells — skipped", seg_i)
            continue

        for sub_j, grp in enumerate(sub_groups):
            grp_ijs = [ijs[k] for k in grp]
            grp_d   = np.array([depth[ri, ci] for ri, ci in grp_ijs], dtype=float)
            best    = int(np.argmax(grp_d))
            ri, ci  = grp_ijs[best]

            edge_label = f"bdy{seg_i}" if n_subs == 1 else f"bdy{seg_i}_{sub_j}"
            starts.append({
                "edge":    edge_label,
                "segment": sub_j,
                "ij":      (ri, ci),
                "lon":     float(lon2d[ri, ci]),
                "lat":     float(lat2d[ri, ci]),
                "depth":   float(depth[ri, ci]),
            })
            logger.debug(
                "      boundary_starts_from_csv: %s  ij=(%d,%d)  lon=%.3f lat=%.3f  depth=%.1f m",
                edge_label, ri, ci, lon2d[ri, ci], lat2d[ri, ci], depth[ri, ci],
            )

    logger.info(
        "      boundary_starts_from_csv: %d CSV segment(s) → %d start(s)",
        len(seg_bounds), len(starts),
    )
    return starts


def boundary_thalwegs(
    src,
    dst,
    min_sill_m: float = 5.0,
    max_detour: float = 2.5,
    sill_dedup_tol_m: float = 2.0,
    boundaries_csv: str | None = None,
) -> list[dict]:
    """Auto-detect thalwegs starting from wet segments on each domain edge.

    Each edge may have multiple disconnected wet segments (e.g. due to an
    island touching the boundary).  One start cell (the deepest) is taken from
    each segment, then every cross-edge pair of starts is connected via the
    max-bottleneck (deepest-route) Dijkstra algorithm on the fine-resolution
    grid and compared against the coarse grid.

    Parameters
    ----------
    src : xr.Dataset
        Fine-resolution source bathymetry (``lon``, ``lat``, ``depth``, ``land``).
    dst : xr.Dataset
        Coarse regridded bathymetry.
    min_sill_m : float
        Discard paths whose fine-resolution sill depth is below this value.
    boundaries_csv : str or None
        Optional path to a lon/lat CSV that defines open boundary cells.  When
        given, :func:`boundary_starts_from_csv` is used instead of the default
        physical-edge detection :func:`_boundary_starts`.

    Returns
    -------
    list[dict]
        Thalweg records with keys ``name``, ``fine``, ``coarse``,
        ``sill_deficit_m``.
    """
    src_lon   = src.lon.values
    src_lat   = src.lat.values
    src_depth = np.where(src["land"].values, 0.0, src["depth"].values).astype(float)
    src_mask  = (~src["land"].values).astype(bool)

    # Build 2-D coordinate grids for the regular fine source
    lon2d_f, lat2d_f = np.meshgrid(src_lon, src_lat)

    dst_lon2d   = dst.lon.values
    dst_lat2d   = dst.lat.values
    dst_depth2d = np.where(dst["mask"].values.astype(bool),
                           dst["depth"].values, np.nan)
    dst_mask2d  = dst["mask"].values.astype(bool)

    coarse_tree, coarse_dep_arr = _build_coarse_tree(
        dst_lon2d, dst_lat2d, dst_depth2d, dst_mask2d
    )
    max_lookup = float(max(abs(np.diff(src_lon)).mean(),
                          abs(np.diff(src_lat)).mean()) * 25)

    # Coarse domain bounds (needed for both MST restriction and snap)
    _dom_lon_min = float(dst_lon2d.min())
    _dom_lon_max = float(dst_lon2d.max())
    _dom_lat_min = float(dst_lat2d.min())
    _dom_lat_max = float(dst_lat2d.max())

    # Restrict fine-grid MST to cells inside the coarse domain.
    # The fine source is padded beyond the coarse domain; without this
    # restriction the max-bottleneck algorithm finds deeper routes through the
    # padded area (e.g. the open North Sea west of a western boundary), which
    # produces paths that exit and re-enter the domain and inflate distances.
    _in_domain_f = (
        (lon2d_f >= _dom_lon_min) & (lon2d_f <= _dom_lon_max) &
        (lat2d_f >= _dom_lat_min) & (lat2d_f <= _dom_lat_max)
    )
    src_mask_mst = src_mask & _in_domain_f

    # Build MST once — all pair queries share it
    logger.info("      building max-bottleneck MST …")
    mst, node_id_f, wet_rc_f = _build_bottleneck_mst(src_depth, src_mask_mst)

    def _snap_to_fine(lo: float, la: float) -> tuple[int, int] | None:
        """Snap to the nearest wet fine-grid cell that lies inside the coarse domain.

        Uses a ±20-cell search radius to handle coasts right at the domain edge.
        Cells outside the domain are only accepted if nothing inside is found
        (rare edge case for purely coastal boundaries).
        """
        ci = int(np.argmin(np.abs(src_lon - lo)))
        ri = int(np.argmin(np.abs(src_lat - la)))
        radius = 20
        r0 = max(0, ri - radius);  r1 = min(len(src_lat), ri + radius + 1)
        c0 = max(0, ci - radius);  c1 = min(len(src_lon), ci + radius + 1)
        sub_mask = src_mask[r0:r1, c0:c1]
        sub_lon  = lon2d_f[r0:r1, c0:c1]
        sub_lat  = lat2d_f[r0:r1, c0:c1]
        sub_d    = src_depth[r0:r1, c0:c1]
        in_dom   = (
            (sub_lon >= _dom_lon_min) & (sub_lon <= _dom_lon_max) &
            (sub_lat >= _dom_lat_min) & (sub_lat <= _dom_lat_max)
        )
        # Prefer wet cells inside the domain; only fall back to outside cells
        # when the boundary position genuinely has no in-domain wet neighbors.
        sub = sub_mask & in_dom
        if not sub.any():
            sub = sub_mask
        if not sub.any():
            return None
        best = np.unravel_index(int(np.where(sub, sub_d, -np.inf).argmax()), sub.shape)
        return (r0 + int(best[0]), c0 + int(best[1]))

    if boundaries_csv:
        logger.info("      reading boundary starts from CSV: %s", boundaries_csv)
        starts = boundary_starts_from_csv(
            boundaries_csv, src_depth, src_mask, lon2d_f, lat2d_f
        )
    else:
        # Detect boundary segments on the coarse grid, then snap to fine.
        # This correctly handles cases where the fine source extends beyond
        # the coarse domain (the fine j=0 row may be all land at latitudes
        # inside the coarse domain, so direct fine-edge detection would miss
        # them; coarse-edge detection gives the right positions).
        coarse_depth_for_bdy = np.where(dst_mask2d, dst_depth2d, 0.0)
        raw_starts = _boundary_starts(coarse_depth_for_bdy, dst_mask2d,
                                      dst_lon2d, dst_lat2d)
        starts = []
        for cs in raw_starts:
            ij = _snap_to_fine(cs["lon"], cs["lat"])
            if ij is None:
                logger.info("      boundary start %s (%.3f,%.3f) — no wet fine cell nearby, skipped",
                            cs["edge"], cs["lon"], cs["lat"])
                continue
            ri, ci = ij
            starts.append({
                "edge":      cs["edge"],
                "segment":   cs["segment"],
                "coarse_ij": cs["ij"],   # (row, col) in coarse grid
                "ij":        ij,         # (row, col) in fine grid
                "lon":       float(lon2d_f[ri, ci]),
                "lat":       float(lat2d_f[ri, ci]),
                "depth":     float(src_depth[ri, ci]),
            })

    logger.info("      %d boundary start(s) on %d edge(s)",
                len(starts),
                len({s["edge"] for s in starts}))

    _SILL_DEDUP_TOL_M  = sill_dedup_tol_m
    _MAX_DETOUR        = max_detour
    seen_sill_depths: dict[frozenset[str], list[float]] = {}
    results: list[dict] = []

    logger.info("      coarse domain: lon [%.3f, %.3f] lat [%.3f, %.3f]",
                _dom_lon_min, _dom_lon_max, _dom_lat_min, _dom_lat_max)
    for s in starts:
        cij = s["coarse_ij"]
        logger.info("      start: %s seg%d  coarse_ij=(%d,%d)  lon=%.3f lat=%.3f  depth=%.1f m",
                    s["edge"], s["segment"], cij[0], cij[1],
                    s["lon"], s["lat"], s["depth"])

    for i, s1 in enumerate(starts):
        for j, s2 in enumerate(starts):
            if i >= j:
                continue

            path = _mst_path(mst, node_id_f, wet_rc_f, s1["ij"], s2["ij"])
            if path is None or len(path) < 5:
                logger.info("      skip %s→%s: no MST path (path=%s len=%d)",
                            s1["edge"], s2["edge"],
                            "None" if path is None else "found", len(path) if path else 0)
                continue

            fine = _path_to_profile(path, src_depth, lon2d_f, lat2d_f)
            fine = _clip_profile_to_domain(
                fine, _dom_lon_min, _dom_lon_max, _dom_lat_min, _dom_lat_max,
            )

            if fine["sill_depth"] < min_sill_m or len(fine["lon"]) < 2:
                logger.info("      skip %s→%s: after clip sill=%.1f m pts=%d",
                            s1["edge"], s2["edge"], fine["sill_depth"], len(fine["lon"]))
                continue

            # Reject paths that detour around the domain instead of crossing it
            direct_km = _great_circle_km(
                float(fine["lon"][0]),  float(fine["lat"][0]),
                float(fine["lon"][-1]), float(fine["lat"][-1]),
            )
            if direct_km > 0 and float(fine["dist_km"][-1]) > _MAX_DETOUR * direct_km:
                logger.info("      skip %s→%s: detour %.1f km vs %.1f km direct (max_detour=%.1f)",
                            s1["edge"], s2["edge"],
                            fine["dist_km"][-1], direct_km, _MAX_DETOUR)
                continue

            # Deduplicate by clipped sill depth (within tolerance) per edge pair
            edge_key: frozenset[str] = frozenset([s1["edge"], s2["edge"]])
            prev_sills = seen_sill_depths.setdefault(edge_key, [])
            if any(abs(fine["sill_depth"] - s) < _SILL_DEDUP_TOL_M for s in prev_sills):
                continue
            prev_sills.append(fine["sill_depth"])

            if fine["sill_depth"] < min_sill_m:
                continue

            coarse_dep = _sample_coarse(
                fine["lon"], fine["lat"], coarse_tree, coarse_dep_arr,  # type: ignore[arg-type]
                max_dist_deg=max_lookup,
            )
            c_valid = np.isfinite(coarse_dep)
            coarse_sill = float(np.nanmin(coarse_dep[c_valid])) if c_valid.any() else np.nan

            # Along-path statistics
            fine_dep_arr = np.array(fine["depth"])
            both_valid = c_valid & np.isfinite(fine_dep_arr)
            if both_valid.any():
                diff = fine_dep_arr[both_valid] - coarse_dep[both_valid]
                path_mae  = float(np.mean(np.abs(diff)))
                path_rmse = float(np.sqrt(np.mean(diff ** 2)))
            else:
                path_mae = path_rmse = float("nan")

            seg1 = s1.get("segment", 0)
            seg2 = s2.get("segment", 0)
            name = f"{s1['edge']}[{seg1}]→{s2['edge']}[{seg2}]"
            cs_str = f"{coarse_sill:.1f} m" if np.isfinite(coarse_sill) else "n/a"
            logger.info("      + %s  fine_sill=%.1f m  coarse_sill=%s  L=%.0f km",
                        name, fine["sill_depth"], cs_str, fine["dist_km"][-1])
            results.append({
                "name": name,
                "fine":  fine,
                "coarse": {
                    "dist_km":    fine["dist_km"],
                    "depth":      coarse_dep,
                    "sill_depth": coarse_sill,
                },
                "sill_deficit_m": (fine["sill_depth"] - coarse_sill
                                   if np.isfinite(coarse_sill) else np.nan),
                "path_mae_m":  path_mae,
                "path_rmse_m": path_rmse,
                "start_lon": s1["lon"],
                "start_lat": s1["lat"],
                "end_lon":   s2["lon"],
                "end_lat":   s2["lat"],
                "lon": float(0.5 * (s1["lon"] + s2["lon"])),
                "lat": float(0.5 * (s1["lat"] + s2["lat"])),
                "direction": "auto",
                "category": "AUTO",
            })

    return results


# ---------------------------------------------------------------------------
# Post-processing: add smoothed-coarse depth profiles to existing records
# ---------------------------------------------------------------------------

def add_smooth_coarse(
    records: list[dict],
    dst,
    smooth_variants: list[tuple[str, npt.NDArray]],
) -> None:
    """Sample smoothed coarse depths along thalweg paths (in-place).

    Called after rx0 smoothing to augment thalweg records with one additional
    coarse depth profile per smoothed variant, without recomputing the MST.
    The original ``record["coarse"]`` (raw pre-smoothing depths) is unchanged.
    Each smooth variant is stored as ``record["smooth_coarse"][label]``.

    Parameters
    ----------
    records : list[dict]
        Thalweg records (modified in-place).
    dst : xr.Dataset
        Coarse regridded bathymetry — mask/lon/lat used; depth field ignored.
    smooth_variants : list of (label, depth_2d)
        Each entry is a (label_string, numpy_depth_array [ny, nx]) pair.
    """
    if not records or not smooth_variants:
        return

    lon_raw = dst.lon.values
    lat_raw = dst.lat.values
    if lon_raw.ndim == 1 and lat_raw.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_raw, lat_raw)
        max_lookup = float(max(abs(np.diff(lon_raw)).mean(),
                               abs(np.diff(lat_raw)).mean()) * 25)
    else:
        lon2d, lat2d = lon_raw, lat_raw
        max_lookup = float(max(abs(np.diff(lon2d, axis=1)).mean(),
                               abs(np.diff(lat2d, axis=0)).mean()) * 25)

    mask2d = dst["mask"].values.astype(bool)

    for label, depth_arr in smooth_variants:
        depth_masked = np.where(mask2d, depth_arr, 0.0)
        tree, dep_vals = _build_coarse_tree(lon2d, lat2d, depth_masked, mask2d)
        for rec in records:
            fine = rec["fine"]
            c_dep = _sample_coarse(
                fine["lon"], fine["lat"], tree, dep_vals,  # type: ignore[arg-type]
                max_dist_deg=max_lookup,
            )
            c_valid = np.isfinite(c_dep)
            sill = float(np.nanmin(c_dep[c_valid])) if c_valid.any() else float("nan")
            fine_dep = np.array(fine["depth"])
            both = c_valid & np.isfinite(fine_dep)
            if both.any():
                diff = fine_dep[both] - c_dep[both]
                mae  = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
            else:
                mae = rmse = float("nan")
            rec.setdefault("smooth_coarse", {})[label] = {
                "dist_km":    fine["dist_km"],
                "depth":      c_dep,
                "sill_depth": sill,
                "deficit_m":  (fine["sill_depth"] - sill
                               if np.isfinite(sill) else float("nan")),
                "path_mae_m":  mae,
                "path_rmse_m": rmse,
            }


# ---------------------------------------------------------------------------
# Mode C: user-specified waypoints
# ---------------------------------------------------------------------------

def waypoint_thalwegs(
    src,
    dst,
    waypoints: list[dict],
    max_detour: float = 2.5,
) -> list[dict]:
    """Compute thalwegs along user-specified start→end waypoints.

    Each waypoint dict must have ``lon_start``, ``lat_start``, ``lon_end``,
    ``lat_end`` (geographic degrees) and optionally ``name``.  The path is
    computed on the fine-resolution grid using the max-bottleneck algorithm
    (deepest possible route between the two points).

    YAML config example::

        thalwegs:
          - name: "Main channel"
            lon_start: 3.0
            lat_start: 58.0
            lon_end: 12.0
            lat_end: 56.5
          - name: "Little Belt"
            lon_start: 9.5
            lat_start: 55.0
            lon_end: 10.5
            lat_end: 56.5

    Parameters
    ----------
    src : xr.Dataset
        Fine-resolution source bathymetry.
    dst : xr.Dataset
        Coarse regridded bathymetry.
    waypoints : list[dict]
        Each dict: lon_start, lat_start, lon_end, lat_end[, name].

    Returns
    -------
    list[dict]
        Thalweg records (same structure as other modes).
    """
    src_lon   = src.lon.values
    src_lat   = src.lat.values
    src_depth = np.where(src["land"].values, 0.0, src["depth"].values).astype(float)
    src_mask  = (~src["land"].values).astype(bool)
    lon2d_f, lat2d_f = np.meshgrid(src_lon, src_lat)

    dst_lon2d   = dst.lon.values
    dst_lat2d   = dst.lat.values
    dst_depth2d = np.where(dst["mask"].values.astype(bool),
                           dst["depth"].values, np.nan)
    dst_mask2d  = dst["mask"].values.astype(bool)
    coarse_tree, coarse_dep_arr = _build_coarse_tree(
        dst_lon2d, dst_lat2d, dst_depth2d, dst_mask2d
    )
    max_lookup = float(max(abs(np.diff(src_lon)).mean(),
                          abs(np.diff(src_lat)).mean()) * 25)

    # Build MST once — shared across all waypoint queries
    logger.info("      building max-bottleneck MST …")
    mst, node_id_f, wet_rc_f = _build_bottleneck_mst(src_depth, src_mask)
    logger.info("      %d waypoint(s) to process", len(waypoints))

    def _nearest_ij(lo: float, la: float) -> tuple[int, int] | None:
        i_lo = int(np.argmin(np.abs(src_lon - lo)))
        i_la = int(np.argmin(np.abs(src_lat - la)))
        r0 = max(0, i_la - 5);  r1 = min(len(src_lat), i_la + 6)
        c0 = max(0, i_lo - 5);  c1 = min(len(src_lon), i_lo + 6)
        sub = src_mask[r0:r1, c0:c1]
        if not sub.any():
            return None
        sub_d = src_depth[r0:r1, c0:c1]
        best = np.unravel_index(int(np.where(sub, sub_d, -np.inf).argmax()),
                                sub.shape)
        return (r0 + int(best[0]), c0 + int(best[1]))

    results: list[dict] = []
    for wp in waypoints:
        name  = str(wp.get("name", "thalweg"))
        lo0, la0 = float(wp["lon_start"]), float(wp["lat_start"])
        lo1, la1 = float(wp["lon_end"]),   float(wp["lat_end"])

        ij_start = _nearest_ij(lo0, la0)
        ij_end   = _nearest_ij(lo1, la1)
        if ij_start is None or ij_end is None:
            logger.warning("thalweg '%s': no wet cell near start or end — skipped", name)
            continue

        path = _mst_path(mst, node_id_f, wet_rc_f, ij_start, ij_end)
        if path is None or len(path) < 3:
            logger.warning("thalweg '%s': no wet path found — skipped", name)
            continue

        fine = _path_to_profile(path, src_depth, lon2d_f, lat2d_f)
        fine = _clip_profile_to_domain(
            fine,
            float(dst_lon2d.min()), float(dst_lon2d.max()),
            float(dst_lat2d.min()), float(dst_lat2d.max()),
        )
        if len(fine["lon"]) >= 2 and max_detour > 0:
            direct_km = _great_circle_km(
                float(fine["lon"][0]),  float(fine["lat"][0]),
                float(fine["lon"][-1]), float(fine["lat"][-1]),
            )
            if direct_km > 0 and float(fine["dist_km"][-1]) > max_detour * direct_km:
                logger.warning("thalweg '%s': path %.0f km vs %.0f km direct — skipped "
                               "(increase max_detour to keep)", name,
                               fine["dist_km"][-1], direct_km)
                continue
        coarse_dep = _sample_coarse(
            fine["lon"], fine["lat"], coarse_tree, coarse_dep_arr,  # type: ignore[arg-type]
            max_dist_deg=max_lookup,
        )
        c_valid = np.isfinite(coarse_dep)
        coarse_sill = float(np.nanmin(coarse_dep[c_valid])) if c_valid.any() else np.nan
        cs_str = f"{coarse_sill:.1f} m" if np.isfinite(coarse_sill) else "n/a"
        logger.info("      + %s  fine_sill=%.1f m  coarse_sill=%s  L=%.0f km",
                    name, fine["sill_depth"], cs_str, fine["dist_km"][-1])

        results.append({
            "name": name,
            "fine": fine,
            "coarse": {
                "dist_km":    fine["dist_km"],
                "depth":      coarse_dep,
                "sill_depth": coarse_sill,
            },
            "sill_deficit_m": (fine["sill_depth"] - coarse_sill
                               if np.isfinite(coarse_sill) else np.nan),
            "lon": float(0.5 * (lo0 + lo1)),
            "lat": float(0.5 * (la0 + la1)),
            "direction": "user",
            "category": "WAYPOINT",
            "zoomable": bool(wp.get("zoomable", False)),
        })

    return results
