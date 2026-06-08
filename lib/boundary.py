"""Write open-boundary coordinate files from a regridded bathymetry dataset.

Four sides are supported: west, north, east, south.  Corner ownership:
west and east own the corner cells; north and south start/end one cell
inside (i=1 and i=nx-2), so no cell appears on two boundaries.

For each position along a boundary edge the code scans inward from the
outer edge and takes the first wet cell.  Positions are traversed in a
fixed order regardless of which side:
  west / east  — south to north  (j = 0 … ny-1)
  north / south — west  to east  (i = 0 … nx-1, corners excluded)

Grid index convention: i = x (longitude column), j = y (latitude row),
consistent with ncview.  The underlying array is indexed [j, i].

Output format per file::

    T-grid
    lon,lat
    lon1,lat1
    …

One file per contiguous wet segment.  If only one segment exists the file
is named ``<prefix>_bdy_west.csv``; multiple segments get a numeric suffix
``_bdy_west_01.csv``, ``_bdy_west_02.csv``, etc.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_wet_inward(mask, fixed_idx, scan_range, axis):
    """Return the first wet cell scanning inward along *axis*.

    Parameters
    ----------
    mask : 2-D bool array, shape (ny, nx)
    fixed_idx : int
        The index that stays fixed (j for west/east, i for north/south).
    scan_range : iterable of int
        Indices to scan in inward order (e.g. range(0, nx) for west side).
    axis : 'i' | 'j'
        Which axis is being scanned.

    Returns
    -------
    (i, j) or None
    """
    for idx in scan_range:
        if axis == "i":          # scanning across longitude columns
            i, j = idx, fixed_idx
        else:                    # scanning across latitude rows
            i, j = fixed_idx, idx
        if mask[j, i]:
            return (i, j)
    return None


def _find_boundary_cells(mask, side):
    """Return ordered (i, j) list for *side*, one entry per traversal position.

    Positions with no wet cell at all are omitted (pure-land rows/columns).
    """
    ny, nx = mask.shape
    cells = []

    if side == "west":
        for j in range(ny):
            c = _first_wet_inward(mask, j, range(nx), axis="i")
            if c is not None:
                cells.append(c)

    elif side == "east":
        for j in range(ny):
            c = _first_wet_inward(mask, j, range(nx - 1, -1, -1), axis="i")
            if c is not None:
                cells.append(c)

    elif side == "north":
        # corners owned by west/east → i = 1 … nx-2
        for i in range(1, nx - 1):
            c = _first_wet_inward(mask, i, range(ny - 1, -1, -1), axis="j")
            if c is not None:
                cells.append(c)

    elif side == "south":
        # corners owned by west/east → i = 1 … nx-2
        for i in range(1, nx - 1):
            c = _first_wet_inward(mask, i, range(ny), axis="j")
            if c is not None:
                cells.append(c)

    else:
        raise ValueError(f"Unknown boundary side {side!r}")

    return cells


def _split_contiguous(cells, key_fn):
    """Split a list of (i,j) cells into contiguous groups.

    Two cells are contiguous when their *key_fn* values differ by exactly 1.
    """
    if not cells:
        return []
    groups: list[list] = [[cells[0]]]
    for cell in cells[1:]:
        if key_fn(cell) == key_fn(groups[-1][-1]) + 1:
            groups[-1].append(cell)
        else:
            groups.append([cell])
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_boundary_coords(dst, report_dir: str, name_prefix: str) -> list[str]:
    """Write open-boundary T-grid coordinate CSV files.

    Parameters
    ----------
    dst : xr.Dataset
        Regridded bathymetry with variables ``mask``, ``depth`` and 1-D or
        2-D coordinates ``lon``, ``lat``.
    report_dir : str
        Directory where the CSV files are written.
    name_prefix : str
        Short identifier prepended to every file name.

    Returns
    -------
    list[str]
        Paths of all files written (relative file names within *report_dir*).
    """
    mask = dst["mask"].values.astype(bool)   # shape (ny, nx)

    lon_arr = dst.lon.values
    lat_arr = dst.lat.values

    def _lon(i, j):
        if lon_arr.ndim == 1:
            return float(lon_arr[i])
        return float(lon_arr[j, i])

    def _lat(i, j):
        if lat_arr.ndim == 1:
            return float(lat_arr[j])
        return float(lat_arr[j, i])

    # key_fn maps (i, j) → the traversal index used to detect contiguity
    key_fns = {
        "west":  lambda c: c[1],   # j varies south→north
        "east":  lambda c: c[1],
        "north": lambda c: c[0],   # i varies west→east
        "south": lambda c: c[0],
    }

    written: list[str] = []
    for side in ("west", "north", "east", "south"):
        cells = _find_boundary_cells(mask, side)
        segments = _split_contiguous(cells, key_fns[side])

        multi = len(segments) > 1
        for k, seg in enumerate(segments):
            suffix = f"_{k + 1:02d}" if multi else ""
            fname = f"{name_prefix}_bdy_{side}{suffix}.csv"
            fpath = os.path.join(report_dir, fname)
            with open(fpath, "w") as f:
                f.write("T-grid\nlon,lat\n")
                for (i, j) in seg:
                    f.write(f"{_lon(i, j):.5f},{_lat(i, j):.5f}\n")
            written.append(fname)

    return written
