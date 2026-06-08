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

All four sides are written to a single file ``<prefix>_bdy.csv`` in the
order west (S→N), north (W→E), east (S→N), south (W→E).
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_boundary_cells(mask, side):
    """Return ordered (i, j) list of wet cells on the grid edge for *side*.

    Only the outermost column/row is checked — no inward scanning.  This
    keeps longitude constant for west/east and latitude constant for
    north/south.  Land edge cells are simply skipped.

    Corner ownership: west and east include the corner cells (full j range);
    north and south skip the first and last columns (i = 1 … nx-2).
    """
    ny, nx = mask.shape
    cells = []

    if side == "west":
        for j in range(ny):
            if mask[j, 0]:
                cells.append((0, j))

    elif side == "east":
        for j in range(ny):
            if mask[j, nx - 1]:
                cells.append((nx - 1, j))

    elif side == "north":
        for i in range(1, nx - 1):
            if mask[ny - 1, i]:
                cells.append((i, ny - 1))

    elif side == "south":
        for i in range(1, nx - 1):
            if mask[0, i]:
                cells.append((i, 0))

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
        Single-element list with the relative file name written.
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

    fname = f"{name_prefix}_bdy.csv"
    fpath = os.path.join(report_dir, fname)
    with open(fpath, "w") as f:
        f.write("T-grid\nlon,lat\n")
        for side in ("west", "north", "east", "south"):
            cells = _find_boundary_cells(mask, side)
            for (i, j) in cells:
                f.write(f"{_lon(i, j):.5f},{_lat(i, j):.5f}\n")

    written = [fname]

    return written
