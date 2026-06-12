"""Estuary cross-section morphology: widths, areas, and volumes along a thalweg.

For each waypoint station a perpendicular transect is cast from the thalweg
centre outward until it hits land (NaN) on each side.  Cross-sectional area
is the integral of depth over the wetted transect.  Volume per station is
area × dx, where dx = half-distance to the previous station + half-distance
to the next station (non-equidistant spacing is handled correctly).

Supports any grid type produced by the bathymetry-regrid pipeline
(regular spherical, rotated-pole, curvilinear, supergrid) because the
coordinate lookup is KDTree-based rather than assuming a regular grid.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid as _trapz
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]


_R_EARTH = 6_371_000.0  # metres


# ---------------------------------------------------------------------------
# Geodesic helpers
# ---------------------------------------------------------------------------

def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2.0 * _R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Forward bearing in degrees, clockwise from North."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlam)
    return float(np.degrees(np.arctan2(x, y)) % 360.0)


def _step(lon: float, lat: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Advance dist_m metres along bearing; return (lon2, lat2)."""
    d = dist_m / _R_EARTH
    b = np.radians(bearing_deg)
    phi1, lam1 = np.radians(lat), np.radians(lon)
    phi2 = np.arcsin(np.sin(phi1) * np.cos(d) + np.cos(phi1) * np.sin(d) * np.cos(b))
    lam2 = lam1 + np.arctan2(
        np.sin(b) * np.sin(d) * np.cos(phi1),
        np.cos(d) - np.sin(phi1) * np.sin(phi2),
    )
    return float(np.degrees(lam2)), float(np.degrees(phi2))


# ---------------------------------------------------------------------------
# Depth lookup (works for any grid type)
# ---------------------------------------------------------------------------

def build_depth_lookup(lon_2d: npt.NDArray, lat_2d: npt.NDArray, depth_2d: npt.NDArray):
    """
    Return a callable ``lookup(lon, lat) → float`` using nearest-neighbour on
    the grid.  Returns the cell depth (NaN for land) or NaN if the query point
    is outside the grid domain.
    """
    lons = lon_2d.ravel()
    lats = lat_2d.ravel()
    depths = depth_2d.ravel().copy()

    # Scale lon by cos(mean_lat) so that E-W and N-S distances are comparable
    cos_lat = float(np.cos(np.radians(float(np.nanmean(lats)))))

    pts = np.column_stack([lons * cos_lat, lats])
    tree = cKDTree(pts)

    # Typical cell spacing (scaled) — used to reject queries outside the domain
    dlat_typ = (
        float(np.nanmean(np.abs(np.diff(lat_2d[:, 0])))) if lat_2d.shape[0] > 1 else 0.1
    )
    dlon_typ = (
        float(np.nanmean(np.abs(np.diff(lon_2d[0, :])))) if lon_2d.shape[1] > 1 else 0.1
    )
    max_dist = 2.0 * max(dlat_typ, dlon_typ * cos_lat)

    def lookup(lon: float, lat: float) -> float:
        _, idx = tree.query([[lon * cos_lat, lat]], distance_upper_bound=max_dist)
        if idx[0] >= len(depths):  # outside domain
            return float("nan")
        return float(depths[idx[0]])

    return lookup


# ---------------------------------------------------------------------------
# Cross-section computation
# ---------------------------------------------------------------------------

def _cast_ray(
    lon_c: float,
    lat_c: float,
    bearing: float,
    lookup,
    sample_ds: float,
    max_half_width: float,
) -> tuple[list[float], list[float]]:
    """
    Step outward from (lon_c, lat_c) along *bearing* in increments of sample_ds.
    Stop at the first land (NaN) cell or after max_half_width metres.
    Returns (distances_m, depths).
    """
    dists: list[float] = []
    deps: list[float] = []
    max_steps = int(max_half_width / sample_ds) + 2
    for k in range(1, max_steps + 1):
        dist = k * sample_ds
        lo, la = _step(lon_c, lat_c, bearing, dist)
        d = lookup(lo, la)
        if np.isnan(d):
            break
        dists.append(dist)
        deps.append(d)
    return dists, deps


def compute_cross_section(
    lon_c: float,
    lat_c: float,
    along_bearing: float,
    lookup,
    sample_ds: float = 50.0,
    max_half_width: float = 10_000.0,
) -> dict:
    """
    Compute one cross-section centred at (lon_c, lat_c), perpendicular to
    along_bearing.  The transect extends outward on each side until the first
    land cell is encountered.

    Returns a dict with:
      area_m2, width_m, left_m, right_m,
      left_bank (lon, lat), right_bank (lon, lat),
      full_dists (m from centre, negative = left), full_depths.
    """
    perp_r = (along_bearing + 90.0) % 360.0
    perp_l = (along_bearing - 90.0) % 360.0

    dists_r, deps_r = _cast_ray(lon_c, lat_c, perp_r, lookup, sample_ds, max_half_width)
    dists_l, deps_l = _cast_ray(lon_c, lat_c, perp_l, lookup, sample_ds, max_half_width)

    center_d = lookup(lon_c, lat_c)
    center_d = max(0.0, float(center_d)) if not np.isnan(center_d) else 0.0

    # Build transect: left side (distances negative), centre, right side
    full_dists = np.array(
        [-d for d in reversed(dists_l)] + [0.0] + list(dists_r), dtype=float
    )
    full_depths = np.maximum(
        np.array(list(reversed(deps_l)) + [center_d] + deps_r, dtype=float), 0.0
    )

    area = float(_trapz(full_depths, full_dists)) if len(full_dists) > 1 else 0.0
    width = float(full_dists[-1] - full_dists[0]) if len(full_dists) > 1 else 0.0

    # Geographic positions of the bank endpoints
    if dists_l:
        lbank = _step(lon_c, lat_c, perp_l, dists_l[-1])
    else:
        lbank = (lon_c, lat_c)
    if dists_r:
        rbank = _step(lon_c, lat_c, perp_r, dists_r[-1])
    else:
        rbank = (lon_c, lat_c)

    return {
        "area_m2": area,
        "width_m": width,
        "left_m": float(dists_l[-1]) if dists_l else 0.0,
        "right_m": float(dists_r[-1]) if dists_r else 0.0,
        "left_bank": lbank,
        "right_bank": rbank,
        "full_dists": full_dists,
        "full_depths": full_depths,
    }


# ---------------------------------------------------------------------------
# Full branch morphology
# ---------------------------------------------------------------------------

def compute_morphology(
    waypoints: list[tuple[float, float]],
    lookup,
    sample_ds: float = 50.0,
    max_half_width: float = 10_000.0,
) -> list[dict]:
    """
    Compute cross-sections, areas, and volumes for a list of (lon, lat) waypoints.

    Waypoints are the stations; spacing need not be equidistant.
    Volume per station = area × dx, where dx = half-distance to neighbour on
    each side.  Cumulative volume is summed from the first waypoint (mouth).

    Returns a list of station dicts, one per waypoint.
    """
    n = len(waypoints)
    if n < 2:
        raise ValueError("Need at least 2 waypoints per branch")

    # Along-channel segment distances and cumulative coordinate s
    seg_dists = [haversine_m(*waypoints[i - 1], *waypoints[i]) for i in range(1, n)]
    s = np.concatenate([[0.0], np.cumsum(seg_dists)])

    stations: list[dict] = []
    for i, (lon_c, lat_c) in enumerate(waypoints):
        # Along-channel direction: use prev→next for interior, endpoint segment elsewhere
        if i == 0:
            along_b = _bearing(lon_c, lat_c, *waypoints[i + 1])
        elif i == n - 1:
            along_b = _bearing(*waypoints[i - 1], lon_c, lat_c)
        else:
            along_b = _bearing(*waypoints[i - 1], *waypoints[i + 1])

        cs = compute_cross_section(lon_c, lat_c, along_b, lookup, sample_ds, max_half_width)

        # dx = half-distance to prev + half-distance to next
        dx = 0.0
        if i > 0:
            dx += seg_dists[i - 1] / 2.0
        if i < n - 1:
            dx += seg_dists[i] / 2.0

        stations.append({
            "lon": lon_c,
            "lat": lat_c,
            "s_m": float(s[i]),
            "along_bearing": along_b,
            "width_m": cs["width_m"],
            "left_m": cs["left_m"],
            "right_m": cs["right_m"],
            "area_m2": cs["area_m2"],
            "dx_m": dx,
            "volume_m3": cs["area_m2"] * dx,
            "left_bank": cs["left_bank"],
            "right_bank": cs["right_bank"],
            "full_dists": cs["full_dists"],
            "full_depths": cs["full_depths"],
        })

    # Cumulative volume from first waypoint (mouth end)
    cumvol = 0.0
    for st in stations:
        cumvol += st["volume_m3"]
        st["cumvol_m3"] = cumvol

    return stations
