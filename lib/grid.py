"""Grid definitions for the bathymetry interpolation pipeline.

Three concrete grid types share a common BaseGrid interface that exposes
corner and centre coordinates in geographic (lon/lat) degrees, as required
by ESMF.  All corner arrays have shape [ny+1, nx+1] and all centre arrays
have shape [ny, nx] following the numpy/xarray [row, col] convention.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt


class BaseGrid(ABC):
    """Abstract base for all target grid types."""

    @property
    @abstractmethod
    def corner_lon(self) -> npt.NDArray[np.float64]:
        """Longitude of cell corners, shape [ny+1, nx+1], degrees East."""

    @property
    @abstractmethod
    def corner_lat(self) -> npt.NDArray[np.float64]:
        """Latitude of cell corners, shape [ny+1, nx+1], degrees North."""

    @property
    @abstractmethod
    def center_lon(self) -> npt.NDArray[np.float64]:
        """Longitude of cell centres, shape [ny, nx], degrees East."""

    @property
    @abstractmethod
    def center_lat(self) -> npt.NDArray[np.float64]:
        """Latitude of cell centres, shape [ny, nx], degrees North."""

    @property
    def nx(self) -> int:
        return self.center_lon.shape[1]

    @property
    def ny(self) -> int:
        return self.center_lon.shape[0]

    @property
    def lon_bounds(self) -> tuple[float, float]:
        return float(self.corner_lon.min()), float(self.corner_lon.max())

    @property
    def lat_bounds(self) -> tuple[float, float]:
        return float(self.corner_lat.min()), float(self.corner_lat.max())

    def summary(self) -> dict:
        return {
            "type": type(self).__name__,
            "nx": self.nx,
            "ny": self.ny,
            "lon_min": f"{float(self.center_lon.min()):.4f}",
            "lon_max": f"{float(self.center_lon.max()):.4f}",
            "lat_min": f"{float(self.center_lat.min()):.4f}",
            "lat_max": f"{float(self.center_lat.max()):.4f}",
        }


def _rotate_spherical(
    lon: npt.NDArray,
    lat: npt.NDArray,
    lon0: float,
    lat0: float,
    angle_deg: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Rotate (lon, lat) points around (lon0, lat0) by angle_deg degrees CCW.

    Uses a local tangent-plane approximation: projects to (east, north) offsets
    in metres, applies a 2D rotation, and converts back.  Accurate for domains
    up to ~2 000 km across.
    """
    cos_lat0 = np.cos(np.radians(lat0))
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Local offsets (degrees, scaled so that one degree is the same in both axes)
    dx = (lon - lon0) * cos_lat0
    dy = lat - lat0

    dx_r = cos_t * dx - sin_t * dy
    dy_r = sin_t * dx + cos_t * dy

    return lon0 + dx_r / cos_lat0, lat0 + dy_r


class SphericalGrid(BaseGrid):
    """Regular or rotated spherical (lon/lat) grid.

    Parameters
    ----------
    lon_min, lon_max : float
        Western/eastern boundary in degrees East.
    lat_min, lat_max : float
        Southern/northern boundary in degrees North.
    dlon, dlat : float
        Cell size in degrees.
    rotation_deg : float
        Rotation of the grid axes counter-clockwise around the grid centre, in
        degrees.  A non-zero value produces a curvilinear corner-coordinate
        array even though the underlying resolution is regular.
    interfaces : bool
        When *True*, lon_min/lat_min/lon_max/lat_max are the positions of the
        outermost cell corners (interfaces).  T-points are offset inward by
        half a cell.
        When *False* (default), lon_min/lat_min/lon_max/lat_max are the
        positions of the first and last T-points (cell centres).  Corner
        arrays are offset outward by half a cell on every side.
    """

    def __init__(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        dlon: float,
        dlat: float,
        rotation_deg: float = 0.0,
        interfaces: bool = False,
    ) -> None:
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlon = dlon
        self.dlat = dlat
        self.rotation_deg = rotation_deg
        self.interfaces = interfaces
        self._build()

    def _build(self) -> None:
        if self.interfaces:
            # lon_min/lat_min are corner positions; T-points are offset inward
            nx = round((self.lon_max - self.lon_min) / self.dlon)
            ny = round((self.lat_max - self.lat_min) / self.dlat)
            lon_c1d = self.lon_min + np.arange(nx + 1) * self.dlon
            lat_c1d = self.lat_min + np.arange(ny + 1) * self.dlat
        else:
            # lon_min/lat_min are first T-point; lon_max/lat_max are last T-point
            nx = round((self.lon_max - self.lon_min) / self.dlon) + 1
            ny = round((self.lat_max - self.lat_min) / self.dlat) + 1
            lon_c1d = (self.lon_min - 0.5 * self.dlon) + np.arange(nx + 1) * self.dlon
            lat_c1d = (self.lat_min - 0.5 * self.dlat) + np.arange(ny + 1) * self.dlat
        lon_c, lat_c = np.meshgrid(lon_c1d, lat_c1d)  # [ny+1, nx+1]

        if self.rotation_deg != 0.0:
            lon0 = (self.lon_min + self.lon_max) / 2.0
            lat0 = (self.lat_min + self.lat_max) / 2.0
            lon_c, lat_c = _rotate_spherical(lon_c, lat_c, lon0, lat0, self.rotation_deg)

        self._corner_lon: npt.NDArray[np.float64] = np.asarray(lon_c, dtype=np.float64)
        self._corner_lat: npt.NDArray[np.float64] = np.asarray(lat_c, dtype=np.float64)
        self._center_lon: npt.NDArray[np.float64] = np.asarray(0.25 * (
            lon_c[:-1, :-1] + lon_c[:-1, 1:] + lon_c[1:, :-1] + lon_c[1:, 1:]
        ), dtype=np.float64)
        self._center_lat: npt.NDArray[np.float64] = np.asarray(0.25 * (
            lat_c[:-1, :-1] + lat_c[:-1, 1:] + lat_c[1:, :-1] + lat_c[1:, 1:]
        ), dtype=np.float64)

    @property
    def corner_lon(self) -> npt.NDArray[np.float64]:
        return self._corner_lon

    @property
    def corner_lat(self) -> npt.NDArray[np.float64]:
        return self._corner_lat

    @property
    def center_lon(self) -> npt.NDArray[np.float64]:
        return self._center_lon

    @property
    def center_lat(self) -> npt.NDArray[np.float64]:
        return self._center_lat

    def summary(self) -> dict:
        d = super().summary()
        # Physical resolution in km at the domain centre
        lat_min = float(self.center_lat.min())
        lat_max = float(self.center_lat.max())
        R = 6371.0
        dy_km = self.dlat * math.pi / 180.0 * R
        dx_km_s = self.dlon * math.pi / 180.0 * R * math.cos(math.radians(lat_max))
        dx_km_n = self.dlon * math.pi / 180.0 * R * math.cos(math.radians(lat_min))
        dx_lo, dx_hi = min(dx_km_s, dx_km_n), max(dx_km_s, dx_km_n)
        if abs(dx_hi - dx_lo) < 0.05 * dy_km:
            res_km = f"Δx≈Δy≈{dy_km:.2f} km"
        else:
            res_km = f"Δy≈{dy_km:.2f} km,  Δx {dx_lo:.2f}–{dx_hi:.2f} km"
        d.update({
            "dlon": self.dlon,
            "dlat": self.dlat,
            "resolution_km": res_km,
            "rotation_deg": self.rotation_deg,
            "coord_convention": "interfaces" if self.interfaces else "T-points",
        })
        return d


class CartesianGrid(BaseGrid):
    """Regular or rotated Cartesian (projected) grid.

    Parameters
    ----------
    x_min, x_max : float
        Western and eastern boundaries in projected units (e.g. metres).
    y_min, y_max : float
        Southern and northern boundaries in projected units.
    dx, dy : float
        Cell size in projected units.
    crs : str
        Proj / EPSG string for the projection (e.g. ``"EPSG:32632"``).
    rotation_deg : float
        Rotation of the grid axes CCW around the grid centre, in degrees.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        dx: float,
        dy: float,
        crs: str,
        rotation_deg: float = 0.0,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dx = dx
        self.dy = dy
        self.crs = crs
        self.rotation_deg = rotation_deg
        self._build()

    def _build(self) -> None:
        import pyproj

        nx = round((self.x_max - self.x_min) / self.dx)
        ny = round((self.y_max - self.y_min) / self.dy)

        x_c1d = self.x_min + np.arange(nx + 1) * self.dx
        y_c1d = self.y_min + np.arange(ny + 1) * self.dy
        x_c, y_c = np.meshgrid(x_c1d, y_c1d)  # [ny+1, nx+1]

        if self.rotation_deg != 0.0:
            x0 = (self.x_min + self.x_max) / 2.0
            y0 = (self.y_min + self.y_max) / 2.0
            theta = np.radians(self.rotation_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dxc, dyc = x_c - x0, y_c - y0
            x_c = x0 + cos_t * dxc - sin_t * dyc
            y_c = y0 + sin_t * dxc + cos_t * dyc

        transformer = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon_c, lat_c = transformer.transform(x_c, y_c)

        self._corner_lon = np.asarray(lon_c, dtype=np.float64)
        self._corner_lat = np.asarray(lat_c, dtype=np.float64)
        self._center_lon = np.asarray(0.25 * (
            lon_c[:-1, :-1] + lon_c[:-1, 1:] + lon_c[1:, :-1] + lon_c[1:, 1:]
        ), dtype=np.float64)
        self._center_lat = np.asarray(0.25 * (
            lat_c[:-1, :-1] + lat_c[:-1, 1:] + lat_c[1:, :-1] + lat_c[1:, 1:]
        ), dtype=np.float64)

    @property
    def corner_lon(self) -> npt.NDArray[np.float64]:
        return self._corner_lon

    @property
    def corner_lat(self) -> npt.NDArray[np.float64]:
        return self._corner_lat

    @property
    def center_lon(self) -> npt.NDArray[np.float64]:
        return self._center_lon

    @property
    def center_lat(self) -> npt.NDArray[np.float64]:
        return self._center_lat

    def summary(self) -> dict:
        d = super().summary()
        d.update({
            "dx": self.dx,
            "dy": self.dy,
            "crs": self.crs,
            "rotation_deg": self.rotation_deg,
        })
        return d


def _rot2geo(
    rlon: npt.NDArray,
    rlat: npt.NDArray,
    pole_lon: float,
    pole_lat: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Inverse rotated-pole transform: (rlon, rlat) → geographic (lon, lat).

    CF convention: pole_lon/pole_lat is the geographic location of the
    rotated North Pole.  The transform is exact (no small-angle approx).
    """
    rr = np.radians(rlon)
    lr = np.radians(rlat)
    pp = np.radians(pole_lat)

    sin_lat = np.sin(pp) * np.sin(lr) + np.cos(pp) * np.cos(lr) * np.cos(rr)
    lat = np.degrees(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))

    cos_lat = np.cos(np.radians(lat))
    safe = cos_lat > 1e-10
    sn = np.where(safe, np.cos(lr) * np.sin(rr) / cos_lat, 0.0)
    cs = np.where(safe,
                  (np.cos(pp) * np.sin(lr)
                   - np.sin(pp) * np.cos(lr) * np.cos(rr)) / cos_lat,
                  np.sign(np.cos(pp)))
    lon = (pole_lon + np.degrees(np.arctan2(sn, cs)) + 180.0) % 360.0 - 180.0
    return lon, lat


def pole_from_center(lon0: float, lat0: float) -> tuple[float, float]:
    """Return the rotated-pole location that places (lon0, lat0) at rlon=0, rlat=0.

    This puts the domain centre exactly on the rotated equator, maximising
    cell equidistance across the domain.

    Returns
    -------
    pole_lon, pole_lat : float
        Geographic longitude and latitude of the rotated North Pole.
    """
    pole_lat = 90.0 - lat0
    pole_lon = (lon0 - 180.0 + 180.0) % 360.0 - 180.0
    return pole_lon, pole_lat


class RotatedPoleGrid(BaseGrid):
    """Rotated-pole spherical grid (CF convention).

    The grid is defined as a regular (rlon, rlat) mesh in rotated coordinates
    and back-transformed to geographic (lon, lat) using the exact spherical
    rotation.  Cells are equidistant by construction near the rotated equator.

    Parameters
    ----------
    pole_lon, pole_lat : float
        Geographic location of the rotated North Pole (degrees).
        Use ``pole_from_center(lon0, lat0)`` to compute this automatically
        for a domain centred at (lon0, lat0).
    rlon_min, rlon_max : float
        Domain extent in rotated longitude (degrees).
    rlat_min, rlat_max : float
        Domain extent in rotated latitude (degrees).
    drot : float
        Cell spacing in rotated degrees (same for both axes — equidistant).
    axis_rotation_deg : float
        Optional additional CCW rotation of the grid axes within the rotated
        system.  Useful for aligning the grid with a coastline or matching
        an existing model configuration.  Default 0.0.
    """

    def __init__(
        self,
        pole_lon: float,
        pole_lat: float,
        rlon_min: float,
        rlon_max: float,
        rlat_min: float,
        rlat_max: float,
        drot: float,
        axis_rotation_deg: float = 0.0,
    ) -> None:
        self.pole_lon = pole_lon
        self.pole_lat = pole_lat
        self.rlon_min = rlon_min
        self.rlon_max = rlon_max
        self.rlat_min = rlat_min
        self.rlat_max = rlat_max
        self.drot = drot
        self.axis_rotation_deg = axis_rotation_deg
        self._build()

    def _build(self) -> None:
        nx = round((self.rlon_max - self.rlon_min) / self.drot)
        ny = round((self.rlat_max - self.rlat_min) / self.drot)

        rlon_1d = self.rlon_min + np.arange(nx + 1) * self.drot
        rlat_1d = self.rlat_min + np.arange(ny + 1) * self.drot
        rlon_c, rlat_c = np.meshgrid(rlon_1d, rlat_1d)   # [ny+1, nx+1]

        if self.axis_rotation_deg != 0.0:
            th = np.radians(self.axis_rotation_deg)
            c, s = np.cos(th), np.sin(th)
            rlon_r = c * rlon_c - s * rlat_c
            rlat_r = s * rlon_c + c * rlat_c
        else:
            rlon_r, rlat_r = rlon_c, rlat_c

        lon_c, lat_c = _rot2geo(rlon_r, rlat_r, self.pole_lon, self.pole_lat)

        self._corner_lon = np.asarray(lon_c, dtype=np.float64)
        self._corner_lat = np.asarray(lat_c, dtype=np.float64)
        self._center_lon = np.asarray(0.25 * (
            lon_c[:-1, :-1] + lon_c[:-1, 1:] + lon_c[1:, :-1] + lon_c[1:, 1:]
        ), dtype=np.float64)
        self._center_lat = np.asarray(0.25 * (
            lat_c[:-1, :-1] + lat_c[:-1, 1:] + lat_c[1:, :-1] + lat_c[1:, 1:]
        ), dtype=np.float64)

        # Local rotation angle α: angle of model x-axis relative to geographic East.
        # Derived from the direction of the right-hand cell edge in geographic space.
        # Useful for vector rotation of atmospheric forcing (wind stress etc.).
        dlon = lon_c[:-1, 1:] - lon_c[:-1, :-1]   # [ny, nx]  approximate
        dlat = lat_c[:-1, 1:] - lat_c[:-1, :-1]
        cos_lat_c = np.cos(np.radians(self._center_lat))
        self._axis_angle = np.arctan2(dlat, dlon * cos_lat_c)   # radians

    @property
    def corner_lon(self) -> npt.NDArray[np.float64]:
        return self._corner_lon

    @property
    def corner_lat(self) -> npt.NDArray[np.float64]:
        return self._corner_lat

    @property
    def center_lon(self) -> npt.NDArray[np.float64]:
        return self._center_lon

    @property
    def center_lat(self) -> npt.NDArray[np.float64]:
        return self._center_lat

    @property
    def axis_angle(self) -> npt.NDArray[np.float64]:
        """Local angle (radians) of model x-axis relative to geographic East.

        Shape [ny, nx].  Used to rotate atmospheric vector forcing from
        geographic (East, North) into model (u, v) grid coordinates:
            u_model =  u_geo * cos(α) + v_geo * sin(α)
            v_model = −u_geo * sin(α) + v_geo * cos(α)
        """
        return self._axis_angle

    def summary(self) -> dict:
        d = super().summary()
        d.update({
            "pole_lon": self.pole_lon,
            "pole_lat": self.pole_lat,
            "rlon_min": self.rlon_min,
            "rlon_max": self.rlon_max,
            "rlat_min": self.rlat_min,
            "rlat_max": self.rlat_max,
            "drot": self.drot,
            "axis_rotation_deg": self.axis_rotation_deg,
        })
        return d


class SuperGrid(BaseGrid):
    """Grid read from a supergrid NetCDF file (MOM6 ocean_hgrid.nc / pyGETM).

    A supergrid has shape ``(2·ny+1) × (2·nx+1)`` and interleaves all four
    Arakawa C-grid staggered positions in a single array:

        Row\\Col  even         odd
        even     Q (corner)   V (N/S face)
        odd      U (E/W face) T (centre)

    The ESMF conservative regridder requires T-point centres and Q-point corners:

    =========  ======================  ===========
    Position   Supergrid slice         Shape
    =========  ======================  ===========
    T-centre   ``[1::2, 1::2]``        ny × nx
    U (E/W)    ``[1::2, 0::2]``        ny × (nx+1)
    V (N/S)    ``[0::2, 1::2]``        (ny+1) × nx
    Q-corner   ``[0::2, 0::2]``        (ny+1) × (nx+1)
    =========  ======================  ===========

    Parameters
    ----------
    path : str | Path
        Path to the supergrid NetCDF file.
    x_var, y_var : str
        Variable names for longitude and latitude in the file.
        MOM6 uses ``"x"`` / ``"y"``; pyGETM may use ``"lon"`` / ``"lat"``.
        Auto-detected from the file if not supplied.
    """

    def __init__(
        self,
        path: "str | Path",
        x_var: str = "",
        y_var: str = "",
    ) -> None:
        from pathlib import Path
        import xarray as xr

        ds = xr.open_dataset(Path(path))
        if not x_var:
            x_var = next(v for v in ("x", "lon", "longitude") if v in ds)
        if not y_var:
            y_var = next(v for v in ("y", "lat", "latitude") if v in ds)
        sg_x = ds[x_var].values.astype(np.float64)
        sg_y = ds[y_var].values.astype(np.float64)
        ds.close()

        if sg_x.ndim != 2 or sg_x.shape[0] % 2 == 0 or sg_x.shape[1] % 2 == 0:
            raise ValueError(
                f"Expected a supergrid with odd dimensions (2·ny+1) × (2·nx+1), "
                f"got {sg_x.shape}.  Check x_var/y_var."
            )

        self._center_lon: npt.NDArray[np.float64] = sg_x[1::2, 1::2]
        self._center_lat: npt.NDArray[np.float64] = sg_y[1::2, 1::2]
        self._corner_lon: npt.NDArray[np.float64] = sg_x[0::2, 0::2]
        self._corner_lat: npt.NDArray[np.float64] = sg_y[0::2, 0::2]

        # Keep U/V coordinate slices for downstream use
        self.u_lon: npt.NDArray[np.float64] = sg_x[1::2, 0::2]
        self.u_lat: npt.NDArray[np.float64] = sg_y[1::2, 0::2]
        self.v_lon: npt.NDArray[np.float64] = sg_x[0::2, 1::2]
        self.v_lat: npt.NDArray[np.float64] = sg_y[0::2, 1::2]

    @property
    def corner_lon(self) -> npt.NDArray[np.float64]:
        return self._corner_lon

    @property
    def corner_lat(self) -> npt.NDArray[np.float64]:
        return self._corner_lat

    @property
    def center_lon(self) -> npt.NDArray[np.float64]:
        return self._center_lon

    @property
    def center_lat(self) -> npt.NDArray[np.float64]:
        return self._center_lat

    def summary(self) -> dict:
        d = super().summary()
        d["type"] = "SuperGrid"
        return d


class CurvilinearGrid(SuperGrid):
    """Alias kept for backwards compatibility — use SuperGrid instead."""

    def __init__(
        self,
        path: "str | Path",
        x_var: str = "",
        y_var: str = "",
    ) -> None:
        super().__init__(path, x_var, y_var)
