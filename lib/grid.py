"""Grid definitions for the bathymetry interpolation pipeline.

Three concrete grid types share a common BaseGrid interface that exposes
corner and centre coordinates in geographic (lon/lat) degrees, as required
by ESMF.  All corner arrays have shape [ny+1, nx+1] and all centre arrays
have shape [ny, nx] following the numpy/xarray [row, col] convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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
            "lon_min": f"{self.lon_bounds[0]:.4f}",
            "lon_max": f"{self.lon_bounds[1]:.4f}",
            "lat_min": f"{self.lat_bounds[0]:.4f}",
            "lat_max": f"{self.lat_bounds[1]:.4f}",
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
        Western and eastern boundaries in degrees East.
    lat_min, lat_max : float
        Southern and northern boundaries in degrees North.
    dlon, dlat : float
        Cell size in degrees.
    rotation_deg : float
        Rotation of the grid axes counter-clockwise around the grid centre, in
        degrees.  A non-zero value produces a curvilinear corner-coordinate
        array even though the underlying resolution is regular.
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
    ) -> None:
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlon = dlon
        self.dlat = dlat
        self.rotation_deg = rotation_deg
        self._build()

    def _build(self) -> None:
        nx = round((self.lon_max - self.lon_min) / self.dlon)
        ny = round((self.lat_max - self.lat_min) / self.dlat)

        lon_c1d = self.lon_min + np.arange(nx + 1) * self.dlon
        lat_c1d = self.lat_min + np.arange(ny + 1) * self.dlat
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
        d.update({"dlon": self.dlon, "dlat": self.dlat, "rotation_deg": self.rotation_deg})
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


class CurvilinearGrid(BaseGrid):
    """Curvilinear grid from pre-computed corner coordinates.

    Not yet implemented — placeholder so the rest of the pipeline can accept
    the type without breaking.
    """

    def __init__(
        self,
        corner_lon: npt.NDArray[np.float64],
        corner_lat: npt.NDArray[np.float64],
    ) -> None:
        raise NotImplementedError(
            "CurvilinearGrid is not yet implemented. "
            "Supply corner_lon and corner_lat arrays of shape [ny+1, nx+1] "
            "once support is added."
        )

    @property
    def corner_lon(self) -> npt.NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    @property
    def corner_lat(self) -> npt.NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    @property
    def center_lon(self) -> npt.NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError

    @property
    def center_lat(self) -> npt.NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError
