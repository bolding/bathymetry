"""Source bathymetry readers.

Two backends are provided:

GEBCO
    Local NetCDF file (e.g. GEBCO_2024.nc). Variable ``elevation`` is
    positive-up; ocean values are negative.

EMODnet
    Downloaded on-the-fly from the EMODnet WCS service as a GeoTIFF, then
    converted to the same xr.Dataset layout as GEBCO.  Requires *rioxarray*.

Both return an ``xr.Dataset`` with:

* ``depth``  – positive-down depth in metres (NaN on land)
* ``lon``    – 1-D longitude coordinate (degrees East)
* ``lat``    – 1-D latitude coordinate (degrees North)
* ``land``   – boolean mask (True = land / no data)
"""

from __future__ import annotations

import tempfile
import urllib.request
from typing import Optional

import numpy as np
import xarray as xr

_EMODNET_BASE_URL = (
    "https://ows.emodnet-bathymetry.eu/wcs?service=wcs&version=1.0.0&"
)
_EMODNET_DEFAULT_RES = 1.0 / 60 / 16  # ~115 m


def read_source(
    source: str,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    pad_deg: float = 1.0,
    **kwargs,
) -> xr.Dataset:
    """Load fine-resolution bathymetry from *source*, clipped to the area of interest.

    Parameters
    ----------
    source : str
        Either a file path to a GEBCO-style NetCDF, or the string ``"emodnet"``
        to fetch from the EMODnet WCS service.
    lon_bounds : (lon_min, lon_max)
        Destination grid longitude extent in degrees East.
    lat_bounds : (lat_min, lat_max)
        Destination grid latitude extent in degrees North.
    pad_deg : float
        Extra margin added on all sides before subsetting, to avoid edge
        artefacts during regridding.
    **kwargs
        Passed through to the backend:
        - GEBCO: ``var_name`` (default ``"elevation"``)
        - EMODnet: ``resolution`` (degrees, default ~115 m), ``tiff_path``
    """
    lon_min = lon_bounds[0] - pad_deg
    lon_max = lon_bounds[1] + pad_deg
    lat_min = lat_bounds[0] - pad_deg
    lat_max = lat_bounds[1] + pad_deg

    if source.lower() == "emodnet":
        return _read_emodnet(lon_min, lon_max, lat_min, lat_max, **kwargs)
    else:
        return _read_gebco(source, lon_min, lon_max, lat_min, lat_max, **kwargs)


# ---------------------------------------------------------------------------
# GEBCO backend
# ---------------------------------------------------------------------------

def _read_gebco(
    path: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    var_name: str = "elevation",
) -> xr.Dataset:
    import os

    # HDF5/netCDF4 tries to create lock files in the same directory as the
    # source file.  On read-only or NFS mounts this raises PermissionError
    # before any data is read.  Setting this env var disables locking.
    # Must be done before the first netCDF4/h5py import.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # Specify engine explicitly to skip xarray's magic-number guessing step,
    # which calls plain open(path, "rb") and fails on restricted mounts even
    # when the file is readable.  Use mode='r' to prevent any write attempt.
    try:
        ds = xr.open_dataset(path, engine="netcdf4", mode="r")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot open {path!r}: permission denied.\n"
            "On NFS / read-only mounts also try:\n"
            "  export HDF5_USE_FILE_LOCKING=FALSE\n"
            "If the file is genuinely unreadable, copy it to a local path and "
            "pass that path via --source."
        ) from exc

    # Normalise coordinate names (GEBCO uses 'lon'/'lat')
    rename = {}
    for name in ds.coords:
        lower = str(name).lower()
        if lower in ("longitude", "x") and "lon" not in ds.coords:
            rename[name] = "lon"
        if lower in ("latitude", "y") and "lat" not in ds.coords:
            rename[name] = "lat"
    if rename:
        ds = ds.rename(rename)

    # Subset to the padded bounding box
    ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    # Some GEBCO files have lat in descending order
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    elevation = ds[var_name].values.astype(np.float64)
    depth = -elevation  # positive-down
    land = depth <= 0.0

    return xr.Dataset(
        {
            "depth": (["lat", "lon"], np.where(land, np.nan, depth)),
            "land": (["lat", "lon"], land),
        },
        coords={"lon": ds.lon.values, "lat": ds.lat.values},
        attrs={
            "source": str(path),
            "lon_min": float(ds.lon.values.min()),
            "lon_max": float(ds.lon.values.max()),
            "lat_min": float(ds.lat.values.min()),
            "lat_max": float(ds.lat.values.max()),
        },
    )


# ---------------------------------------------------------------------------
# EMODnet backend
# ---------------------------------------------------------------------------

def _read_emodnet(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: float = _EMODNET_DEFAULT_RES,
    tiff_path: Optional[str] = None,
) -> xr.Dataset:
    try:
        import rioxarray  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "rioxarray is required for EMODnet downloads. "
            "Install it with: pip install rioxarray"
        ) from exc

    import rioxarray

    url = (
        _EMODNET_BASE_URL
        + f"request=GetCoverage&coverage=emodnet:mean&crs=EPSG:4326"
        f"&BBOX={lon_min - resolution},{lat_min - resolution},"
        f"{lon_max + resolution},{lat_max + resolution}"
        f"&format=GeoTIFF&interpolation=nearest"
        f"&resx={resolution}&resy={resolution}"
    )

    if tiff_path is not None:
        fout = open(tiff_path, "wb")
        dest = tiff_path
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        fout = tmp
        dest = tmp.name

    print(f"Downloading EMODnet bathymetry from WCS …")
    with urllib.request.urlopen(url) as resp, fout:
        fout.write(resp.read())

    da = rioxarray.open_rasterio(dest, default_name="elevation")
    if da.ndim == 3 and da.shape[0] == 1:
        da = da[0, :, :]

    # rioxarray uses x (lon) and y (lat) as coordinate names
    elevation = da.values.astype(np.float64)
    lon_1d = da.x.values.astype(np.float64)
    lat_1d = da.y.values.astype(np.float64)

    # Ensure lat is ascending
    if lat_1d[0] > lat_1d[-1]:
        elevation = elevation[::-1, :]
        lat_1d = lat_1d[::-1]

    depth = -elevation
    land = depth <= 0.0

    return xr.Dataset(
        {
            "depth": (["lat", "lon"], np.where(land, np.nan, depth)),
            "land": (["lat", "lon"], land),
        },
        coords={"lon": lon_1d, "lat": lat_1d},
        attrs={
            "source": "emodnet",
            "resolution_deg": resolution,
            "lon_min": float(lon_1d.min()),
            "lon_max": float(lon_1d.max()),
            "lat_min": float(lat_1d.min()),
            "lat_max": float(lat_1d.max()),
        },
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def source_summary(ds: xr.Dataset) -> dict:
    """Return a summary dict suitable for report.print_table."""
    depth = ds["depth"].values
    ocean = ~np.isnan(depth)
    dlon = float(np.diff(ds.lon.values).mean())
    dlat = float(np.diff(ds.lat.values).mean())
    return {
        "source": ds.attrs.get("source", "unknown"),
        "resolution (deg)": f"{dlon:.6f} × {dlat:.6f}",
        "nx × ny": f"{ds.sizes['lon']} × {ds.sizes['lat']}",
        "lon range": f"{ds.lon.values.min():.3f} – {ds.lon.values.max():.3f}",
        "lat range": f"{ds.lat.values.min():.3f} – {ds.lat.values.max():.3f}",
        "ocean cells": int(ocean.sum()),
        "min depth (m)": f"{float(np.nanmin(depth)):.1f}",
        "max depth (m)": f"{float(np.nanmax(depth)):.1f}",
    }
