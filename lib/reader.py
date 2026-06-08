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

import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

_EMODNET_BASE_URL = (
    "https://ows.emodnet-bathymetry.eu/wcs?service=wcs&version=1.0.0&"
)
_EMODNET_DEFAULT_RES = 1.0 / 480  # native EMODnet ≈ 230 m (7.5 arcseconds)
_EMODNET_CACHE_DIR   = "./emodnet_cache"
_EMODNET_MAX_CELLS   = 20_000_000  # ~80 MB at 4 bytes/cell; server limit is ~97 MB


def read_source(
    source: str,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    pad_deg: float = 1.0,
    emodnet_cache_dir: str = _EMODNET_CACHE_DIR,
    emodnet_resolution: Optional[float] = None,
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
    emodnet_cache_dir : str
        Directory where downloaded EMODnet tiles are cached as NetCDF.
        Set to ``""`` to disable caching (default: ``"./emodnet_cache"``).
    emodnet_resolution : float or None
        Download resolution in degrees. ``None`` → native EMODnet (~230 m).
        Large domains are automatically split into tiles; results are cached.
    **kwargs
        Passed through to the GEBCO backend (``var_name`` etc.).
    """
    lon_min = lon_bounds[0] - pad_deg
    lon_max = lon_bounds[1] + pad_deg
    lat_min = lat_bounds[0] - pad_deg
    lat_max = lat_bounds[1] + pad_deg

    if source.lower() == "emodnet":
        kw: dict = {}
        if emodnet_resolution is not None:
            kw["resolution"] = emodnet_resolution
        return _read_emodnet(lon_min, lon_max, lat_min, lat_max,
                             cache_dir=emodnet_cache_dir, **kw)
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
    cache_dir: str = _EMODNET_CACHE_DIR,
) -> xr.Dataset:
    try:
        import rioxarray
    except ImportError as exc:
        raise ImportError(
            "rioxarray is required for EMODnet downloads. "
            "Install it with: pip install rioxarray"
        ) from exc

    # Merged-result cache (NetCDF keyed on bbox + resolution)
    cache_path: Optional[Path] = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        res_tag = f"{resolution:.10f}".rstrip("0").replace(".", "p")
        fname = (
            f"emodnet_{lon_min:.4f}_{lon_max:.4f}"
            f"_{lat_min:.4f}_{lat_max:.4f}_{res_tag}.nc"
        )
        cache_path = Path(cache_dir) / fname

    if cache_path is not None and cache_path.exists():
        print(f"  Using cached EMODnet data: {cache_path}")
        return xr.open_dataset(str(cache_path))

    # Estimate number of cells and split into lat strips if needed
    n_lon_cells = int(np.ceil((lon_max - lon_min) / resolution)) + 2
    n_lat_cells = int(np.ceil((lat_max - lat_min) / resolution)) + 2
    total_cells = n_lon_cells * n_lat_cells

    if total_cells <= _EMODNET_MAX_CELLS:
        lat_strips = [(lat_min, lat_max)]
    else:
        n_strips = int(np.ceil(total_cells / _EMODNET_MAX_CELLS))
        strip_h = (lat_max - lat_min) / n_strips
        lat_strips = [
            (lat_min + i * strip_h, lat_min + (i + 1) * strip_h)
            for i in range(n_strips)
        ]
        print(
            f"  Domain too large for one WCS request "
            f"({total_cells / 1e6:.1f}M cells at {resolution:.6g}°); "
            f"splitting into {n_strips} lat strips …"
        )

    lon_out: list = []
    lat_out: list = []
    elev_out: list = []

    for k, (tlat_min, tlat_max) in enumerate(lat_strips):
        if len(lat_strips) > 1:
            print(f"  Downloading strip {k + 1}/{len(lat_strips)} "
                  f"({tlat_min:.2f}°–{tlat_max:.2f}°N) …")
        else:
            print(f"  Downloading EMODnet bathymetry from WCS ({total_cells / 1e6:.1f}M cells) …")

        url = (
            _EMODNET_BASE_URL
            + f"request=GetCoverage&coverage=emodnet:mean&crs=EPSG:4326"
            f"&BBOX={lon_min - resolution},{tlat_min - resolution},"
            f"{lon_max + resolution},{tlat_max + resolution}"
            f"&format=GeoTIFF&interpolation=nearest"
            f"&resx={resolution}&resy={resolution}"
        )

        with urllib.request.urlopen(url) as resp:
            content = resp.read()

        if content[:5] == b"<?xml" or b"ServiceException" in content[:500]:
            match = re.search(
                rb"<ServiceException[^>]*>(.*?)</ServiceException>",
                content, re.DOTALL,
            )
            msg = (
                match.group(1).decode(errors="replace").strip()
                if match
                else content[:400].decode(errors="replace")
            )
            raise RuntimeError(
                f"EMODnet WCS error (strip {k + 1}/{len(lat_strips)}):\n  {msg}\n\n"
                f"Hint: set 'emodnet_resolution' in the config to a coarser value "
                f"(current: {resolution:.6g}°)."
            )

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            da = rioxarray.open_rasterio(tmp_path, default_name="elevation")
            if da.ndim == 3 and da.shape[0] == 1:
                da = da[0, :, :]
            strip_elev = da.values.astype(np.float64)
            strip_lon  = da.x.values.astype(np.float64)
            strip_lat  = da.y.values.astype(np.float64)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if strip_lat[0] > strip_lat[-1]:
            strip_elev = strip_elev[::-1, :]
            strip_lat  = strip_lat[::-1]

        lon_out.append(strip_lon)
        lat_out.append(strip_lat)
        elev_out.append(strip_elev)

    # Merge strips: same lon coords, concatenate lat; deduplicate overlap rows
    lon_1d   = lon_out[0]
    lat_1d   = np.concatenate(lat_out)
    elevation = np.concatenate(elev_out, axis=0)
    _, unique_idx = np.unique(lat_1d, return_index=True)
    lat_1d    = lat_1d[unique_idx]
    elevation = elevation[unique_idx, :]

    depth = -elevation
    land  = depth <= 0.0

    ds = xr.Dataset(
        {
            "depth": (["lat", "lon"], np.where(land, np.nan, depth)),
            "land":  (["lat", "lon"], land),
        },
        coords={"lon": lon_1d, "lat": lat_1d},
        attrs={
            "source":         "emodnet",
            "resolution_deg": resolution,
            "lon_min":        float(lon_1d.min()),
            "lon_max":        float(lon_1d.max()),
            "lat_min":        float(lat_1d.min()),
            "lat_max":        float(lat_1d.max()),
        },
    )

    if cache_path is not None:
        ds.to_netcdf(str(cache_path))
        print(f"  Cached to: {cache_path}")

    return ds


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
