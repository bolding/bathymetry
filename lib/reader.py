"""Source bathymetry readers.

Three backends are provided:

GEBCO (local file)
    Local NetCDF file (e.g. GEBCO_2024.nc). Variable ``elevation`` is
    positive-up; ocean values are negative.

GEBCO (auto-download)
    Use ``source: gebco`` (or ``gebco2025``) to download GEBCO 2025 on the
    fly from CEDA and cache it at ``~/.cache/gebco/gebco_2025.nc``.  The
    ~10 GB ZIP is downloaded once; subsequent runs use the cached NetCDF.

EMODnet
    Downloaded on-the-fly from the EMODnet WCS service as a GeoTIFF, then
    converted to the same xr.Dataset layout as GEBCO.  Requires *rioxarray*.

Both return an ``xr.Dataset`` with:

* ``depth``  – positive-down depth in metres (NaN on land)
* ``lon``    – 1-D longitude coordinate (degrees East)
* ``lat``    – 1-D latitude coordinate (degrees North)
* ``land``   – boolean mask (True = land / no data)

Coastline masking
-----------------
``apply_coastline_mask(ds, resolution)`` overlays a high-resolution land
polygon dataset on top of the raw bathymetry, forcing any source cell whose
centre falls inside a land polygon to land regardless of its GEBCO depth value.
Supported *resolution* values (Natural Earth datasets, auto-downloaded by
cartopy):

    ``"10m"``  – 1:10 000 000  (~1 km features; default)
    ``"50m"``  – 1:50 000 000  (~5 km features)
    ``"110m"`` – 1:110 000 000 (~10 km features)

Requires *rasterio* (``pip install rasterio``).
"""

from __future__ import annotations

import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

_EMODNET_BASE_URL = (
    "https://ows.emodnet-bathymetry.eu/wcs?service=wcs&version=1.0.0&"
)

_GEBCO_CACHE_DIR = Path.home() / ".cache" / "gebco"
_GEBCO_VERSIONS: dict[str, str] = {
    "2025": (
        "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/"
        "ice_surface_elevation/netcdf/gebco_2025.zip?download=1"
    ),
}
_EMODNET_DEFAULT_RES    = 1.0 / 480  # native EMODnet ≈ 230 m (7.5 arcseconds)
_EMODNET_CACHE_DIR      = "./emodnet_cache"
# Server size limit is ~97.66 MB. Empirically, 17°×4° ≈ 478 MB (>> limit), so
# the server counts native-resolution source cells (~7 MB/deg²).  Use 12 deg²
# per tile (≈ 84 MB) with 2-D tiling so any domain works at full resolution.
_EMODNET_MAX_AREA_DEG2  = 12.0


def _ensure_gebco(version: str = "2025") -> str:
    """Return path to a locally cached GEBCO NetCDF, downloading if necessary.

    The global GEBCO ZIP (~10 GB) is downloaded once and cached at
    ``~/.cache/gebco/gebco_<version>.nc``.  The ZIP is deleted after
    extraction to free disk space.
    """
    nc_path = _GEBCO_CACHE_DIR / f"gebco_{version}.nc"
    if nc_path.exists():
        print(f"  Using cached GEBCO {version}: {nc_path}")
        return str(nc_path)

    if version not in _GEBCO_VERSIONS:
        raise ValueError(
            f"No download URL known for GEBCO version {version!r}. "
            "Use source: /path/to/GEBCO_<year>.nc instead."
        )
    url = _GEBCO_VERSIONS[version]

    _GEBCO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _GEBCO_CACHE_DIR / f"gebco_{version}.zip"

    print(f"  Downloading GEBCO {version} (~10 GB) from CEDA …")
    print(f"    → {zip_path}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        mb = downloaded / 1024 ** 2
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            total_mb = total_size / 1024 ** 2
            print(f"\r    {pct:5.1f}%  ({mb:.0f} / {total_mb:.0f} MB)",
                  end="", flush=True)
        else:
            print(f"\r    {mb:.0f} MB downloaded", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress)
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise
    print()  # newline after progress line

    print(f"  Extracting NetCDF from ZIP …")
    try:
        with zipfile.ZipFile(str(zip_path)) as zf:
            nc_names = [n for n in zf.namelist() if n.lower().endswith(".nc")]
            if not nc_names:
                raise RuntimeError(
                    f"No .nc file found inside {zip_path}. "
                    "The GEBCO ZIP may have an unexpected layout."
                )
            extracted = Path(zf.extract(nc_names[0], path=str(_GEBCO_CACHE_DIR)))
            extracted.rename(nc_path)
    except Exception:
        nc_path.unlink(missing_ok=True)
        raise
    finally:
        zip_path.unlink(missing_ok=True)

    print(f"  GEBCO {version} cached at: {nc_path}")
    return str(nc_path)


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
        File path to a GEBCO-style NetCDF, the string ``"emodnet"`` to fetch
        from the EMODnet WCS service, or the string ``"gebco"`` / ``"gebco2025"``
        to auto-download GEBCO 2025 from CEDA and cache at
        ``~/.cache/gebco/gebco_2025.nc``.
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

    src_key = source.lower().strip()
    if src_key in ("gebco", "gebco2025"):
        source = _ensure_gebco("2025")
        return _read_gebco(source, lon_min, lon_max, lat_min, lat_max, **kwargs)
    elif src_key == "emodnet":
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

    # 2-D tiling: the WCS server reads native-resolution source data internally
    # before resampling, so the effective cost is ~7 MB/deg² (empirical).
    # Split into tiles no larger than _EMODNET_MAX_AREA_DEG2 to stay within
    # the server's 97.66 MB hard limit.
    W = lon_max - lon_min
    H = lat_max - lat_min
    area = W * H

    if area <= _EMODNET_MAX_AREA_DEG2:
        n_tlon, n_tlat = 1, 1
    else:
        n_tiles = int(np.ceil(area / _EMODNET_MAX_AREA_DEG2))
        n_tlon  = max(1, round(np.sqrt(n_tiles * W / H)))
        n_tlat  = max(1, int(np.ceil(n_tiles / n_tlon)))
        # Guarantee tile area ≤ limit (rounding may overshoot)
        while (W / n_tlon) * (H / n_tlat) > _EMODNET_MAX_AREA_DEG2:
            n_tlat += 1
        print(
            f"  Domain ({W:.1f}°×{H:.1f}° = {area:.0f} deg²) exceeds WCS limit "
            f"({_EMODNET_MAX_AREA_DEG2} deg²/tile); "
            f"tiling {n_tlon}×{n_tlat} (lon×lat) = {n_tlon * n_tlat} tiles …"
        )

    tile_w = W / n_tlon
    tile_h = H / n_tlat

    # Download tiles row-by-row (lat ascending); collect lon/lat/elev arrays
    # Grid: rows[j][i] = (lon_1d, lat_1d, elev_2d) for tile (i, j)
    grid: list = [[None] * n_tlon for _ in range(n_tlat)]
    n_total = n_tlon * n_tlat

    for j in range(n_tlat):
        for i in range(n_tlon):
            k = j * n_tlon + i
            tlon_min = lon_min + i * tile_w
            tlon_max = lon_min + (i + 1) * tile_w
            tlat_min = lat_min + j * tile_h
            tlat_max = lat_min + (j + 1) * tile_h

            if n_total > 1:
                print(f"  Tile {k + 1}/{n_total}: "
                      f"lon {tlon_min:.2f}°–{tlon_max:.2f}°, "
                      f"lat {tlat_min:.2f}°–{tlat_max:.2f}°N …")
            else:
                tile_area = (tlon_max - tlon_min) * (tlat_max - tlat_min)
                print(f"  Downloading EMODnet from WCS "
                      f"({tile_area:.1f} deg² at {resolution:.6g}°/cell) …")

            url = (
                _EMODNET_BASE_URL
                + f"request=GetCoverage&coverage=emodnet:mean&crs=EPSG:4326"
                f"&BBOX={tlon_min - resolution},{tlat_min - resolution},"
                f"{tlon_max + resolution},{tlat_max + resolution}"
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
                    f"EMODnet WCS error (tile {k + 1}/{n_total}):\n  {msg}\n\n"
                    f"Reduce _EMODNET_MAX_AREA_DEG2 (currently {_EMODNET_MAX_AREA_DEG2}) "
                    f"or set 'emodnet_resolution' to a coarser value."
                )

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                da = rioxarray.open_rasterio(tmp_path, default_name="elevation")
                if da.ndim == 3 and da.shape[0] == 1:
                    da = da[0, :, :]
                t_elev = da.values.astype(np.float64)
                t_lon  = da.x.values.astype(np.float64)
                t_lat  = da.y.values.astype(np.float64)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if t_lat[0] > t_lat[-1]:
                t_elev = t_elev[::-1, :]
                t_lat  = t_lat[::-1]

            grid[j][i] = (t_lon, t_lat, t_elev)

    # Merge: first concatenate tiles within each lat-row along lon, then stack rows
    row_lons: list = []
    row_lats: list = []
    row_elevs: list = []
    for j in range(n_tlat):
        lons  = np.concatenate([grid[j][i][0] for i in range(n_tlon)])
        lat   = grid[j][0][1]  # same for all tiles in a lat-row
        elevs = np.concatenate([grid[j][i][2] for i in range(n_tlon)], axis=1)
        # deduplicate overlapping lon boundary columns
        _, uidx = np.unique(lons, return_index=True)
        row_lons.append(lons[uidx])
        row_lats.append(lat)
        row_elevs.append(elevs[:, uidx])

    lon_1d    = row_lons[0]
    lat_1d    = np.concatenate(row_lats)
    elevation = np.concatenate(row_elevs, axis=0)
    # deduplicate overlapping lat boundary rows
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
# Coastline masking
# ---------------------------------------------------------------------------

_NE_RESOLUTIONS = ("10m", "50m", "110m")


def apply_coastline_mask(
    ds: xr.Dataset,
    resolution: str = "10m",
) -> xr.Dataset:
    """Force source cells inside land polygons to land.

    Uses Natural Earth land polygons (via cartopy + rasterio) to override the
    raw GEBCO/EMODnet land mask.  Any source cell whose centre falls inside a
    land polygon is set to land (``depth=NaN``, ``land=True``), regardless of
    its elevation value.

    Cells that GEBCO already marks as land are unaffected (no ocean cells are
    opened by this step).

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset as returned by :func:`read_source`.
    resolution : str
        Natural Earth resolution: ``"10m"`` (default), ``"50m"``, or
        ``"110m"``.  The shapefile is downloaded automatically by cartopy on
        first use and cached in ``~/.local/share/cartopy/``.

    Returns
    -------
    xr.Dataset
        A copy of *ds* with the ``depth`` and ``land`` arrays updated.
    """
    if resolution not in _NE_RESOLUTIONS:
        raise ValueError(
            f"resolution must be one of {_NE_RESOLUTIONS}, got {resolution!r}"
        )

    try:
        import cartopy.io.shapereader as shpreader
    except ImportError as exc:
        raise ImportError(
            "cartopy is required for coastline masking. "
            "Install it with: conda install cartopy"
        ) from exc

    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for coastline masking. "
            "Install it with: pip install rasterio"
        ) from exc

    lon = ds.lon.values
    lat = ds.lat.values
    ny, nx = len(lat), len(lon)

    print(f"  Loading Natural Earth land polygons ({resolution}) …", end=" ", flush=True)
    shp_path = shpreader.natural_earth(
        resolution=resolution, category="physical", name="land"
    )
    reader_ne = shpreader.Reader(shp_path)
    geoms = [rec.geometry for rec in reader_ne.records()]
    print(f"{len(geoms)} polygon(s) loaded")

    print(f"  Rasterizing onto {nx}×{ny} source grid …", end=" ", flush=True)
    # rasterio rasterize expects row-0 = north (descending lat).
    # from_bounds(west, south, east, north, width, height) produces that.
    transform = from_bounds(lon[0], lat[0], lon[-1], lat[-1], nx, ny)
    land_raster = rasterize(
        [(g, 1) for g in geoms],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    # rasterio row-0 = north; flip to ascending-lat order
    land_raster = land_raster[::-1, :]
    coast_land = land_raster.astype(bool)

    old_ocean = ~ds["land"].values
    new_land = ds["land"].values | coast_land
    n_added = int((new_land & old_ocean).sum())
    print(f"{n_added} additional land cells from coastline mask")

    depth = ds["depth"].values.copy()
    depth[new_land] = np.nan

    return ds.assign(
        {
            "depth": (["lat", "lon"], depth),
            "land":  (["lat", "lon"], new_land),
        }
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
