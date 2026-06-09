"""bathymetry-regrid: conservative bathymetry interpolation pipeline.

Configuration can be provided entirely via a YAML file (``--config``).
Any option given on the command line overrides the corresponding YAML value.

Example YAML (``north_sea.yaml``)::

    source: /server/data/GEBCO/GEBCO_2023.nc

    grid:
      type: spherical
      lon_min: 0.0
      lon_max: 10.0
      lat_min: 50.0
      lat_max: 60.0
      dlon: 0.05        # omit dlon (or dlat) and set equidistant: true
      dlat: 0.05        # to compute the missing spacing from the central latitude
      equidistant: false  # true → dlon = dlat / cos(lat_center) for square cells
      rotation: 0.0
      interfaces: false   # true → lon/lat bounds are cell corners; false (default) → T-points

    regridding:
      cache_dir: ./regrid_weights
      min_depth: 2.0
      min_wet_fraction: 0.05
      # tile_cells: 50      # enable tiled regridding (50×50 dst cells per tile)
      # tile_buf_deg: 0.5   # source-side buffer around each tile (degrees)
      # pad_deg: 1.0        # extra source margin beyond grid extent (expert)

    analysis:
      nkeep_basins: 1
      max_section_profiles: 10   # section plots shown in report (all saved to CSV)
      wet_frac_threshold: 0.3
      sill_ratio_threshold: 0.7
      area_ratio_threshold: 0.5

    smooth:
      rx0: 0.2        # omit or set to null to skip smoothing

    output:
      file: bathy_northsea.nc
      report_dir: ./report/northsea

    fixes:             # optional: embed fixes here, or use --accept-fixes / --fixes-file
      - lon: 5.3
        lat: 55.7
        action: set_depth
        value: 22.0

Then run::

    bathymetry-regrid --config north_sea.yaml

Or override individual options::

    bathymetry-regrid --config north_sea.yaml --smooth-rx0 0.15  # APPENDS 0.15; YAML rx0 kept
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Allow importing lib/ modules without installation
_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
if _lib not in sys.path:
    sys.path.insert(0, os.path.abspath(_lib))

import numpy as np
import xarray as xr

import analysis
import boundary as boundarymod
import grid as gridmod
import interpolate
import reader
import report
import smooth as smoothmod
import thalweg as thalwegmod

# ---------------------------------------------------------------------------
# YAML config loading and merge
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve_fix_keys(fixes: list[dict], suggested_yaml: str) -> list[dict]:
    """Expand key-only fix references against fixes_suggested.yaml.

    A fix entry with only a ``key:`` field (no ``lon``/``lat``) is looked up
    in *suggested_yaml* and replaced with the full fix dict from that file.
    Entries that already have ``lon``/``lat`` are passed through unchanged —
    the ``key`` field, if present, is kept as a label but has no effect.

    Raises ``KeyError`` if a referenced key is not found in the suggested file.
    """
    # Short-circuit: no key-only entries
    if not any("key" in f and "lon" not in f for f in fixes):
        return fixes

    if not os.path.exists(suggested_yaml):
        raise FileNotFoundError(
            f"Fix key resolution requires '{suggested_yaml}' but the file does not exist.\n"
            "Run without --skip-regrid first to generate it."
        )

    all_suggested = _load_yaml(suggested_yaml).get("fixes") or []
    key_db: dict[str, dict] = {
        f["key"]: f for f in all_suggested if "key" in f
    }

    resolved = []
    for fix in fixes:
        if "key" in fix and "lon" not in fix:
            k = fix["key"]
            if k not in key_db:
                raise KeyError(
                    f"Fix key '{k}' not found in '{suggested_yaml}'.\n"
                    f"Available keys: {sorted(key_db)}"
                )
            resolved.append(dict(key_db[k]))   # copy so original is untouched
        else:
            resolved.append(fix)
    return resolved


def _nested_get(cfg: dict, *keys, default=None):
    """Traverse nested dicts, returning *default* if any key is missing."""
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k)
    return val if val is not None else default


def _merge(cli_val, cfg: dict, *yaml_keys, default=None):
    """Return CLI value if not None, else YAML nested value, else default."""
    if cli_val is not None:
        return cli_val
    return _nested_get(cfg, *yaml_keys, default=default)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _build_grid(cfg: dict, args: argparse.Namespace) -> gridmod.BaseGrid:
    grid_type = _merge(args.grid, cfg, "grid", "type", default="spherical")

    if grid_type == "spherical":
        lon_min = _merge(args.lon_min, cfg, "grid", "lon_min")
        lon_max = _merge(args.lon_max, cfg, "grid", "lon_max")
        lat_min = _merge(args.lat_min, cfg, "grid", "lat_min")
        lat_max = _merge(args.lat_max, cfg, "grid", "lat_max")
        dlon    = _merge(args.dlon,    cfg, "grid", "dlon")
        dlat    = _merge(args.dlat,    cfg, "grid", "dlat")
        rot     = _merge(args.rotation, cfg, "grid", "rotation", default=0.0)
        equidist   = _merge(args.equidistant, cfg, "grid", "equidistant", default=False)
        interfaces = _merge(args.interfaces,  cfg, "grid", "interfaces",  default=False)
        for name, val in [("lon-min", lon_min), ("lon-max", lon_max),
                          ("lat-min", lat_min), ("lat-max", lat_max)]:
            if val is None:
                raise ValueError(f"Missing required grid parameter: {name}")
        if equidist:
            if dlat is None and dlon is None:
                raise ValueError("--equidistant requires dlat (or dlon) to be set")
            lat_center = (float(lat_min) + float(lat_max)) / 2.0
            cos_lat = math.cos(math.radians(lat_center))
            if dlat is not None:
                # dlat is the authoritative spacing; always derive dlon
                dlon = round(float(dlat) / cos_lat, 6)
            else:
                # only dlon given — derive dlat
                dlat = round(float(dlon) * cos_lat, 6)
            nx_eq = round((float(lon_max) - float(lon_min)) / float(dlon))
            ny_eq = round((float(lat_max) - float(lat_min)) / float(dlat))
            logger.info(f"      [equidistant] lat_center={lat_center:.2f}°  "
                  f"dlon={float(dlon):.6g}°  dlat={float(dlat):.6g}°  "
                  f"→ grid {nx_eq} × {ny_eq} (lon × lat)")
        else:
            for name, val in [("dlon", dlon), ("dlat", dlat)]:
                if val is None:
                    raise ValueError(f"Missing required grid parameter: {name}")
        return gridmod.SphericalGrid(lon_min, lon_max, lat_min, lat_max,
                                     float(dlon), float(dlat), float(rot),
                                     interfaces=bool(interfaces))

    elif grid_type == "cartesian":
        x_min = _merge(args.x_min, cfg, "grid", "x_min")
        x_max = _merge(args.x_max, cfg, "grid", "x_max")
        y_min = _merge(args.y_min, cfg, "grid", "y_min")
        y_max = _merge(args.y_max, cfg, "grid", "y_max")
        dx    = _merge(args.dx,    cfg, "grid", "dx")
        dy    = _merge(args.dy,    cfg, "grid", "dy")
        crs   = _merge(args.crs,   cfg, "grid", "crs")
        rot   = _merge(args.rotation, cfg, "grid", "rotation", default=0.0)
        for name, val in [("x-min", x_min), ("x-max", x_max),
                          ("y-min", y_min), ("y-max", y_max),
                          ("dx", dx), ("dy", dy), ("crs", crs)]:
            if val is None:
                raise ValueError(f"Missing required grid parameter: {name}")
        return gridmod.CartesianGrid(x_min, x_max, y_min, y_max,
                                     float(dx), float(dy), crs, float(rot))

    elif grid_type == "rotated_pole":
        pole_lon  = _merge(args.pole_lon,  cfg, "grid", "pole_lon")
        pole_lat  = _merge(args.pole_lat,  cfg, "grid", "pole_lat")
        rlon_min  = _merge(args.rlon_min,  cfg, "grid", "rlon_min")
        rlon_max  = _merge(args.rlon_max,  cfg, "grid", "rlon_max")
        rlat_min  = _merge(args.rlat_min,  cfg, "grid", "rlat_min")
        rlat_max  = _merge(args.rlat_max,  cfg, "grid", "rlat_max")
        drot      = _merge(args.drot,      cfg, "grid", "drot")
        axis_rot  = _merge(args.axis_rotation, cfg, "grid", "axis_rotation", default=0.0)

        # Convenience: auto-compute pole from domain centre lon/lat
        lon_center = _merge(args.lon_center, cfg, "grid", "lon_center")
        lat_center = _merge(args.lat_center, cfg, "grid", "lat_center")
        if pole_lon is None or pole_lat is None:
            if lon_center is None or lat_center is None:
                raise ValueError(
                    "rotated_pole grid requires either (pole_lon + pole_lat) "
                    "or (lon_center + lat_center) to locate the rotated pole."
                )
            pole_lon, pole_lat = gridmod.pole_from_center(
                float(lon_center), float(lat_center))
            logger.info(f"      [rotated_pole] auto pole: "
                  f"pole_lat={pole_lat:.4f}°  pole_lon={pole_lon:.4f}°  "
                  f"(from centre {float(lat_center):.2f}°N, {float(lon_center):.2f}°E)")

        for name, val in [("rlon_min", rlon_min), ("rlon_max", rlon_max),
                          ("rlat_min", rlat_min), ("rlat_max", rlat_max),
                          ("drot", drot)]:
            if val is None:
                raise ValueError(f"Missing required rotated_pole grid parameter: {name}")

        nx_rp = round((float(rlon_max) - float(rlon_min)) / float(drot))
        ny_rp = round((float(rlat_max) - float(rlat_min)) / float(drot))
        logger.info(f"      [rotated_pole] drot={float(drot):.4g}°  "
              f"→ grid {nx_rp} × {ny_rp} (rlon × rlat)")
        return gridmod.RotatedPoleGrid(
            float(pole_lon), float(pole_lat),
            float(rlon_min), float(rlon_max),
            float(rlat_min), float(rlat_max),
            float(drot), float(axis_rot),
        )

    elif grid_type in ("supergrid", "curvilinear"):
        sg_file = _merge(None, cfg, "grid", "file")
        if sg_file is None:
            raise ValueError("supergrid grid type requires 'grid.file' pointing to the supergrid NetCDF")
        x_var = _merge(None, cfg, "grid", "x_var", default="")
        y_var = _merge(None, cfg, "grid", "y_var", default="")
        logger.info(f"      [supergrid] reading {sg_file}")
        return gridmod.SuperGrid(sg_file, x_var=str(x_var), y_var=str(y_var))

    raise ValueError(f"Unknown grid type: {grid_type!r}")


# ---------------------------------------------------------------------------
# Output variable naming
# ---------------------------------------------------------------------------

def _smooth_var_name(rx0: float) -> str:
    """Return a NetCDF-safe variable name encoding the rx0 value.

    Examples: rx0=0.2 → 'depth_rx0_0p20', rx0=0.15 → 'depth_rx0_0p15'.
    """
    return f"depth_rx0_{rx0:.2f}".replace(".", "p")


def _plot_subtitle(dst_grid, dst) -> str:
    """Build the two-line subtitle shown below the final bathymetry plot.

    Line 1: grid dimensions, wet-cell count, depth range.
    Line 2: resolution in degrees and approximate km (km varies with latitude
             for spherical grids, so Δx is given as a min–max range).
    """
    import xarray as xr

    mask = dst["mask"].values.astype(bool)
    depth = dst["depth"].values
    n_wet = int(mask.sum())
    d_min = float(np.nanmin(depth)) if mask.any() else float("nan")
    d_max = float(np.nanmax(depth)) if mask.any() else float("nan")

    line1 = (
        f"{dst_grid.nx} × {dst_grid.ny} cells  |  "
        f"{n_wet} wet  |  "
        f"depth {d_min:.0f}–{d_max:.0f} m"
    )

    R = 6371.0  # km
    if hasattr(dst_grid, "dlon"):
        # Spherical grid: Δy is constant, Δx varies with cos(lat)
        dlon, dlat = dst_grid.dlon, dst_grid.dlat
        lat_min, lat_max = dst_grid.lat_bounds
        dy_km = dlat * math.pi / 180.0 * R
        dx_km_eq = dlon * math.pi / 180.0 * R * math.cos(lat_min * math.pi / 180.0)
        dx_km_po = dlon * math.pi / 180.0 * R * math.cos(lat_max * math.pi / 180.0)
        dx_lo, dx_hi = min(dx_km_eq, dx_km_po), max(dx_km_eq, dx_km_po)
        deg_str = (
            f"Δlon=Δlat={dlon:.4g}°" if abs(dlon - dlat) < 1e-8
            else f"Δlon={dlon:.4g}°, Δlat={dlat:.4g}°"
        )
        if abs(dx_hi - dx_lo) < 0.05 * dy_km:
            km_str = f"Δx≈Δy≈{dy_km:.1f} km"
        else:
            km_str = f"Δy≈{dy_km:.1f} km,  Δx {dx_lo:.1f}–{dx_hi:.1f} km"
        line2 = f"{deg_str}  |  {km_str}"
    elif hasattr(dst_grid, "dx"):
        # Cartesian grid: both spacings are constant
        dx_km = dst_grid.dx / 1000.0
        dy_km = dst_grid.dy / 1000.0
        if abs(dx_km - dy_km) < 1e-6:
            km_str = f"Δx=Δy={dx_km:.3g} km"
        else:
            km_str = f"Δx={dx_km:.3g} km, Δy={dy_km:.3g} km"
        line2 = f"{km_str}  |  {dst_grid.crs}"
    else:
        line2 = ""

    return f"{line1}\n{line2}" if line2 else line1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Conservative bathymetry interpolation — YAML-driven with CLI overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # YAML config
    parser.add_argument("--config", metavar="FILE",
                        help="YAML configuration file.  CLI flags override YAML values.")

    # Source
    src_grp = parser.add_argument_group("Source")
    src_grp.add_argument("--source", default=None,
                         help='Primary source: GEBCO NetCDF path or "emodnet".')
    src_grp.add_argument("--source2", default=None,
                         help='Secondary source for comparison plot (NetCDF path or "emodnet").')
    src_grp.add_argument("--pad-deg", type=float, default=None,
                         help="Padding (degrees) around target grid when reading source.")

    # Grid
    grd = parser.add_argument_group("Grid type")
    grd.add_argument("--grid", choices=["spherical", "cartesian", "rotated_pole"], default=None)
    grd.add_argument("--rotation", type=float, default=None,
                     help="Grid rotation in degrees CCW.")

    sph = parser.add_argument_group("Spherical grid")
    sph.add_argument("--lon-min", type=float, default=None)
    sph.add_argument("--lon-max", type=float, default=None)
    sph.add_argument("--lat-min", type=float, default=None)
    sph.add_argument("--lat-max", type=float, default=None)
    sph.add_argument("--dlon", type=float, default=None)
    sph.add_argument("--dlat", type=float, default=None)
    sph.add_argument("--equidistant", action="store_true", default=None,
                     help="Compute the missing dlon (or dlat) from the other using the "
                          "central latitude so that grid cells are approximately square "
                          "in physical distance.  Specify exactly one of --dlon / --dlat.")
    sph.add_argument("--interfaces", action="store_true", default=None,
                     help="Treat lon_min/lat_min/lon_max/lat_max as cell-corner (interface) "
                          "positions.  Default: they are T-point (cell-centre) positions, "
                          "so lon_max/lat_max become the last T-point.")

    rtp = parser.add_argument_group("Rotated-pole grid")
    rtp.add_argument("--pole-lon", type=float, default=None, dest="pole_lon",
                     help="Geographic longitude of the rotated North Pole (degrees).")
    rtp.add_argument("--pole-lat", type=float, default=None, dest="pole_lat",
                     help="Geographic latitude of the rotated North Pole (degrees).")
    rtp.add_argument("--lon-center", type=float, default=None, dest="lon_center",
                     help="Geographic longitude of domain centre; pole is computed "
                          "automatically (alternative to --pole-lon/--pole-lat).")
    rtp.add_argument("--lat-center", type=float, default=None, dest="lat_center",
                     help="Geographic latitude of domain centre; pole is computed "
                          "automatically (alternative to --pole-lon/--pole-lat).")
    rtp.add_argument("--rlon-min", type=float, default=None, dest="rlon_min")
    rtp.add_argument("--rlon-max", type=float, default=None, dest="rlon_max")
    rtp.add_argument("--rlat-min", type=float, default=None, dest="rlat_min")
    rtp.add_argument("--rlat-max", type=float, default=None, dest="rlat_max")
    rtp.add_argument("--drot", type=float, default=None,
                     help="Cell spacing in rotated degrees (same for both axes).")
    rtp.add_argument("--axis-rotation", type=float, default=None, dest="axis_rotation",
                     help="Extra CCW rotation of grid axes within the rotated system (degrees).")

    crt = parser.add_argument_group("Cartesian grid")
    crt.add_argument("--x-min", type=float, default=None, dest="x_min")
    crt.add_argument("--x-max", type=float, default=None, dest="x_max")
    crt.add_argument("--y-min", type=float, default=None, dest="y_min")
    crt.add_argument("--y-max", type=float, default=None, dest="y_max")
    crt.add_argument("--dx",    type=float, default=None)
    crt.add_argument("--dy",    type=float, default=None)
    crt.add_argument("--crs",   default=None, help="CRS string, e.g. EPSG:32632.")

    # Regridding
    rg = parser.add_argument_group("Regridding")
    rg.add_argument("--cache-dir", default=None,
                    help="Directory for cached xESMF weight files.")
    rg.add_argument("--min-depth", type=float, default=None,
                    help="Minimum ocean depth after regridding (m).")
    rg.add_argument("--min-wet-fraction", type=float, default=None,
                    help="Force cells with wet_fraction below this threshold to land.")
    rg.add_argument("--skip-regrid", action="store_true",
                    help="Skip source reading and regridding (steps 1–3); load the "
                         "cached raw-regrid file from the cache directory instead.  "
                         "Use this to re-apply fixes or re-run analysis without "
                         "repeating the expensive regridding step.")

    # Analysis
    an = parser.add_argument_group("Analysis")
    an.add_argument("--nkeep-basins", type=int, default=None,
                    help="Number of connected ocean basins to retain.")
    an.add_argument("--wet-frac-threshold", type=float, default=None)
    an.add_argument("--sill-ratio-threshold", type=float, default=None)
    an.add_argument("--area-ratio-threshold", type=float, default=None)

    # Smoothing
    sm = parser.add_argument_group("Smoothing")
    sm.add_argument("--smooth-rx0", type=float, nargs='+', default=None,
                    metavar="RX0",
                    help="One or more rx0 targets to smooth to (e.g. --smooth-rx0 0.1 0.15). "
                         "Appended to any value(s) already set via 'smooth.rx0' in the YAML; "
                         "the YAML value is always kept.  Each produces a separate "
                         "output variable (depth_rx0_0p20, depth_rx0_0p15, …).")

    # Fixes
    fx = parser.add_argument_group("Fixes")
    fx.add_argument("--fixes-file", default=None, metavar="FILE",
                    help="YAML file containing a 'fixes:' list to apply "
                         "(e.g. the generated fixes_suggested.yaml). "
                         "Merged with any 'fixes:' already in the config.")
    fx.add_argument("--accept-fixes", action="store_true",
                    help="Automatically load fixes_suggested.yaml from the "
                         "report directory (equivalent to "
                         "--fixes-file <report_dir>/fixes_suggested.yaml).")

    tw = parser.add_argument_group("Thalweg analysis")
    tw.add_argument("--thalweg", action="store_true", default=None,
                    help="Enable thalweg analysis (fine-vs-coarse depth profiles). "
                         "Default: enabled when thalweg section present in config, "
                         "or use --thalweg to force on.")
    tw.add_argument("--no-thalweg", action="store_true",
                    help="Disable thalweg analysis even if config contains a "
                         "`thalwegs:` section.")

    # Output
    out = parser.add_argument_group("Output")
    out.add_argument("--name", default=None,
                     help="Short identifier for this bathymetry (used in file names). "
                          "E.g. 'northsea_0p05deg'. Derived from source name if omitted.")
    out.add_argument("--output", default=None, help="Output NetCDF file.")
    out.add_argument("--report-dir", default=None,
                     help="Directory for report figures, CSV, and Markdown.")
    out.add_argument("--write-boundaries", action="store_true",
                     help="Write open-boundary T-grid coordinate CSV files "
                          "(one per contiguous wet segment on each side: "
                          "west, north, east, south) into the report directory.")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).  Use DEBUG for internal progress "
             "details, WARNING to suppress step banners.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        stream=sys.stdout,
        force=True,  # override any handlers set by imported libraries
    )

    # ------------------------------------------------------------------
    # Load YAML and resolve final parameter values
    # ------------------------------------------------------------------
    cfg: dict = _load_yaml(args.config) if args.config else {}

    source        = _merge(args.source,        cfg, "source")
    source2       = _merge(args.source2,       cfg, "source2",            default=None)
    pad_deg       = _merge(args.pad_deg,       cfg, "regridding", "pad_deg", default=1.0)
    cache_dir     = _merge(args.cache_dir,    cfg, "regridding", "cache_dir",
                           default="./regrid_weights")
    emodnet_cache = _merge(None, cfg, "regridding", "emodnet_cache_dir",
                           default=reader._EMODNET_CACHE_DIR)
    emodnet_res   = _merge(None, cfg, "regridding", "emodnet_resolution",
                           default=None)   # None → reader default (1 arcminute)
    min_depth     = _merge(args.min_depth,    cfg, "regridding", "min_depth",     default=0.0)
    min_wf        = _merge(args.min_wet_fraction, cfg, "regridding", "min_wet_fraction",
                           default=0.0)
    coastline_res = _merge(None, cfg, "regridding", "coastline_mask", default=None)
    skip_regrid   = bool(args.skip_regrid)
    tile_cells    = int(_merge(None, cfg, "regridding", "tile_cells", default=0))
    tile_buf_deg  = float(_merge(None, cfg, "regridding", "tile_buf_deg", default=0.5))
    nkeep         = _merge(args.nkeep_basins, cfg, "analysis",   "nkeep_basins",  default=1)
    max_sections  = int(_merge(None, cfg, "analysis", "max_section_profiles", default=10))
    wf_thr        = _merge(args.wet_frac_threshold,  cfg, "analysis", "wet_frac_threshold",
                           default=0.3)
    sill_thr      = _merge(args.sill_ratio_threshold, cfg, "analysis", "sill_ratio_threshold",
                           default=0.7)
    area_thr      = _merge(args.area_ratio_threshold, cfg, "analysis", "area_ratio_threshold",
                           default=0.5)
    # Collect all rx0 targets: YAML value(s) are always kept; CLI --smooth-rx0 appends.
    _yaml_rx0 = _nested_get(cfg, "smooth", "rx0")
    if _yaml_rx0 is None:
        rx0_list: list[float] = []
    elif isinstance(_yaml_rx0, list):
        rx0_list = [float(v) for v in _yaml_rx0]
    else:
        rx0_list = [float(_yaml_rx0)]
    if args.smooth_rx0 is not None:
        for _cli_rx0 in args.smooth_rx0:
            if _cli_rx0 not in rx0_list:
                rx0_list.append(_cli_rx0)
    fixes_list    = list(_nested_get(cfg, "fixes") or [])

    # Thalweg: read from `thalweg:` section (new) or legacy `thalwegs:` list.
    _tw_cfg = _nested_get(cfg, "thalweg") or {}
    if not isinstance(_tw_cfg, dict):
        _tw_cfg = {}
    user_waypoints_cfg = list(
        _tw_cfg.get("waypoints", None)
        or _nested_get(cfg, "thalwegs")
        or []
    )
    thalweg_min_sill    = float(_tw_cfg.get("min_sill_m",     5.0))
    thalweg_max_detour  = float(_tw_cfg.get("max_detour",     2.5))
    # Boundary auto-detection uses a looser detour limit: the MST finds the
    # deepest-water route between two model boundaries, which can be longer
    # than the straight-line distance.  The waypoint limit (2.5×) blocked
    # legitimate deep-basin detours; 20× still rejects truly circular paths.
    thalweg_bdy_detour  = float(_tw_cfg.get("boundary_max_detour", 20.0))
    thalweg_sill_dedup  = float(_tw_cfg.get("sill_dedup_tol_m", 2.0))
    _thalweg_boundaries_csv = _tw_cfg.get("boundaries_csv", None)
    if _thalweg_boundaries_csv:
        # Resolve path relative to config file
        _cfg_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else "."
        _thalweg_boundaries_csv = os.path.join(_cfg_dir, _thalweg_boundaries_csv)
    _thalweg_default = bool(_tw_cfg.get("enabled", bool(user_waypoints_cfg)))
    run_thalweg = (not args.no_thalweg) and (args.thalweg or _thalweg_default)
    # Mode B (boundary auto-detection) is off by default — it relies on
    # _boundary_starts() which needs further tuning for complex domains.
    # Enable with thalweg.boundary_thalwegs: true in the config.
    run_boundary_thalweg = bool(_tw_cfg.get("boundary_thalwegs", False))

    if source is None:
        parser.error("--source (or 'source:' in YAML) is required")

    # ------------------------------------------------------------------
    # Resolve 'name' — used as prefix for output files and report dir
    # ------------------------------------------------------------------
    name = _merge(args.name, cfg, "name")
    if name is None:
        # Derive from source file stem, stripping common suffixes
        _src_key = source.lower().strip()
        if _src_key == "emodnet":
            stem = "emodnet"
        elif _src_key in ("gebco", "gebco2025"):
            stem = "gebco2025"
        else:
            stem = Path(source).stem
        name = stem.replace(" ", "_")

    # Defaults that depend on name
    output_file = _merge(args.output, cfg, "output", "file",
                         default=f"{name}.nc")
    report_dir  = _merge(args.report_dir, cfg, "output", "report_dir",
                         default=f"./report/{name}")
    log_depth_scale = bool((cfg.get("output") or {}).get("log_depth_scale", False))

    os.makedirs(report_dir, exist_ok=True)

    # Path to the auto-generated suggestions file (needed for key resolution)
    _suggested_yaml = os.path.join(report_dir, "fixes_suggested.yaml")

    # Load extra fixes from --fixes-file or --accept-fixes
    fixes_file = args.fixes_file
    if not fixes_file and args.accept_fixes:
        fixes_file = _suggested_yaml
    if fixes_file:
        if not os.path.exists(fixes_file):
            parser.error(f"fixes file not found: {fixes_file}")
        extra = _load_yaml(fixes_file).get("fixes") or []
        fixes_list = fixes_list + extra
        logger.info(f"Loaded {len(extra)} fix(es) from {fixes_file}")

    # Resolve key-only entries (e.g. - key: b001) against fixes_suggested.yaml
    try:
        fixes_list = _resolve_fix_keys(fixes_list, _suggested_yaml)
    except (FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))

    # Title for the Markdown report
    rpt_title = (
        f"Bathymetry — {name}"
    )
    rpt = report.MarkdownReport(rpt_title)

    # ------------------------------------------------------------------
    # Step 1 – Build target grid
    # Prefix for all figure/CSV files in the report dir
    pfx = name + "_"

    # ------------------------------------------------------------------
    logger.info(f"\n[1/6] Building target grid … (name='{name}')")
    dst_grid = _build_grid(cfg, args)
    grid_summary = dst_grid.summary()
    report.print_table(grid_summary, title="Target grid")

    # Rotated-pole: two-panel globe plot showing grid footprint and pole location
    grid_images: list[str] = []
    grid_extra_text = ""
    if isinstance(dst_grid, gridmod.RotatedPoleGrid):
        rp_plot = pfx + "01_rotated_pole.png"
        report.plot_rotated_pole(
            dst_grid.corner_lon, dst_grid.corner_lat,
            dst_grid.pole_lon, dst_grid.pole_lat,
            title=f"{name} — rotated-pole grid geometry",
            path=os.path.join(report_dir, rp_plot),
        )
        grid_images.append(rp_plot)
        grid_extra_text = (
            f"  \nRotated North Pole at "
            f"({dst_grid.pole_lon:.4f}°E, {dst_grid.pole_lat:.4f}°N).  "
            "Left globe: geographic view; right globe: view centred on the rotated pole."
        )

    rpt.add_section(
        "Target grid",
        text=(
            f"Grid type: **{type(dst_grid).__name__}**, "
            f"{dst_grid.nx} × {dst_grid.ny} cells."
            + grid_extra_text
        ),
        table=grid_summary,
        images=grid_images,
    )

    # Raw-regrid cache: saved after step 3, reloaded by --skip-regrid
    raw_regrid_cache = os.path.join(str(cache_dir), f"{name}_raw_regrid.nc")

    # ------------------------------------------------------------------
    # Steps 2 + 3 – Read source and regrid  (skipped by --skip-regrid)
    # ------------------------------------------------------------------
    src = None   # populated in the else branch; None signals skip_regrid to callers
    if skip_regrid:
        if not os.path.exists(raw_regrid_cache):
            parser.error(
                f"--skip-regrid: raw-regrid cache not found: {raw_regrid_cache}\n"
                "Run without --skip-regrid first to create it."
            )
        logger.info(f"\n[2/6] Skipped (--skip-regrid)")
        # Re-include any source plots produced by the previous full run.
        src_images = [
            f for f in (pfx + "02a_source_raw.png",
                        pfx + "02b_source_coastline_masked.png")
            if os.path.exists(os.path.join(report_dir, f))
        ]
        rpt.add_section(
            "Source bathymetry",
            text="Source reading skipped (`--skip-regrid`); plots from previous full run.",
            images=src_images,
        )

        logger.info(f"\n[3/6] Loading cached raw-regrid result: {raw_regrid_cache}")
        dst = xr.open_dataset(raw_regrid_cache).load()
        # Validate cache shape against current grid spec.
        # A mismatch means the grid config changed since the cache was built
        # (e.g. the T-point convention fix that adds +1 to nx/ny).
        _cached_ny = dst.sizes.get("lat", dst.sizes.get("y", -1))
        _cached_nx = dst.sizes.get("lon", dst.sizes.get("x", -1))
        if (_cached_ny, _cached_nx) != (dst_grid.ny, dst_grid.nx):
            parser.error(
                f"--skip-regrid: cached grid shape ({_cached_ny}×{_cached_nx}) "
                f"does not match the current config ({dst_grid.ny}×{dst_grid.nx}).\n"
                f"The cache is stale — delete it and re-run without --skip-regrid:\n"
                f"  rm {raw_regrid_cache}"
            )
        dst_sum = interpolate.regrid_summary(dst, dst_grid)
        report.print_table(dst_sum, title="Regridded destination (cached)")
        rpt.add_section(
            "Conservative regridding",
            text=(
                f"Loaded from cache (`--skip-regrid`): `{raw_regrid_cache}`.  "
                "Source reading and regridding were skipped."
            ),
            table=dst_sum,
        )
    else:
        # ------------------------------------------------------------------
        # Step 2 – Read source bathymetry
        # ------------------------------------------------------------------
        logger.info("\n[2/6] Reading source bathymetry …")
        t0 = time.time()
        src = reader.read_source(
            source, dst_grid.lon_bounds, dst_grid.lat_bounds, pad_deg=float(pad_deg),
            emodnet_cache_dir=str(emodnet_cache),
            emodnet_resolution=float(emodnet_res) if emodnet_res is not None else None,
        )

        _src_key2 = source.lower().strip()
        if _src_key2 == "emodnet":
            src_label = "EMODnet"
        elif _src_key2 in ("gebco", "gebco2025"):
            src_label = "GEBCO 2025"
        else:
            src_label = Path(source).name

        def _save_src_plot(tag: str, title: str) -> tuple[str, str]:
            """Save PNG + HTML for current state of *src*; return (png_name, html_name)."""
            png = pfx + tag + ".png"
            htm = pfx + tag + ".html"
            report.plot_depth(
                src.lon.values, src.lat.values,
                src["depth"].values, (~src["land"].values).astype(float),
                title=title,
                path=os.path.join(report_dir, png),
                log_scale=log_depth_scale,
                interactive=True,
            )
            return png, htm

        src_images: list[str] = []
        src_links:  list[str] = []

        # 2a — raw source
        raw_png, raw_html = _save_src_plot("02a_source_raw", f"Source (raw): {src_label}")
        src_images.append(raw_png)
        src_links.append(f"[Interactive — raw]({raw_html})")

        # 2b — after coastline mask (only when enabled)
        coast_note = ""
        if coastline_res:
            src = reader.apply_coastline_mask(src, resolution=str(coastline_res))
            cm_png, cm_html = _save_src_plot(
                "02b_source_coastline_masked",
                f"Source after coastline mask (NE {coastline_res}): {src_label}",
            )
            src_images.append(cm_png)
            src_links.append(f"[Interactive — coastline masked]({cm_html})")
            coast_note = f"  Coastline mask applied (Natural Earth {coastline_res})."

        src_sum = reader.source_summary(src)
        report.print_table(src_sum, title="Source bathymetry")
        logger.info(f"      done in {time.time()-t0:.1f} s")

        rpt.add_section(
            "Source bathymetry",
            text=(
                f"Source data read and clipped to the target domain "
                f"(±{pad_deg}° buffer applied).{coast_note}  "
                + "  ".join(src_links)
            ),
            table=src_sum,
            images=src_images,
        )

        # ------------------------------------------------------------------
        # Step 3 – ESMF conservative interpolation
        # ------------------------------------------------------------------
        logger.info("\n[3/6] Conservative regridding (xESMF) …  (may take a minute for large grids)")
        t0 = time.time()
        dst = interpolate.regrid(
            src, dst_grid,
            min_depth=float(min_depth),
            min_wet_fraction=float(min_wf),
            cache_dir=str(cache_dir),
            tile_cells=tile_cells,
            tile_buf_deg=tile_buf_deg,
        )
        # Save raw post-regrid result for --skip-regrid on subsequent runs
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        dst.to_netcdf(raw_regrid_cache)
        logger.info(f"  Raw-regrid result cached: {raw_regrid_cache}")

        dst_sum = interpolate.regrid_summary(dst, dst_grid)
        report.print_table(dst_sum, title="Regridded destination")
        logger.info(f"      done in {time.time()-t0:.1f} s")

        regrid_plot = pfx + "03_regrid_result.png"
        wf_plot = pfx + "03_wet_fraction.png"
        report.plot_depth(
            dst.lon.values, dst.lat.values,
            dst["depth"].values, dst["mask"].values,
            title=f"{name} — regridded depth (m)",
            path=os.path.join(report_dir, regrid_plot),
            interactive=True,
            log_scale=log_depth_scale,
        )
        report.plot_depth(
            dst.lon.values, dst.lat.values,
            dst["wet_fraction"].values, dst["mask"].values,
            title=f"{name} — wet fraction",
            path=os.path.join(report_dir, wf_plot),
            cmap=report._cm_fraction(),
            vmin=0.0, vmax=1.0,
        )
        rpt.add_section(
            "Conservative regridding",
            text=(
                "ESMF first-order conservative regridding (FRACAREA normalisation). "
                f"Weight file cached in `{cache_dir}`. "
                f"Minimum depth: {min_depth} m."
            ),
            table=dst_sum,
            images=[regrid_plot, wf_plot],
        )

    # ------------------------------------------------------------------
    # Step 3b – Second source comparison (optional)
    # ------------------------------------------------------------------
    dst2_mask = None        # set below if source2 is provided
    dst2_depth_vals = None  # raw regridded depth from source2
    def _src_display_label(s: str) -> str:
        k = s.lower().strip()
        if k == "emodnet":
            return "EMODnet"
        if k in ("gebco", "gebco2025"):
            return "GEBCO 2025"
        return Path(s).name

    src1_label = _src_display_label(source)
    src2_label: str = ""
    if source2 is not None:
        src2_label = _src_display_label(source2)
        logger.info(f"\n[3b] Reading and regridding second source ({src2_label}) for comparison …")
        t0 = time.time()
        src2 = reader.read_source(
            source2, dst_grid.lon_bounds, dst_grid.lat_bounds, pad_deg=float(pad_deg),
            emodnet_cache_dir=str(emodnet_cache),
            emodnet_resolution=float(emodnet_res) if emodnet_res is not None else None,
        )
        dst2 = interpolate.regrid(
            src2, dst_grid,
            min_depth=float(min_depth),
            min_wet_fraction=float(min_wf),
            cache_dir=str(cache_dir),
            tile_cells=tile_cells,
            tile_buf_deg=tile_buf_deg,
        )
        logger.info(f"      done in {time.time()-t0:.1f} s")

        dst2_mask        = dst2["mask"].values
        dst2_depth_vals  = dst2["depth"].values
        common_ocean_3b  = (dst["mask"].values == 1) & (dst2_mask == 1)

        cmp_plot = pfx + "03b_source_comparison.png"
        report.plot_source_comparison(
            dst.lon.values, dst.lat.values,
            dst["mask"].values, dst2_mask,
            name1=src1_label, name2=src2_label,
            path=os.path.join(report_dir, cmp_plot),
        )

        diff_plot_3b = pfx + "03b_depth_diff.png"
        diff_arr_3b = np.where(
            common_ocean_3b,
            dst["depth"].values - dst2_depth_vals,
            np.nan,
        )
        report.plot_depth_diff(
            dst.lon.values, dst.lat.values,
            diff_arr_3b,
            title=f"{name} — raw depth difference: {src1_label} − {src2_label}",
            path=os.path.join(report_dir, diff_plot_3b),
        )

        n_both   = int(((dst["mask"].values == 1) & (dst2_mask == 1)).sum())
        n_src1   = int(((dst["mask"].values == 1) & (dst2_mask == 0)).sum())
        n_src2   = int(((dst["mask"].values == 0) & (dst2_mask == 1)).sum())
        n_land   = int(((dst["mask"].values == 0) & (dst2_mask == 0)).sum())
        rpt.add_section(
            "Source mask comparison",
            text=(
                f"Both sources regridded to the target grid "
                f"({dst_grid.nx} × {dst_grid.ny} cells).  "
                f"Discrepancies reveal coastline differences at model resolution."
            ),
            table={
                "Source 1": src1_label,
                "Source 2": src2_label,
                "Common water cells": n_both,
                "Common land cells":  n_land,
                f"{src1_label} only (water)": n_src1,
                f"{src2_label} only (water)": n_src2,
            },
            images=[cmp_plot, diff_plot_3b],
        )

    # ------------------------------------------------------------------
    # Step 4a – Apply user fixes (if any)
    # Run first so fixes affect which cells are kept by basin removal and
    # which interfaces are checked by strait detection.
    # ------------------------------------------------------------------
    if fixes_list:
        logger.info(f"\n[4a] Applying {len(fixes_list)} fix(es) from config …")
        dst, applied_fixes = analysis.apply_fixes(dst, fixes_list)
        report.print_table({"fixes applied": len(applied_fixes)}, title="User fixes")
        report.save_csv(applied_fixes, os.path.join(report_dir, pfx + "04a_fixes_applied.csv"))
        rpt.add_section(
            "User-specified fixes",
            text=f"{len(applied_fixes)} fix(es) applied from configuration.",
            table_rows=applied_fixes,
        )
    else:
        logger.info("\n[4a] No fixes configured — skipping.")

    # ------------------------------------------------------------------
    # Step 4b – Explicit mask regions (force areas to land)
    # Runs BEFORE isolated-cell masking so that blocking a fjord mouth
    # causes the interior cells to be picked up as isolated and removed.
    # ------------------------------------------------------------------
    mask_regions_list = _nested_get(cfg, "mask_regions") or []
    if mask_regions_list:
        logger.info(f"\n[4b] Applying {len(mask_regions_list)} explicit mask region(s) …")
        dst, mr_applied = analysis.apply_mask_regions(dst, mask_regions_list)
        mr_sum = analysis.mask_regions_summary(mr_applied)
        report.print_table(mr_sum, title="Explicit mask regions")
        report.save_csv(mr_applied, os.path.join(report_dir, pfx + "04b_mask_regions.csv"))
        mr_plot = pfx + "04b_mask_regions.png"
        report.plot_depth(
            dst.lon.values, dst.lat.values,
            dst["depth"].values, dst["mask"].values,
            title=f"{name} — after explicit masking",
            path=os.path.join(report_dir, mr_plot),
            log_scale=log_depth_scale,
        )
        rpt.add_section(
            "Explicit mask regions",
            text=(
                f"{len(mask_regions_list)} region(s) forced to land regardless of "
                "bathymetry (e.g. closing a fjord mouth so the interior is later "
                "removed by the isolation step). "
                "Specified under `mask_regions:` in the YAML config. "
                "Supported types: `rectangle`, `polygon`, `point`."
            ),
            table=mr_sum,
            table_rows=mr_applied,
            images=[mr_plot],
        )
    else:
        logger.info("\n[4b] No explicit mask regions configured — skipping.")

    # ------------------------------------------------------------------
    # Step 4c – Isolated-cell masking
    # Runs AFTER explicit masking (fjord mouth closed → interior removed)
    # and BEFORE strait detection so that isolated basins do not generate
    # spurious interface flags.
    # Save wet_fraction now: mask_isolated zeroes it for removed cells, but
    # detect_land_bridges needs the pre-isolation values to find cells that
    # were forced to land by min_wet_fraction.
    # ------------------------------------------------------------------
    _pre_isolation_wf = dst["wet_fraction"].values.copy()
    logger.info(f"\n[4c/6] Masking isolated ocean regions (keep {nkeep}) …")
    dst_clean, basin_records = analysis.mask_isolated(dst, nkeep=int(nkeep))
    iso_sum = analysis.isolation_summary(basin_records)
    report.print_table(iso_sum, title="Isolated cells")
    report.save_csv(basin_records, os.path.join(report_dir, pfx + "04c_basins.csv"))
    basins_plot = pfx + "04c_basins.png"
    report.plot_basins(
        dst_clean.lon.values, dst_clean.lat.values,
        dst_clean["basin_labels"].values,
        title="Connected ocean basins",
        path=os.path.join(report_dir, basins_plot),
        nkeep=int(nkeep),
    )
    rpt.add_section(
        "Isolated-cell masking",
        text=(
            f"Connected-component labelling (4-connectivity). "
            f"Keeping {nkeep} largest basin(s). "
            "Runs after explicit mask regions so that fjord interiors "
            "whose mouth was closed above are removed here automatically, "
            "and before strait detection so isolated basins do not generate "
            "spurious interface flags."
        ),
        table=iso_sum,
        images=[basins_plot],
    )
    dst = dst_clean

    # ------------------------------------------------------------------
    # Step 4c-ii – Land-bridge detection
    # Find land cells (wet_fraction > 0, forced below min_wet_fraction) that
    # sit between two disconnected wet basins → suggest open_cell fixes.
    # ------------------------------------------------------------------
    bridge_records = analysis.detect_land_bridges(
        dst["mask"].values,
        _pre_isolation_wf,
        dst.lon.values,
        dst.lat.values,
        depth=np.where(dst["mask"].values, dst["depth"].values, np.nan),
    )
    bridge_sum = analysis.land_bridge_summary(bridge_records)
    report.print_table(bridge_sum, title="Land-bridge detection")
    if bridge_records:
        logger.info(f"      {len(bridge_records)} land-bridge cell(s) found")
        report.save_csv(
            [{k: v for k, v in r.items() if k != "adjacent_components"}
             for r in bridge_records],
            os.path.join(report_dir, pfx + "04c_land_bridges.csv"),
        )
        rpt.add_section(
            "Land-bridge detection",
            text=(
                f"{len(bridge_records)} land cell(s) with non-zero wet_fraction "
                "sit between two or more disconnected wet basins.  "
                "These were forced to land by `regridding.min_wet_fraction`.  "
                "Applying an `open_cell` fix restores the connection.  "
                "Suggested fixes are included in `fixes_suggested.yaml`."
            ),
            table=bridge_sum,
        )
    else:
        rpt.add_section("Land-bridge detection", text="No land bridges found.")

    # ------------------------------------------------------------------
    # Step 4d – Strait detection
    # Runs on the basin-cleaned grid so only interfaces between genuinely
    # connected ocean cells are checked — no spurious flags from isolated
    # basins or enclosed seas.
    # ------------------------------------------------------------------
    logger.info("\n[4d/6] Detecting narrow straits …")
    fixes_yaml_path = os.path.join(report_dir, "fixes_suggested.yaml")
    if src is None:
        logger.info("      Skipped (--skip-regrid: fine source not available; see previous report).")
        strait_records = []
        if bridge_records:
            report.save_fixes_yaml([], fixes_yaml_path, bridge_records=bridge_records)
        rpt.add_section(
            "Strait and connectivity analysis",
            text="Skipped (`--skip-regrid`): fine-resolution source not loaded. "
                 "Strait results from the initial full run are still in the report directory.",
        )
    else:
        t0 = time.time()
        strait_records = analysis.sort_straits(analysis.find_straits(
            src, dst, dst_grid,
            wet_frac_threshold=float(wf_thr),
            sill_ratio_threshold=float(sill_thr),
            area_ratio_threshold=float(area_thr),
        ))

        strait_sum = analysis.strait_summary(strait_records)
        report.print_table(strait_sum, title="Strait detection")
        logger.info(f"      {len(strait_records)} interfaces flagged in {time.time()-t0:.1f} s")

        clean_records = [{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in strait_records]
        csv_path = os.path.join(report_dir, pfx + "04d_straits.csv")
        report.save_csv(clean_records, csv_path)

        report.save_fixes_yaml(clean_records, fixes_yaml_path,
                               bridge_records=bridge_records)

        straits_plot = pfx + "04d_straits.png"
        report.plot_straits(
            dst.lon.values, dst.lat.values,
            dst["depth"].values, dst["mask"].values,
            strait_records=clean_records,
            path=os.path.join(report_dir, straits_plot),
            domain_bounds=(
                dst_grid.lon_bounds[0], dst_grid.lon_bounds[1],
                dst_grid.lat_bounds[0], dst_grid.lat_bounds[1],
            ),
        )

        warn_msgs = []
        if strait_sum.get("BLOCKED", 0):
            warn_msgs.append(
                f"{strait_sum['BLOCKED']} BLOCKED interface(s) — no fine wet path found. "
                "Review `fixes_suggested.yaml` in the report directory, then re-run with --accept-fixes."
            )
        if strait_sum.get("SILL_DEFICIT", 0):
            warn_msgs.append(
                f"{strait_sum['SILL_DEFICIT']} SILL_DEFICIT interface(s) — coarse sill shallower "
                "than fine-grid sill. Dense bottom-water inflow may be blocked."
            )
        rpt.add_section(
            "Strait and connectivity analysis",
            text=(
                "Each interface between adjacent wet cells is checked for narrow width, "
                "sill-depth deficit, and connectivity breaks. "
                "Runs after basin removal so only connected-ocean interfaces are checked.\n\n"
                f"Suggested fixes written to `{fixes_yaml_path}`. "
                "To adopt: copy the relevant entries into the `fixes:` section of your YAML "
                "config and re-run."
            ),
            table=strait_sum,
            images=[straits_plot],
            warnings=warn_msgs,
        )

    # Cross-section profiles — one report section per flagged interface.
    # Capped at max_sections (YAML: analysis.max_section_profiles, default 10).
    n_total_straits = len(strait_records)
    shown_records   = strait_records[:max_sections]
    if n_total_straits > max_sections:
        logger.info(f"      (showing {max_sections} of {n_total_straits} section profiles; "
              f"set analysis.max_section_profiles in YAML to show more)")
    _zoom_lon = getattr(dst_grid, "dlon", 0.1) * 10
    _zoom_lat = getattr(dst_grid, "dlat", 0.1) * 10
    for k, rec in enumerate(shown_records):
        dist_key = "_dlat_km" if rec["direction"] == "U" else "_dlon_km"
        depth_sec = rec["_depth_section"]
        cell_km = rec.get(dist_key, 0.1)
        dist = np.arange(len(depth_sec)) * cell_km
        img = pfx + f"04d_section_{k:03d}.png"
        report.plot_section_profile(
            dist, depth_sec, rec["sill_depth_coarse"],
            title=(f"{rec['category']} | lon={rec['lon']:.3f}, lat={rec['lat']:.3f}"),
            path=os.path.join(report_dir, img),
            inset_lon=rec["lon"],
            inset_lat=rec["lat"],
            inset_bounds=(
                rec["lon"] - _zoom_lon, rec["lon"] + _zoom_lon,
                rec["lat"] - _zoom_lat, rec["lat"] + _zoom_lat,
            ),
            fine_sub=rec.get("_fine_sub"),
            fine_lons=rec.get("_fine_lons"),
            fine_lats=rec.get("_fine_lats"),
            cs_col=rec.get("_cs_col"),
            cs_row=rec.get("_cs_row"),
            coarse_corner_lons=rec.get("_coarse_corner_lons"),
            coarse_corner_lats=rec.get("_coarse_corner_lats"),
        )
        direction_label = "U (N–S section)" if rec["direction"] == "U" else "V (E–W section)"
        rpt.add_section(
            f"Interface {k + 1}/{n_total_straits}: "
            f"{rec['category']} | {direction_label} | "
            f"lon={rec['lon']:.3f}°, lat={rec['lat']:.3f}°",
            table={
                "sill deficit (m)":    f"{rec['sill_depth_coarse'] * (1 - rec['sill_ratio']):.1f}",
                "sill depth (fine)":   f"{rec['sill_depth_fine']:.1f} m",
                "sill depth (coarse)": f"{rec['sill_depth_coarse']:.1f} m",
                "sill ratio":          f"{rec['sill_ratio']:.3f}",
                "width (fine)":        f"{rec['width_km']:.2f} km",
                "area ratio":          f"{rec['area_ratio']:.3f}",
                "connected":           str(rec["connected"]),
                "suggested fix":       rec.get("suggested_fix", ""),
            },
            images=[img],
        )
    if n_total_straits > max_sections:
        rpt.add_section(
            f"… {n_total_straits - max_sections} more interface(s) not shown",
            text=(
                f"Set `analysis.max_section_profiles: {n_total_straits}` in your YAML "
                "config to generate profiles for all flagged interfaces. "
                "All interfaces are listed in the straits CSV and `fixes_suggested.yaml`."
            ),
        )

    # ------------------------------------------------------------------
    # Step 4e – Thalweg comparison (fine vs coarse)  [--thalweg / --no-thalweg]
    # ------------------------------------------------------------------
    if run_thalweg:
        logger.info("\n[4e/6] Computing thalwegs …")
        t0 = time.time()

        thalweg_records: list[dict] = []

        # With --skip-regrid the fine source was not loaded; load it now.
        # This is cheap relative to the MST build that follows.
        _thalweg_src = src
        if _thalweg_src is None and source is not None:
            logger.info("      Loading fine source for thalweg …")
            _t_src = time.time()
            try:
                _thalweg_src = reader.read_source(
                    source, dst_grid.lon_bounds, dst_grid.lat_bounds,
                    pad_deg=float(pad_deg),
                    emodnet_cache_dir=str(emodnet_cache),
                    emodnet_resolution=(float(emodnet_res)
                                        if emodnet_res is not None else None),
                )
                if coastline_res is not None:
                    _thalweg_src = reader.apply_coastline_mask(
                        _thalweg_src, resolution=str(coastline_res)
                    )
                logger.info("      Fine source loaded in %.1f s",
                            time.time() - _t_src)
            except Exception as exc:
                logger.warning("      Could not load fine source (%s) — "
                               "thalweg skipped.", exc)
                _thalweg_src = None

        if _thalweg_src is None:
            logger.warning("      Thalweg skipped: fine source unavailable.")
        else:
            # Mode A: strait-based (top straits from step 4d)
            if strait_records:
                tw_a = thalwegmod.compute_strait_thalwegs(src, dst, strait_records,
                                                           n_straits=max_sections)
                thalweg_records.extend(tw_a)
                logger.info("      strait-based: %d thalweg(s)", len(tw_a))

            # Always write the boundary CSV from the finalised mask so the
            # user has the file for reference.
            _auto_bdy_csv = os.path.join(report_dir, f"{name}_bdy.csv")
            if not _thalweg_boundaries_csv:
                _bdy_files, _bdy_segs = boundarymod.write_boundary_coords(
                    dst, report_dir, name
                )
                logger.info("      boundary CSV: %s (%d cell(s))",
                            _auto_bdy_csv,
                            sum(s["n_cells"] for s in _bdy_segs))

            # Mode B: boundary auto-detection thalwegs.
            # Disabled by default — enable with thalweg.boundary_thalwegs: true
            if run_boundary_thalweg:
                tw_b = thalwegmod.boundary_thalwegs(
                    _thalweg_src, dst,
                    min_sill_m=thalweg_min_sill,
                    max_detour=thalweg_bdy_detour,
                    sill_dedup_tol_m=thalweg_sill_dedup,
                    boundaries_csv=_thalweg_boundaries_csv,
                )
                thalweg_records.extend(tw_b)
                logger.info("      boundary auto: %d thalweg(s)", len(tw_b))
            else:
                logger.info("      boundary auto: disabled "
                            "(set thalweg.boundary_thalwegs: true to enable)")

            # Mode C: user waypoints
            if user_waypoints_cfg:
                tw_c = thalwegmod.waypoint_thalwegs(
                    _thalweg_src, dst, user_waypoints_cfg,
                    max_detour=thalweg_max_detour,
                )
                thalweg_records.extend(tw_c)
                logger.info("      waypoints:     %d thalweg(s)", len(tw_c))

        logger.info("      total: %d thalweg(s) in %.1f s",
                    len(thalweg_records), time.time() - t0)

        thalwegmod.print_thalweg_table(thalweg_records)
        tw_sum = thalwegmod.thalweg_summary(thalweg_records)
        report.print_table(tw_sum, title="Thalweg summary")

        for k, tw in enumerate(thalweg_records):
            cat = tw.get("category", "tw")
            tw_name = tw.get("name", f"thalweg_{k:03d}")
            safe = (tw_name.replace(" ", "_").replace("→", "-")
                    .replace("[", "").replace("]", ""))
            img = pfx + f"04e_thalweg_{k:03d}_{safe}.png"
            report.plot_thalweg_comparison(
                tw, _thalweg_src, dst,
                png_path=os.path.join(report_dir, img),
            )
            deficit = tw.get("sill_deficit_m", float("nan"))
            coarse_sill = tw["coarse"]["sill_depth"]
            rpt.add_section(
                f"Thalweg: {tw_name}",
                text=(
                    f"Category: **{cat}**.  "
                    f"Fine sill depth: **{tw['fine']['sill_depth']:.1f} m**.  "
                    f"Coarse sill depth: **{coarse_sill:.1f} m**.  "
                    f"Deficit: **{deficit:.1f} m**."
                    if np.isfinite(deficit) else
                    f"Category: **{cat}**.  Fine sill: {tw['fine']['sill_depth']:.1f} m."
                ),
                table={},
                images=[img],
            )
            csv_name = pfx + f"04e_thalweg_{k:03d}_{safe}.csv"
            thalwegmod.write_thalweg_csv(
                tw, csv_path=os.path.join(report_dir, csv_name)
            )
            logger.info("        wrote %s", csv_name)
            if tw.get("zoomable", False):
                html_name = pfx + f"04e_thalweg_{k:03d}_{safe}.html"
                report.plot_thalweg_html(
                    tw, html_path=os.path.join(report_dir, html_name)
                )
                logger.info("        wrote %s (zoomable)", html_name)

        if not thalweg_records:
            logger.info("      no thalwegs to plot")
    else:
        logger.info("\n[4e/6] Thalweg analysis skipped (use --thalweg to enable).")

    # ------------------------------------------------------------------
    # Step 5 – Optional rx0 smoothing (one pass per target value)
    # ------------------------------------------------------------------
    # List of (rx0_value, depth_smooth, corrections) for all successful runs
    smooth_variants: list[tuple[float, np.ndarray, np.ndarray]] = []

    if rx0_list:
        depth_arr = np.where(dst["mask"].values, dst["depth"].values, 0.0)
        mask_arr  = dst["mask"].values
        rx0_u_b, rx0_v_b = smoothmod.compute_rx0(depth_arr, mask_arr)
        rx0_before = np.maximum(
            np.pad(rx0_u_b, ((0, 0), (0, 1))),
            np.pad(rx0_v_b, ((0, 1), (0, 0))),
        )
        for rx0_val in sorted(rx0_list, reverse=True):   # coarsest first
            smooth_var = _smooth_var_name(rx0_val)
            logger.info(f"\n[5/6] rx0 smoothing (target={rx0_val}, variable → '{smooth_var}') …")
            t0 = time.time()
            depth_smooth, corrections = smoothmod.smooth_rx0(depth_arr, mask_arr,
                                                              rx0=rx0_val)
            rx0_u_a, rx0_v_a = smoothmod.compute_rx0(depth_smooth, mask_arr)
            smooth_sum = smoothmod.smooth_summary(
                depth_arr, depth_smooth, mask_arr, corrections, rx0_val
            )
            report.print_table(smooth_sum, title=f"rx0 smoothing → {smooth_var}")
            logger.info(f"      done in {time.time()-t0:.1f} s")

            rx0_after = np.maximum(
                np.pad(rx0_u_a, ((0, 0), (0, 1))),
                np.pad(rx0_v_a, ((0, 1), (0, 0))),
            )
            report.plot_rx0_diagnostics(
                rx0_before, rx0_after, corrections,
                dst.lon.values, dst.lat.values, mask_arr,
                path_prefix=os.path.join(report_dir, pfx + f"05_smooth_{smooth_var}"),
            )
            rpt.add_section(
                f"rx0 smoothing (`{smooth_var}`)",
                text=(
                    f"Linear-programming smoothing minimising depth corrections "
                    f"subject to rx0 ≤ {rx0_val} at every wet interface.  "
                    f"Smoothed field saved as NetCDF variable **`{smooth_var}`** "
                    f"alongside the unsmoothed `depth`, allowing multiple "
                    f"bathymetry variants in a single file."
                ),
                table=smooth_sum,
                images=[
                    pfx + f"05_smooth_{smooth_var}_rx0_histogram.png",
                    pfx + f"05_smooth_{smooth_var}_corrections.png",
                ],
            )
            smooth_variants.append((rx0_val, depth_smooth, corrections))
    else:
        logger.info("\n[5/6] Smoothing skipped.")

    # ------------------------------------------------------------------
    # Step 6 – Build output dataset, final plots for every depth variant
    # ------------------------------------------------------------------
    logger.info("\n[6/6] Writing output …")

    # C-grid staggered depths: U = east face, V = north face of each T-cell.
    # Both have the same shape [ny, nx] as the T-point depth.
    # NaN propagates: a face is land if either bordering T-cell is land.
    depth_t = np.where(dst["mask"].values, dst["depth"].values, np.nan)
    depth_u_vals, depth_v_vals = interpolate.compute_cgrid_depth(depth_t)

    out_vars: dict = {
        "depth": dst["depth"],
        "depth_u": xr.DataArray(
            depth_u_vals, dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": "Sea floor depth at eastern U-face", "units": "m",
                   "comment": "min(depth_t[i,j], depth_t[i,j+1]); boundary = depth_t"},
        ),
        "depth_v": xr.DataArray(
            depth_v_vals, dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": "Sea floor depth at northern V-face", "units": "m",
                   "comment": "min(depth_t[i,j], depth_t[i+1,j]); boundary = depth_t"},
        ),
        "wet_fraction": dst["wet_fraction"],
        "mask": dst["mask"],
    }
    if "basin_labels" in dst.data_vars:
        out_vars["basin_labels"] = dst["basin_labels"]
    if dst2_mask is not None and dst2_depth_vals is not None:
        out_vars["depth_source2"] = xr.DataArray(
            np.where(dst2_mask, dst2_depth_vals, np.nan),
            dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": f"Sea floor depth from {src2_label}", "units": "m"},
        )
    if dst2_mask is not None:
        comp_arr = np.zeros(dst["mask"].shape, dtype=np.int8)
        comp_arr[(dst["mask"].values == 1) & (dst2_mask == 1)] = 1
        comp_arr[(dst["mask"].values == 1) & (dst2_mask == 0)] = 2
        comp_arr[(dst["mask"].values == 0) & (dst2_mask == 1)] = 3
        out_vars["source_comparison"] = xr.DataArray(
            comp_arr, dims=["lat", "lon"], coords=dst.coords,
            attrs={
                "long_name": "Source mask comparison",
                "flag_values": "0 1 2 3",
                "flag_meanings": "common_land common_water source1_only source2_only",
                "source1": str(source),
                "source2": str(source2),
            },
        )
    for rx0_val, d_smooth, d_corr in smooth_variants:
        sv = _smooth_var_name(rx0_val)
        out_vars[sv] = xr.DataArray(
            np.where(dst["mask"].values, d_smooth, np.nan),
            dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": f"Bathymetry smoothed to rx0≤{rx0_val}",
                   "units": "m", "rx0_target": rx0_val},
        )
        out_vars[f"depth_corrections_{sv}"] = xr.DataArray(
            d_corr, dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": f"Depth corrections from rx0 smoothing (rx0≤{rx0_val})",
                   "units": "m"},
        )

    out_ds = xr.Dataset(out_vars, coords=dst.coords)
    out_ds["depth"].attrs.update({"long_name": "Sea floor depth", "units": "m"})
    out_ds["mask"].attrs.update({"long_name": "Ocean mask (1=ocean, 0=land)"})
    out_ds.to_netcdf(output_file)

    # Produce one final plot per depth variable (raw + each smoothed version).
    # Shared colour scale so panels are visually comparable when diffed later.
    depth_variants: list[tuple[str, np.ndarray, str]] = [
        ("depth", dst["depth"].values, "unsmoothed"),
    ]
    for rx0_val, d_smooth, _ in smooth_variants:
        sv = _smooth_var_name(rx0_val)
        depth_variants.append(
            (sv, np.where(dst["mask"].values, d_smooth, np.nan), f"rx0≤{rx0_val}")
        )

    subtitle = _plot_subtitle(dst_grid, dst)
    final_plots: list[str] = []
    for var_name, depth_arr, label in depth_variants:
        fname = pfx + f"06_final_{var_name}.png"
        report.plot_depth(
            dst.lon.values, dst.lat.values,
            depth_arr, dst["mask"].values,
            title=f"{name} — {label}",
            subtitle=subtitle,
            path=os.path.join(report_dir, fname),
            interactive=True,
            log_scale=log_depth_scale,
        )
        final_plots.append(fname)
        logger.info(f"  {fname}")

    # Depth difference plots between the two sources (one per depth variant)
    if dst2_mask is not None and dst2_depth_vals is not None:
        common_ocean = (dst["mask"].values == 1) & (dst2_mask == 1)
        for var_name, depth_arr, label in depth_variants:
            diff_arr = np.where(common_ocean, depth_arr - dst2_depth_vals, np.nan)
            diff_fname = pfx + f"06_diff_{var_name}.png"
            report.plot_depth_diff(
                dst.lon.values, dst.lat.values,
                diff_arr,
                title=f"{name} — depth difference: {src1_label} − {src2_label} ({label})",
                subtitle=subtitle,
                path=os.path.join(report_dir, diff_fname),
            )
            final_plots.append(diff_fname)
            logger.info(f"  {diff_fname}")

    var_list = ", ".join(f"`{v}`" for v in out_vars if v not in ("wet_fraction", "mask", "basin_labels", "depth_corrections"))
    rpt.add_section(
        "Output",
        text=(
            f"Output written to `{output_file}`.\n\n"
            f"Depth variables: {var_list}. "
            "Each has a corresponding PNG and interactive HTML final plot."
        ),
        images=final_plots,
    )

    if args.write_boundaries:
        logger.info("\nWriting boundary coordinate files …")
        bdy_dir = os.path.dirname(os.path.abspath(output_file))
        bdy_files, bdy_segments = boundarymod.write_boundary_coords(dst, bdy_dir, name)
        for f in bdy_files:
            logger.info(f"  {os.path.join(bdy_dir, f)}")
        # Build segment summary for the report
        seg_table: dict = {}
        for s in bdy_segments:
            key = f"{s['side']} seg {s['segment']}"
            seg_table[key] = (
                f"i={s['i_start']}..{s['i_end']},  "
                f"j={s['j_start']}..{s['j_end']},  "
                f"n={s['n_cells']}"
            )
        rpt.add_section(
            "Open boundaries",
            text=(
                f"T-grid coordinate file: `{bdy_files[0]}`  \n"
                f"{sum(s['n_cells'] for s in bdy_segments)} boundary cells across "
                f"{len(bdy_segments)} segment(s).  "
                "Sides: west (S→N), north (W→E), east (S→N), south (W→E).  "
                "Corner ownership: west/east include corners; north/south start at i=1."
            ),
            table=seg_table,
        )

    report_md = os.path.join(report_dir, pfx + "report.md")
    rpt.write(report_md)

    logger.info(f"\nDone.")
    logger.info(f"  Output NetCDF : {output_file}")
    logger.info(f"  Markdown report: {report_md}")
    logger.info(f"  Report dir    : {report_dir}/")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
