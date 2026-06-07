"""bathymetry-regrid: conservative bathymetry interpolation pipeline.

Configuration can be provided entirely via a YAML file (``--config``).
Any option given on the command line overrides the corresponding YAML value.

Example YAML (``north_sea.yaml``)::

    source: /server/data/GEBCO/GEBCO_2023.nc
    pad_deg: 1.0

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

    regridding:
      cache_dir: ./regrid_weights
      min_depth: 2.0
      min_wet_fraction: 0.05

    analysis:
      nkeep_basins: 1
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

    bathymetry-regrid --config north_sea.yaml --smooth-rx0 0.15 --output bathy_v2.nc
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Allow importing lib/ modules without installation
_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
if _lib not in sys.path:
    sys.path.insert(0, os.path.abspath(_lib))

import numpy as np

import analysis
import grid as gridmod
import interpolate
import reader
import report
import smooth as smoothmod

# ---------------------------------------------------------------------------
# YAML config loading and merge
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


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
        equidist = _merge(args.equidistant, cfg, "grid", "equidistant", default=False)
        for name, val in [("lon-min", lon_min), ("lon-max", lon_max),
                          ("lat-min", lat_min), ("lat-max", lat_max)]:
            if val is None:
                raise ValueError(f"Missing required grid parameter: {name}")
        if equidist:
            if dlon is None and dlat is None:
                raise ValueError("--equidistant requires at least one of --dlon / --dlat")
            lat_center = (float(lat_min) + float(lat_max)) / 2.0
            cos_lat = math.cos(math.radians(lat_center))
            if dlon is None:
                dlon = round(float(dlat) / cos_lat, 6)
            elif dlat is None:
                dlat = round(float(dlon) * cos_lat, 6)
            nx_eq = round((float(lon_max) - float(lon_min)) / float(dlon))
            ny_eq = round((float(lat_max) - float(lat_min)) / float(dlat))
            print(f"      [equidistant] lat_center={lat_center:.2f}°  "
                  f"dlon={float(dlon):.6g}°  dlat={float(dlat):.6g}°  "
                  f"→ grid {nx_eq} × {ny_eq} (lon × lat)")
        else:
            for name, val in [("dlon", dlon), ("dlat", dlat)]:
                if val is None:
                    raise ValueError(f"Missing required grid parameter: {name}")
        return gridmod.SphericalGrid(lon_min, lon_max, lat_min, lat_max,
                                     float(dlon), float(dlat), float(rot))

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
                         help='GEBCO NetCDF path or "emodnet".')
    src_grp.add_argument("--pad-deg", type=float, default=None,
                         help="Padding (degrees) around target grid when reading source.")

    # Grid
    grd = parser.add_argument_group("Grid type")
    grd.add_argument("--grid", choices=["spherical", "cartesian"], default=None)
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
                    help="Flag cells with wet_fraction below this.")

    # Analysis
    an = parser.add_argument_group("Analysis")
    an.add_argument("--nkeep-basins", type=int, default=None,
                    help="Number of connected ocean basins to retain.")
    an.add_argument("--wet-frac-threshold", type=float, default=None)
    an.add_argument("--sill-ratio-threshold", type=float, default=None)
    an.add_argument("--area-ratio-threshold", type=float, default=None)

    # Smoothing
    sm = parser.add_argument_group("Smoothing")
    sm.add_argument("--smooth-rx0", type=float, default=None,
                    help="Target rx0 (omit to skip smoothing).")

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

    # Output
    out = parser.add_argument_group("Output")
    out.add_argument("--name", default=None,
                     help="Short identifier for this bathymetry (used in file names). "
                          "E.g. 'northsea_0p05deg'. Derived from source name if omitted.")
    out.add_argument("--output", default=None, help="Output NetCDF file.")
    out.add_argument("--report-dir", default=None,
                     help="Directory for report figures, CSV, and Markdown.")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load YAML and resolve final parameter values
    # ------------------------------------------------------------------
    cfg: dict = _load_yaml(args.config) if args.config else {}

    source        = _merge(args.source,       cfg, "source")
    pad_deg       = _merge(args.pad_deg,      cfg, "pad_deg",            default=1.0)
    cache_dir     = _merge(args.cache_dir,    cfg, "regridding", "cache_dir",
                           default="./regrid_weights")
    min_depth     = _merge(args.min_depth,    cfg, "regridding", "min_depth",     default=0.0)
    min_wf        = _merge(args.min_wet_fraction, cfg, "regridding", "min_wet_fraction",
                           default=0.0)
    nkeep         = _merge(args.nkeep_basins, cfg, "analysis",   "nkeep_basins",  default=1)
    wf_thr        = _merge(args.wet_frac_threshold,  cfg, "analysis", "wet_frac_threshold",
                           default=0.3)
    sill_thr      = _merge(args.sill_ratio_threshold, cfg, "analysis", "sill_ratio_threshold",
                           default=0.7)
    area_thr      = _merge(args.area_ratio_threshold, cfg, "analysis", "area_ratio_threshold",
                           default=0.5)
    rx0           = _merge(args.smooth_rx0,   cfg, "smooth", "rx0")   # None = skip
    fixes_list    = list(_nested_get(cfg, "fixes") or [])

    if source is None:
        parser.error("--source (or 'source:' in YAML) is required")

    # ------------------------------------------------------------------
    # Resolve 'name' — used as prefix for output files and report dir
    # ------------------------------------------------------------------
    name = _merge(args.name, cfg, "name")
    if name is None:
        # Derive from source file stem, stripping common suffixes
        stem = Path(source).stem if source != "emodnet" else "emodnet"
        name = stem.replace(" ", "_")

    # Defaults that depend on name
    output_file = _merge(args.output, cfg, "output", "file",
                         default=f"{name}.nc")
    report_dir  = _merge(args.report_dir, cfg, "output", "report_dir",
                         default=f"./report/{name}")

    os.makedirs(report_dir, exist_ok=True)

    # Load extra fixes from --fixes-file or --accept-fixes
    fixes_file = args.fixes_file
    if not fixes_file and args.accept_fixes:
        fixes_file = os.path.join(report_dir, "fixes_suggested.yaml")
    if fixes_file:
        if not os.path.exists(fixes_file):
            parser.error(f"fixes file not found: {fixes_file}")
        extra = _load_yaml(fixes_file).get("fixes") or []
        fixes_list = fixes_list + extra
        print(f"Loaded {len(extra)} fix(es) from {fixes_file}")

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
    print(f"\n[1/6] Building target grid … (name='{name}')")
    dst_grid = _build_grid(cfg, args)
    grid_summary = dst_grid.summary()
    report.print_table(grid_summary, title="Target grid")
    rpt.add_section(
        "Target grid",
        text=f"Grid type: **{type(dst_grid).__name__}**, "
             f"{dst_grid.nx} × {dst_grid.ny} cells.",
        table=grid_summary,
    )

    # ------------------------------------------------------------------
    # Step 2 – Read source bathymetry
    # ------------------------------------------------------------------
    print("\n[2/6] Reading source bathymetry …")
    t0 = time.time()
    src = reader.read_source(
        source, dst_grid.lon_bounds, dst_grid.lat_bounds, pad_deg=float(pad_deg)
    )
    src_sum = reader.source_summary(src)
    report.print_table(src_sum, title="Source bathymetry")
    print(f"      done in {time.time()-t0:.1f} s")

    src_plot = pfx + "02_source_depth.png"
    report.plot_depth(
        src.lon.values, src.lat.values,
        src["depth"].values, (~src["land"].values).astype(float),
        title=f"Source: {Path(source).name if source != 'emodnet' else 'EMODnet'}",
        path=os.path.join(report_dir, src_plot),
    )
    rpt.add_section(
        "Source bathymetry",
        text=(
            f"Source data read and clipped to the target domain "
            f"(±{pad_deg}° buffer applied)."
        ),
        table=src_sum,
        images=[src_plot],
    )

    # ------------------------------------------------------------------
    # Step 3 – ESMF conservative interpolation
    # ------------------------------------------------------------------
    print("\n[3/6] Conservative regridding (xESMF) …  (may take a minute for large grids)")
    t0 = time.time()
    dst = interpolate.regrid(
        src, dst_grid,
        min_depth=float(min_depth),
        min_wet_fraction=float(min_wf),
        cache_dir=str(cache_dir),
    )
    dst_sum = interpolate.regrid_summary(dst, dst_grid)
    report.print_table(dst_sum, title="Regridded destination")
    print(f"      done in {time.time()-t0:.1f} s")

    regrid_plot = pfx + "03_regrid_result.png"
    wf_plot = pfx + "03_wet_fraction.png"
    report.plot_depth(
        dst.lon.values, dst.lat.values,
        dst["depth"].values, dst["mask"].values,
        title=f"{name} — regridded depth (m)",
        path=os.path.join(report_dir, regrid_plot),
        interactive=True,
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
    # Step 4a – Strait detection
    # ------------------------------------------------------------------
    print("\n[4a/6] Detecting narrow straits …")
    t0 = time.time()

    strait_records = analysis.find_straits(
        src, dst, dst_grid,
        wet_frac_threshold=float(wf_thr),
        sill_ratio_threshold=float(sill_thr),
        area_ratio_threshold=float(area_thr),
    )

    strait_sum = analysis.strait_summary(strait_records)
    report.print_table(strait_sum, title="Strait detection")
    print(f"      {len(strait_records)} interfaces flagged in {time.time()-t0:.1f} s")

    clean_records = [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in strait_records]
    csv_path = os.path.join(report_dir, pfx + "04a_straits.csv")
    report.save_csv(clean_records, csv_path)

    fixes_yaml_path = os.path.join(report_dir, "fixes_suggested.yaml")
    report.save_fixes_yaml(clean_records, fixes_yaml_path)

    straits_plot = pfx + "04a_straits.png"
    report.plot_straits(
        dst.lon.values, dst.lat.values,
        dst["depth"].values, dst["mask"].values,
        strait_records=clean_records,
        path=os.path.join(report_dir, straits_plot),
        # Small inset showing the domain in regional geographic context
        domain_bounds=(
            dst_grid.lon_bounds[0], dst_grid.lon_bounds[1],
            dst_grid.lat_bounds[0], dst_grid.lat_bounds[1],
        ),
    )

    # Cross-section profiles for the first 10 flagged interfaces.
    # Each inset is zoomed to a tight area around the specific interface
    # (~10 coarse cells on each side) so the local geography is visible.
    section_images = []
    _zoom_lon = getattr(dst_grid, "dlon", 0.1) * 10
    _zoom_lat = getattr(dst_grid, "dlat", 0.1) * 10
    for k, rec in enumerate(strait_records[:10]):
        dist_key = "_dlat_km" if rec["direction"] == "U" else "_dlon_km"
        depth_sec = rec["_depth_section"]
        cell_km = rec.get(dist_key, 0.1)
        dist = np.arange(len(depth_sec)) * cell_km
        img = pfx + f"04a_section_{k:03d}.png"
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
        )
        section_images.append(img)

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
            "sill-depth deficit, and connectivity breaks.\n\n"
            f"Suggested fixes written to `{fixes_yaml_path}`. "
            "To adopt: copy the relevant entries into the `fixes:` section of your YAML "
            "config and re-run."
        ),
        table=strait_sum,
        images=[straits_plot] + section_images,
        warnings=warn_msgs,
    )

    # ------------------------------------------------------------------
    # Step 4b – Apply user fixes (if any)
    # ------------------------------------------------------------------
    if fixes_list:
        print(f"\n[4b] Applying {len(fixes_list)} fix(es) from config …")
        dst, applied_fixes = analysis.apply_fixes(dst, fixes_list)
        report.print_table({"fixes applied": len(applied_fixes)}, title="User fixes")
        report.save_csv(applied_fixes, os.path.join(report_dir, pfx + "04b_fixes_applied.csv"))
        rpt.add_section(
            "User-specified fixes",
            text=f"{len(applied_fixes)} fix(es) applied from configuration.",
            table_rows=applied_fixes,
        )
    else:
        print("\n[4b] No fixes configured — skipping.")

    # ------------------------------------------------------------------
    # Step 4c – Explicit mask regions (force areas to land)
    # Runs BEFORE isolated-cell masking so that blocking a fjord mouth
    # causes the interior cells to be picked up as isolated and removed.
    # ------------------------------------------------------------------
    mask_regions_list = _nested_get(cfg, "mask_regions") or []
    if mask_regions_list:
        print(f"\n[4c] Applying {len(mask_regions_list)} explicit mask region(s) …")
        dst, mr_applied = analysis.apply_mask_regions(dst, mask_regions_list)
        mr_sum = analysis.mask_regions_summary(mr_applied)
        report.print_table(mr_sum, title="Explicit mask regions")
        report.save_csv(mr_applied, os.path.join(report_dir, pfx + "04c_mask_regions.csv"))
        mr_plot = pfx + "04c_mask_regions.png"
        report.plot_depth(
            dst.lon.values, dst.lat.values,
            dst["depth"].values, dst["mask"].values,
            title=f"{name} — after explicit masking",
            path=os.path.join(report_dir, mr_plot),
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
        print("\n[4c] No explicit mask regions configured — skipping.")

    # ------------------------------------------------------------------
    # Step 4d – Isolated-cell masking
    # Runs AFTER explicit masking so fjord interiors (whose mouth was
    # just closed) are correctly identified as disconnected and removed.
    # ------------------------------------------------------------------
    print(f"\n[4d/6] Masking isolated ocean regions (keep {nkeep}) …")
    dst_clean, basin_records = analysis.mask_isolated(dst, nkeep=int(nkeep))
    iso_sum = analysis.isolation_summary(basin_records)
    report.print_table(iso_sum, title="Isolated cells")
    report.save_csv(basin_records, os.path.join(report_dir, pfx + "04d_basins.csv"))
    basins_plot = pfx + "04d_basins.png"
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
            "whose mouth was closed above are removed here automatically."
        ),
        table=iso_sum,
        images=[basins_plot],
    )
    dst = dst_clean

    # ------------------------------------------------------------------
    # Step 5 – Optional rx0 smoothing
    # ------------------------------------------------------------------
    depth_smooth = None
    corrections = None
    if rx0 is not None:
        smooth_var = _smooth_var_name(float(rx0))
        print(f"\n[5/6] rx0 smoothing (target={rx0}, variable → '{smooth_var}') …")
        t0 = time.time()
        depth_arr  = np.where(dst["mask"].values, dst["depth"].values, 0.0)
        mask_arr   = dst["mask"].values
        rx0_u_b, rx0_v_b = smoothmod.compute_rx0(depth_arr, mask_arr)
        depth_smooth, corrections = smoothmod.smooth_rx0(depth_arr, mask_arr, rx0=float(rx0))
        rx0_u_a, rx0_v_a = smoothmod.compute_rx0(depth_smooth, mask_arr)
        smooth_sum = smoothmod.smooth_summary(
            depth_arr, depth_smooth, mask_arr, corrections, float(rx0)
        )
        report.print_table(smooth_sum, title=f"rx0 smoothing → {smooth_var}")
        print(f"      done in {time.time()-t0:.1f} s")

        rx0_before = np.maximum(
            np.pad(rx0_u_b, ((0, 0), (0, 1))),
            np.pad(rx0_v_b, ((0, 1), (0, 0))),
        )
        rx0_after = np.maximum(
            np.pad(rx0_u_a, ((0, 0), (0, 1))),
            np.pad(rx0_v_a, ((0, 1), (0, 0))),
        )
        report.plot_rx0_diagnostics(
            rx0_before, rx0_after, corrections,
            dst.lon.values, dst.lat.values, mask_arr,
            path_prefix=os.path.join(report_dir, pfx + "05_smooth"),
        )
        rpt.add_section(
            f"rx0 smoothing (`{smooth_var}`)",
            text=(
                f"Linear-programming smoothing minimising depth corrections "
                f"subject to rx0 ≤ {rx0} at every wet interface.  "
                f"Smoothed field saved as NetCDF variable **`{smooth_var}`** "
                f"alongside the unsmoothed `depth`, allowing multiple "
                f"bathymetry variants in a single file."
            ),
            table=smooth_sum,
            images=[
                pfx + "05_smooth_rx0_histogram.png",
                pfx + "05_smooth_corrections.png",
            ],
        )
    else:
        print("\n[5/6] Smoothing skipped.")

    # ------------------------------------------------------------------
    # Step 6 – Build output dataset, final plots for every depth variant
    # ------------------------------------------------------------------
    print("\n[6/6] Writing output …")

    import xarray as xr

    out_vars: dict = {
        "depth": dst["depth"],
        "wet_fraction": dst["wet_fraction"],
        "mask": dst["mask"],
    }
    if "basin_labels" in dst.data_vars:
        out_vars["basin_labels"] = dst["basin_labels"]
    if depth_smooth is not None and rx0 is not None:
        smooth_var = _smooth_var_name(float(rx0))
        out_vars[smooth_var] = xr.DataArray(
            np.where(dst["mask"].values, depth_smooth, np.nan),
            dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": f"Bathymetry smoothed to rx0≤{rx0}",
                   "units": "m", "rx0_target": float(rx0)},
        )
        out_vars["depth_corrections"] = xr.DataArray(
            corrections, dims=["lat", "lon"], coords=dst.coords,
            attrs={"long_name": "Depth corrections from rx0 smoothing", "units": "m"},
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
    if depth_smooth is not None and rx0 is not None:
        depth_variants.append(
            (smooth_var, np.where(dst["mask"].values, depth_smooth, np.nan),
             f"rx0≤{rx0}")
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
        )
        final_plots.append(fname)
        print(f"  {fname}")

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

    report_md = os.path.join(report_dir, pfx + "report.md")
    rpt.write(report_md)

    print(f"\nDone.")
    print(f"  Output NetCDF : {output_file}")
    print(f"  Markdown report: {report_md}")
    print(f"  Report dir    : {report_dir}/")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
