#!/usr/bin/env python3
"""
estuary-morphology: cross-sectional area and volume along an estuary thalweg.

Reads a bathymetry NetCDF produced by bathymetry-regrid, casts perpendicular
cross-sections at user-defined waypoints, and reports widths, areas, and
volumes per station.  Multiple branches (tributaries) are supported.

Usage:
    estuary-morphology --config tamar_morphology.yaml
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import xarray as xr
import yaml

# Allow running without installation (lib/ added to path at build time when installed)
_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
if _lib not in sys.path:
    sys.path.insert(0, os.path.abspath(_lib))

import morphology as morph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / I-O helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_waypoints(raw: list) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            out.append((float(item[0]), float(item[1])))
        elif isinstance(item, dict):
            out.append((float(item["lon"]), float(item["lat"])))
        else:
            raise ValueError(f"Unrecognised waypoint format: {item!r}")
    return out


def _load_bathymetry(
    nc_path: str, depth_var: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load lon_2d, lat_2d, depth_2d from a bathymetry-regrid output NetCDF."""
    ds = xr.open_dataset(nc_path)

    if depth_var is None:
        for candidate in ("depth", "depth_fixes", "depth_raw"):
            if candidate in ds:
                depth_var = candidate
                break
        if depth_var is None:
            dvars = [v for v in ds.data_vars if "depth" in str(v).lower()]
            if not dvars:
                raise ValueError(f"No depth variable found in {nc_path}")
            depth_var = dvars[0]
    logger.info(f"Depth variable: '{depth_var}'")

    depth = ds[depth_var].values  # (ny, nx)
    lon_arr = ds["lon"].values
    lat_arr = ds["lat"].values

    # Coordinates may be 1-D (rare for pipeline output) or 2-D
    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
    else:
        lon_2d = lon_arr
        lat_2d = lat_arr

    return lon_2d, lat_2d, depth


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(
    all_stations: list[tuple[str, list[dict]]], path: str
) -> None:
    fields = [
        "branch", "station", "lon", "lat", "s_km",
        "width_m", "area_m2", "dx_m", "volume_Mm3", "cumvol_Mm3",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for branch_name, stations in all_stations:
            for k, st in enumerate(stations):
                w.writerow({
                    "branch":      branch_name,
                    "station":     k + 1,
                    "lon":         f"{st['lon']:.6f}",
                    "lat":         f"{st['lat']:.6f}",
                    "s_km":        f"{st['s_m'] / 1000:.3f}",
                    "width_m":     f"{st['width_m']:.1f}",
                    "area_m2":     f"{st['area_m2']:.1f}",
                    "dx_m":        f"{st['dx_m']:.1f}",
                    "volume_Mm3":  f"{st['volume_m3'] / 1e6:.6f}",
                    "cumvol_Mm3":  f"{st['cumvol_m3'] / 1e6:.6f}",
                })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_map(
    all_stations: list[tuple[str, list[dict]]],
    lon_2d: np.ndarray,
    lat_2d: np.ndarray,
    depth_2d: np.ndarray,
    report_dir: str,
    name: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(10, 9))

    # Bathymetry background
    depth_plot = np.where(depth_2d > 0, depth_2d, np.nan)
    vmax = float(np.nanpercentile(depth_plot, 98)) if np.any(np.isfinite(depth_plot)) else 50.0
    pcm = ax.pcolormesh(
        lon_2d, lat_2d, depth_plot,
        cmap="Blues", vmin=0.0, vmax=vmax,
        shading="nearest", zorder=0,
    )
    plt.colorbar(pcm, ax=ax, label="Depth (m)", shrink=0.7, pad=0.01)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for bi, (branch_name, stations) in enumerate(all_stations):
        col = prop_cycle[bi % len(prop_cycle)]

        # Filled trapezoids between consecutive cross-sections
        patches = []
        patch_areas = []
        for i in range(len(stations) - 1):
            ll = stations[i]["left_bank"]
            rl = stations[i]["right_bank"]
            lr = stations[i + 1]["left_bank"]
            rr = stations[i + 1]["right_bank"]
            verts = [ll, rl, rr, lr]
            patches.append(Polygon(verts, closed=True))
            patch_areas.append(
                (stations[i]["area_m2"] + stations[i + 1]["area_m2"]) / 2.0
            )

        if patches:
            pc = PatchCollection(patches, cmap="YlOrRd_r", alpha=0.55, zorder=1,
                                 linewidths=0.4, edgecolors="0.5")
            pc.set_array(np.array(patch_areas))
            ax.add_collection(pc)
            plt.colorbar(pc, ax=ax, label="Mean cross-section area (m²)",
                         shrink=0.5, pad=0.08)

        # Cross-section lines
        for st in stations:
            llon, llat = st["left_bank"]
            rlon, rlat = st["right_bank"]
            ax.plot([llon, rlon], [llat, rlat], "-", color=col, alpha=0.7,
                    lw=0.8, zorder=2)

        # Thalweg
        tlons = [st["lon"] for st in stations]
        tlats = [st["lat"] for st in stations]
        ax.plot(tlons, tlats, "-o", color=col, ms=3.5, lw=1.5, zorder=4,
                label=f"{branch_name}")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{name} — estuary cross-sections")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")

    out = os.path.join(report_dir, f"{name}_morphology_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Map:      {out}")


def _plot_profiles(
    all_stations: list[tuple[str, list[dict]]],
    report_dir: str,
    name: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for bi, (branch_name, stations) in enumerate(all_stations):
        col = prop_cycle[bi % len(prop_cycle)]
        s_km       = [st["s_m"] / 1000.0 for st in stations]
        areas      = [st["area_m2"]       for st in stations]
        cumvols    = [st["cumvol_m3"] / 1e6 for st in stations]

        axes[0].plot(s_km, areas, "-o", color=col, ms=4, lw=1.5, label=branch_name)
        axes[1].plot(s_km, cumvols, "-o", color=col, ms=4, lw=1.5, label=branch_name)

    axes[0].set_ylabel("Cross-sectional area (m²)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Cumulative volume (Mm³)")
    axes[1].set_xlabel("Distance from mouth (km)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[0].set_title(f"{name} — morphology profiles")

    out = os.path.join(report_dir, f"{name}_morphology_profiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Profiles: {out}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_report(
    all_stations: list[tuple[str, list[dict]]],
    grand_total: float,
    cfg: dict,
    report_dir: str,
    name: str,
) -> str:
    import datetime

    map_png      = f"{name}_morphology_map.png"
    profiles_png = f"{name}_morphology_profiles.png"
    csv_file     = f"{name}_morphology.csv"
    nc_path      = cfg.get("bathymetry", "—")
    morph_cfg    = cfg.get("morphology", {})
    sample_ds    = morph_cfg.get("sample_ds", 50.0)
    max_hw       = morph_cfg.get("max_half_width_m", 10_000.0)
    now          = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append(f"# {name} — estuary morphology\n")
    lines.append(f"Generated: {now}\n")
    lines.append("")
    lines.append("## Configuration\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Bathymetry | `{nc_path}` |")
    lines.append(f"| sample_ds | {sample_ds} m |")
    lines.append(f"| max_half_width | {max_hw} m |")
    lines.append("")

    lines.append("## Cross-section map\n")
    lines.append(f"![Cross-section map]({map_png})\n")

    lines.append("## Morphology profiles\n")
    lines.append(f"![Morphology profiles]({profiles_png})\n")

    for branch_name, stations in all_stations:
        bvol = stations[-1]["cumvol_m3"]
        lines.append(f"## Branch: {branch_name}\n")
        lines.append(f"Total thalweg length: {stations[-1]['s_m']/1000:.2f} km  |  "
                     f"Total volume: **{bvol/1e6:.3f} Mm³**\n")
        lines.append("| # | lon | lat | s (km) | width (m) | area (m²) | dx (m) | volume (Mm³) | Σ volume (Mm³) |")
        lines.append("|---|-----|-----|--------|-----------|-----------|--------|--------------|----------------|")
        for k, st in enumerate(stations):
            lines.append(
                f"| {k+1} "
                f"| {st['lon']:.4f} "
                f"| {st['lat']:.4f} "
                f"| {st['s_m']/1000:.2f} "
                f"| {st['width_m']:.0f} "
                f"| {st['area_m2']:.0f} "
                f"| {st['dx_m']:.0f} "
                f"| {st['volume_m3']/1e6:.4f} "
                f"| {st['cumvol_m3']/1e6:.4f} |"
            )
        lines.append("")

    if len(all_stations) > 1:
        lines.append("## Volume summary\n")
        lines.append("| Branch | Volume (Mm³) |")
        lines.append("|--------|-------------|")
        for branch_name, stations in all_stations:
            lines.append(f"| {branch_name} | {stations[-1]['cumvol_m3']/1e6:.3f} |")
        lines.append(f"| **Grand total** | **{grand_total/1e6:.3f}** |")
        lines.append("")

    lines.append(f"*Full per-station data: [{csv_file}]({csv_file})*\n")

    md_path = os.path.join(report_dir, f"{name}_morphology_report.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return md_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute estuary cross-section areas and volumes along a thalweg"
    )
    parser.add_argument("--config", required=True, metavar="YAML",
                        help="YAML config file (see config/tamar_morphology.yaml)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg       = _load_config(args.config)
    name      = cfg["name"]
    nc_path   = cfg["bathymetry"]
    morph_cfg = cfg.get("morphology", {})
    sample_ds      = float(morph_cfg.get("sample_ds", 50.0))
    max_half_width = float(morph_cfg.get("max_half_width_m", 10_000.0))

    out_cfg    = cfg.get("output", {})
    report_dir = out_cfg.get("report_dir", f"./report/{name}_morphology")
    os.makedirs(report_dir, exist_ok=True)

    logger.info(f"Loading {nc_path}")
    lon_2d, lat_2d, depth_2d = _load_bathymetry(nc_path, cfg.get("depth_var"))
    logger.info(f"Grid shape: {depth_2d.shape}  (ny × nx)")

    logger.info("Building depth lookup…")
    lookup = morph.build_depth_lookup(lon_2d, lat_2d, depth_2d)

    # Build branch list (support both flat `waypoints:` and `branches:`)
    branches_cfg = morph_cfg.get("branches")
    if branches_cfg is None:
        if "waypoints" not in morph_cfg:
            raise ValueError("Config must contain morphology.branches or morphology.waypoints")
        branches_cfg = [{"name": name, "waypoints": morph_cfg["waypoints"]}]

    all_stations: list[tuple[str, list[dict]]] = []
    grand_total = 0.0

    for br in branches_cfg:
        bname = str(br.get("name", "branch"))
        wpts  = _parse_waypoints(br["waypoints"])
        logger.info(f"Branch '{bname}': {len(wpts)} stations  "
                    f"sample_ds={sample_ds} m  max_half_width={max_half_width} m")

        stations = morph.compute_morphology(wpts, lookup, sample_ds, max_half_width)
        all_stations.append((bname, stations))

        header = f"  {'#':>4}  {'s(km)':>8}  {'W(m)':>7}  {'A(m²)':>9}  {'dx(m)':>7}  {'V(Mm³)':>9}  {'ΣV(Mm³)':>9}"
        logger.info(header)
        for k, st in enumerate(stations):
            logger.info(
                f"  {k+1:4d}  {st['s_m']/1000:8.2f}  {st['width_m']:7.0f}  "
                f"{st['area_m2']:9.0f}  {st['dx_m']:7.0f}  "
                f"{st['volume_m3']/1e6:9.4f}  {st['cumvol_m3']/1e6:9.4f}"
            )
        bvol = stations[-1]["cumvol_m3"]
        grand_total += bvol
        logger.info(f"  → '{bname}' total: {bvol/1e6:.4f} Mm³")

    logger.info(f"Grand total volume: {grand_total/1e6:.4f} Mm³")

    # Write CSV
    csv_path = os.path.join(report_dir, f"{name}_morphology.csv")
    _write_csv(all_stations, csv_path)
    logger.info(f"CSV:      {csv_path}")

    # Plots
    _plot_map(all_stations, lon_2d, lat_2d, depth_2d, report_dir, name)
    _plot_profiles(all_stations, report_dir, name)

    # Markdown report
    md_path = _write_report(all_stations, grand_total, cfg, report_dir, name)
    logger.info(f"Report:   {md_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
