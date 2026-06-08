"""Reporting helpers: Markdown reports, ASCII/CSV tables, static and interactive plots.

Static figures (PNG) use cartopy.  Interactive/zoomable figures (HTML) use
plotly and are written alongside the PNG with the same stem.

Cartopy gridlines are always drawn with labels on the left and bottom axes
only.  Colorbars are placed with a fixed fraction/pad so they stay aligned
with the axes regardless of the map aspect ratio.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Colourmap helpers (cmocean preferred, matplotlib fallback)
# ---------------------------------------------------------------------------

try:
    import cmocean.cm as _cmo
    _CM_DEPTH     = _cmo.deep_r   # shallow → light, deep → dark
    _CM_FRACTION  = _cmo.amp      # 0 → white, 1 → dark orange
    _CM_BALANCE   = _cmo.balance  # diverging around zero
    _CM_AMP       = _cmo.amp      # absolute magnitude
except ImportError:
    _CM_DEPTH    = "Blues_r"
    _CM_FRACTION = "YlOrRd"
    _CM_BALANCE  = "RdBu_r"
    _CM_AMP      = "OrRd"

def _cm_depth():
    """deep_r: shallow → light, deep → dark (standard oceanographic display)."""
    return _CM_DEPTH

def _cm_fraction():
    """amp: 0 → white, 1 → dark orange (good for wet-fraction 0–1)."""
    return _CM_FRACTION

def _cm_correction():
    """balance: diverging around zero for signed depth corrections."""
    return _CM_BALANCE

def _cm_amp():
    """amp: absolute magnitudes (always positive)."""
    return _CM_AMP

def _plotly_colorscale(cmap) -> str | list:
    """Convert a matplotlib/cmocean colormap to a plotly-compatible colorscale."""
    try:
        import matplotlib as mpl
        import matplotlib.colors as mcolors
        n = 64
        c = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
        return [[i / (n - 1), mcolors.to_hex(c(i / (n - 1)))] for i in range(n)]
    except Exception:
        return "Blues_r"


# ---------------------------------------------------------------------------
# Markdown report accumulator
# ---------------------------------------------------------------------------

class MarkdownReport:
    """Accumulate pipeline sections and write a single Markdown file."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._sections: list[dict] = []
        self._created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_section(
        self,
        heading: str,
        text: str = "",
        table: dict[str, Any] | None = None,
        table_rows: list[dict[str, Any]] | None = None,
        images: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self._sections.append(dict(
            heading=heading, text=text, table=table, table_rows=table_rows,
            images=images or [], warnings=warnings or [],
        ))

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            f"# {self.title}", "",
            f"*Generated: {self._created}*", "",
        ]
        for sec in self._sections:
            lines += [f"## {sec['heading']}", ""]
            if sec["text"]:
                lines += [sec["text"], ""]
            for w in sec["warnings"]:
                lines += [f"> ⚠ {w}", ""]
            if sec["table"]:
                lines += _md_kv_table(sec["table"]) + [""]
            if sec["table_rows"]:
                lines += _md_row_table(sec["table_rows"]) + [""]
            for img in sec["images"]:
                lines += [f"![{Path(img).name}]({img})", ""]
        path.write_text("\n".join(lines), encoding="utf-8")


def _md_kv_table(d: dict[str, Any]) -> list[str]:
    out = ["| Parameter | Value |", "|-----------|-------|"]
    for k, v in d.items():
        out.append(f"| {k} | {v} |")
    return out


def _md_row_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    hdrs = list(rows[0].keys())
    out = [
        "| " + " | ".join(str(h) for h in hdrs) + " |",
        "| " + " | ".join("---" for _ in hdrs) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in hdrs) + " |")
    return out


# ---------------------------------------------------------------------------
# ASCII / CSV tables
# ---------------------------------------------------------------------------

def summary_table(data: dict[str, Any], title: str = "") -> str:
    lines: list[str] = []
    if title:
        lines += [title, "-" * max(len(title), 40)]
    w = max((len(k) for k in data), default=20) + 2
    for k, v in data.items():
        lines.append(f"  {k:<{w}}: {v}")
    return "\n".join(lines)


def print_table(data: dict[str, Any], title: str = "") -> None:
    print(summary_table(data, title))
    print()


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_fixes_yaml(records: list[dict], path: str | Path) -> None:
    """Write suggested fixes grouped by cause (BLOCKED → SILL_DEFICIT → AREA_DEFICIT)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-category lists in priority order
    groups: dict[str, list[dict]] = {"BLOCKED": [], "SILL_DEFICIT": [], "AREA_DEFICIT": []}
    for r in records:
        cat = r["category"]
        if cat == "BLOCKED":
            groups["BLOCKED"].append({"lon": r["lon"], "lat": r["lat"],
                                      "action": "open_cell", "depth": r["sill_depth_fine"],
                                      "_note": "BLOCKED — no fine wet path"})
        elif cat == "SILL_DEFICIT":
            groups["SILL_DEFICIT"].append({"lon": r["lon"], "lat": r["lat"],
                                           "action": "set_depth", "value": r["sill_depth_fine"],
                                           "_note": f"SILL_DEFICIT — sill_ratio={r['sill_ratio']}"})
        elif cat == "AREA_DEFICIT":
            groups["AREA_DEFICIT"].append({"lon": r["lon"], "lat": r["lat"],
                                           "action": "set_depth",
                                           "value": round(r["sill_depth_fine"] * 0.9, 1),
                                           "_note": f"AREA_DEFICIT — area_ratio={r['area_ratio']}"})

    def _write_fix(fh, fix: dict) -> None:
        fh.write(f"  - lon: {fix['lon']}\n")
        fh.write(f"    lat: {fix['lat']}\n")
        fh.write(f"    action: {fix['action']}\n")
        if "value" in fix:
            fh.write(f"    value: {fix['value']}\n")
        if "depth" in fix:
            fh.write(f"    depth: {fix['depth']}\n")
        fh.write(f"    # {fix['_note']}\n")

    category_desc = {
        "BLOCKED":      "No fine wet path — open_cell to reconnect",
        "SILL_DEFICIT": "Coarse sill too shallow — set_depth to fine-grid sill",
        "AREA_DEFICIT": "Cross-section under-represented — deepen to improve transport",
    }

    with open(path, "w") as fh:
        fh.write("# Suggested fixes — re-run with --accept-fixes to apply all,\n")
        fh.write("# or paste selected entries into the 'fixes:' section of your config.\n")
        fh.write("# '_note' lines are ignored by the loader; no need to remove them.\n\n")
        fh.write("fixes:\n")
        for cat, fixes in groups.items():
            if not fixes:
                continue
            fh.write(f"\n  # --- {cat}: {category_desc[cat]} ({len(fixes)} interface(s)) ---\n")
            for fix in fixes:
                _write_fix(fh, fix)


# ---------------------------------------------------------------------------
# Cartopy helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _apply_gridlines(ax) -> None:
    """Add gridlines with labels on left and bottom only."""
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False


def _inset_gridlines(ax, extent: tuple) -> None:
    """Add labelled gridlines to a small inset Cartopy axes.

    Tick spacing is chosen automatically from the extent span so that the
    inset is neither over-ticked nor under-ticked regardless of domain size.
    """
    import matplotlib.ticker as mticker

    lon_min, lon_max, lat_min, lat_max = extent

    def _step(span: float) -> float:
        if span > 20: return 10.0
        if span > 10: return 5.0
        if span >  5: return 2.0
        if span >  2: return 1.0
        return 0.5

    dlon = _step(lon_max - lon_min)
    dlat = _step(lat_max - lat_min)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.6)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.MultipleLocator(dlon)
    gl.ylocator     = mticker.MultipleLocator(dlat)
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}


def _add_colorbar(fig, ax, pcm, label: str, cmap: str = "") -> None:
    """Add a colorbar that stays aligned with the cartopy axes."""
    fig.colorbar(pcm, ax=ax, label=label, fraction=0.03, pad=0.04, aspect=30)


# ---------------------------------------------------------------------------
# Static (cartopy) plots
# ---------------------------------------------------------------------------

def plot_rotated_pole(
    corner_lon: npt.NDArray,
    corner_lat: npt.NDArray,
    pole_lon: float,
    pole_lat: float,
    title: str,
    path: str | Path,
) -> None:
    """Two-panel globe plot illustrating the rotated-pole grid geometry.

    Left  — Mollweide (whole-world) projection: always shows both the domain
            footprint and the rotated-pole marker regardless of their angular
            separation.
    Right — AzimuthalEquidistant centred on the rotated North Pole: the pole
            is at the map centre; the domain (exactly 90° away by construction)
            always appears at mid-map radius.  The rotated equator (great
            circle 90° from the pole) is drawn as a dashed reference.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        return

    import matplotlib.pyplot as plt

    path = _ensure_dir(path)

    # Domain perimeter from corner-coordinate grid
    perim_lon = np.concatenate([
        corner_lon[0, :],          # bottom: left → right
        corner_lon[1:, -1],        # right:  bottom → top
        corner_lon[-1, -2::-1],    # top:    right → left
        corner_lon[-2::-1, 0],     # left:   top → bottom
    ])
    perim_lat = np.concatenate([
        corner_lat[0, :],
        corner_lat[1:, -1],
        corner_lat[-1, -2::-1],
        corner_lat[-2::-1, 0],
    ])

    geo = ccrs.PlateCarree()
    _DOMAIN_COLOR = "#e07b00"   # orange — contrasts well with both ocean and land

    # ------------------------------------------------------------------ #
    # Rotated-equator (great circle 90° from the rotated pole) in       #
    # geographic coordinates, for the right panel.                       #
    # Inline inverse rotated-pole transform: (rlon, rlat=0) → (geo_lon, geo_lat)
    # ------------------------------------------------------------------ #
    rlon_eq = np.linspace(-180.0, 180.0, 361)
    pp = np.radians(pole_lat)
    rr = np.radians(rlon_eq)
    sin_lat = np.sin(pp) * np.sin(np.zeros(361)) + np.cos(pp) * np.cos(np.zeros(361)) * np.cos(rr)
    sin_lat = np.clip(sin_lat, -1.0, 1.0)
    req_lat = np.degrees(np.arcsin(sin_lat))
    cos_req = np.cos(np.radians(req_lat))
    safe = cos_req > 1e-10
    sn = np.where(safe, np.cos(np.zeros(361)) * np.sin(rr) / cos_req, 0.0)
    cs = np.where(
        safe,
        (np.cos(pp) * np.sin(np.zeros(361)) - np.sin(pp) * np.cos(np.zeros(361)) * np.cos(rr)) / cos_req,
        np.sign(np.cos(pp)),
    )
    req_lon = (pole_lon + np.degrees(np.arctan2(sn, cs)) + 180.0) % 360.0 - 180.0

    fig = plt.figure(figsize=(13, 5))

    # ------------------------------------------------------------------ #
    # Left panel — Mollweide (global, shows everything)                  #
    # ------------------------------------------------------------------ #
    ax_l = fig.add_subplot(1, 2, 1, projection=ccrs.Mollweide())
    ax_l.set_global()  # type: ignore[union-attr]
    ax_l.add_feature(cfeature.OCEAN, facecolor="#cde8f6", zorder=0)  # type: ignore[union-attr]
    ax_l.add_feature(cfeature.LAND,  facecolor="#e8dcc8", zorder=1)  # type: ignore[union-attr]
    ax_l.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)  # type: ignore[union-attr]
    ax_l.gridlines(linewidth=0.25, color="grey", alpha=0.5, zorder=2)  # type: ignore[union-attr]

    ax_l.fill(  # type: ignore[union-attr]
        np.append(perim_lon, perim_lon[0]),
        np.append(perim_lat, perim_lat[0]),
        color=_DOMAIN_COLOR, alpha=0.45, transform=geo, zorder=3,
    )
    ax_l.plot(  # type: ignore[union-attr]
        np.append(perim_lon, perim_lon[0]),
        np.append(perim_lat, perim_lat[0]),
        color=_DOMAIN_COLOR, linewidth=1.5, transform=geo, zorder=4,
    )
    ax_l.plot(  # type: ignore[union-attr]
        pole_lon, pole_lat,
        marker="*", markersize=14, color="firebrick",
        transform=geo, zorder=5,
        label=f"Rotated N-pole ({pole_lon:.1f}°E, {pole_lat:.1f}°N)",
    )
    ax_l.legend(loc="lower left", fontsize=7, framealpha=0.85)  # type: ignore[union-attr]
    ax_l.set_title(
        f"Geographic view  |  rotated pole at ({pole_lon:.2f}°E, {pole_lat:.2f}°N)",
        fontsize=8, pad=4,
    )  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # Right panel — AzimuthalEquidistant centred on rotated pole        #
    # ------------------------------------------------------------------ #
    proj_r = ccrs.AzimuthalEquidistant(
        central_longitude=pole_lon, central_latitude=pole_lat,
    )
    ax_r = fig.add_subplot(1, 2, 2, projection=proj_r)
    ax_r.set_global()  # type: ignore[union-attr]
    ax_r.add_feature(cfeature.OCEAN, facecolor="#cde8f6", zorder=0)  # type: ignore[union-attr]
    ax_r.add_feature(cfeature.LAND,  facecolor="#e8dcc8", zorder=1)  # type: ignore[union-attr]
    ax_r.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)  # type: ignore[union-attr]
    ax_r.gridlines(linewidth=0.25, color="grey", alpha=0.5, zorder=2)  # type: ignore[union-attr]

    # Rotated equator — dashed grey great circle
    ax_r.plot(  # type: ignore[union-attr]
        req_lon, req_lat,
        color="grey", linewidth=1.0, linestyle="--",
        transform=geo, zorder=3, label="Rotated equator",
    )

    ax_r.fill(  # type: ignore[union-attr]
        np.append(perim_lon, perim_lon[0]),
        np.append(perim_lat, perim_lat[0]),
        color=_DOMAIN_COLOR, alpha=0.45, transform=geo, zorder=4,
    )
    ax_r.plot(  # type: ignore[union-attr]
        np.append(perim_lon, perim_lon[0]),
        np.append(perim_lat, perim_lat[0]),
        color=_DOMAIN_COLOR, linewidth=1.5, transform=geo, zorder=5,
    )
    ax_r.plot(  # type: ignore[union-attr]
        pole_lon, pole_lat,
        marker="*", markersize=14, color="firebrick",
        transform=geo, zorder=6,
    )
    ax_r.legend(loc="lower left", fontsize=7, framealpha=0.85)  # type: ignore[union-attr]
    ax_r.set_title(
        "View from rotated North Pole  |  domain at 90° radius",
        fontsize=8, pad=4,
    )  # type: ignore[union-attr]

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_depth_diff(
    lon: npt.NDArray,
    lat: npt.NDArray,
    diff: npt.NDArray,
    title: str,
    path: str | Path,
    subtitle: str = "",
    vmax: Optional[float] = None,
) -> None:
    """Signed depth difference map (source1 − source2) at common ocean cells.

    *diff* should be NaN wherever either source is land.  A diverging colormap
    is used, symmetric around zero.  Mean, std and RMSE are printed as a
    subtitle so the reader can judge the magnitude of the discrepancy.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    path = _ensure_dir(path)

    valid = ~np.isnan(diff)
    if not valid.any():
        return

    dvals = diff[valid]
    mean_d = float(np.mean(dvals))
    std_d  = float(np.std(dvals))
    rmse   = float(np.sqrt(np.mean(dvals ** 2)))
    n_cells = int(valid.sum())

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(diff), 99))
    vmax = max(vmax, 0.1)

    cmap = _cm_correction()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    stats_str = (
        f"mean={mean_d:+.1f} m   std={std_d:.1f} m   RMSE={rmse:.1f} m   "
        f"n={n_cells:,} cells"
    )
    full_title = "\n".join(t for t in [title, subtitle, stats_str] if t)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, diff,  # type: ignore[union-attr]
                           cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        plt.colorbar(im, ax=ax, label="Depth difference (m)", fraction=0.046, pad=0.04)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)  # type: ignore[union-attr]
        _apply_gridlines(ax)  # type: ignore[arg-type]
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(lon, lat, diff, cmap=cmap, norm=norm)  # type: ignore[union-attr]
        plt.colorbar(im, ax=ax, label="Depth difference (m)")

    ax.set_title(full_title, fontsize=10)  # type: ignore[union-attr]
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    _save_diff_html(lon, lat, diff, full_title, Path(path), vmax)


def plot_source_comparison(
    lon: npt.NDArray,
    lat: npt.NDArray,
    mask1: npt.NDArray,
    mask2: npt.NDArray,
    name1: str,
    name2: str,
    path: str | Path,
) -> None:
    """Four-class comparison map between two regridded source masks.

    Each cell is classified as one of:

    * **Common land**  — both sources say land
    * **Common water** — both sources say ocean
    * **name1 only**   — source 1 is ocean, source 2 is land
    * **name2 only**   — source 1 is land, source 2 is ocean

    The third and fourth classes reveal coastline disagreements between the
    two sources at the model resolution.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    path = _ensure_dir(path)

    # 0=common land, 1=common water, 2=name1-only, 3=name2-only
    comp = np.zeros(mask1.shape, dtype=np.float32)
    comp[(mask1 == 1) & (mask2 == 1)] = 1.0
    comp[(mask1 == 1) & (mask2 == 0)] = 2.0
    comp[(mask1 == 0) & (mask2 == 1)] = 3.0

    colours = ["#c8c8c8", "#4a90d9", "#e07b39", "#5cb85c"]
    labels  = [
        "Common land",
        "Common water",
        f"{name1} only",
        f"{name2} only",
    ]
    counts = [
        int((comp == v).sum()) for v in range(4)
    ]
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.pcolormesh(lon, lat, comp,  # type: ignore[union-attr]
                      cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)  # type: ignore[union-attr]
        _apply_gridlines(ax)  # type: ignore[arg-type]
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pcolormesh(lon, lat, comp, cmap=cmap, norm=norm)  # type: ignore[union-attr]

    patches = [
        mpatches.Patch(color=colours[i], label=f"{labels[i]}  ({counts[i]:,} cells)")
        for i in range(4)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)  # type: ignore[union-attr]
    ax.set_title(f"Source mask comparison: {name1} vs {name2}", fontsize=11)  # type: ignore[union-attr]
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_depth(
    lon: npt.NDArray,
    lat: npt.NDArray,
    depth: npt.NDArray,
    mask: npt.NDArray,
    title: str,
    path: str | Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap=None,
    interactive: bool = False,
    subtitle: str = "",
    colorbar_label: str = "Depth (m)",
    log_scale: bool = False,
) -> None:
    """Plot a scalar field.  Saves PNG; optionally also saves a plotly HTML.

    Parameters
    ----------
    subtitle : str
        Optional second line rendered below the main title in a smaller,
        italic font — intended for grid metadata (nx×ny, wet cells, resolution).
    colorbar_label : str
        Label for the colorbar (default ``"Depth (m)"``).
    log_scale : bool
        Use a logarithmic colour scale.  Useful when depth spans several orders
        of magnitude (shallow shelf to deep ocean).  Values ≤ 0 are masked.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if cmap is None:
        cmap = _cm_depth()
    path = _ensure_dir(path)
    masked = np.where(mask, depth, np.nan)

    if log_scale:
        # Mask non-positive values so LogNorm doesn't fail
        masked = np.where(masked > 0, masked, np.nan)
        _vmin = vmin if vmin is not None else float(np.nanmin(masked))
        _vmax = vmax if vmax is not None else float(np.nanmax(masked))
        _vmin = max(_vmin, 1e-3)
        norm: Optional[mcolors.Normalize] = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
        kw: dict = dict(cmap=cmap, norm=norm)
    else:
        vmin = vmin if vmin is not None else float(np.nanmin(masked))
        vmax = vmax if vmax is not None else float(np.nanmax(masked))
        norm = None
        kw = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon, lat, masked, transform=ccrs.PlateCarree(), **kw)  # type: ignore[union-attr]
        ax.add_feature(cfeature.LAND, facecolor="tan", zorder=2)  # type: ignore[union-attr]
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)  # type: ignore[union-attr]
        _apply_gridlines(ax)  # type: ignore[arg-type]
        _add_colorbar(fig, ax, pcm, colorbar_label)
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(lon, lat, masked, **kw)  # type: ignore[union-attr]
        ax.set_xlabel("Longitude")  # type: ignore[union-attr]
        ax.set_ylabel("Latitude")  # type: ignore[union-attr]
        fig.colorbar(pcm, ax=ax, label=colorbar_label, fraction=0.03, pad=0.04)

    if subtitle:
        ax.set_title(  # type: ignore[union-attr]
            f"{title}\n{subtitle}", fontsize=11, pad=4,
            linespacing=1.5,
        )
    else:
        ax.set_title(title, fontsize=11, pad=4)  # type: ignore[union-attr]
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if interactive:
        _save_depth_html(lon, lat, masked, title, path, log_scale=log_scale)


def plot_comparison(
    lon: npt.NDArray,
    lat: npt.NDArray,
    fields: Sequence[npt.NDArray],
    masks: Sequence[npt.NDArray],
    titles: Sequence[str],
    path: str | Path,
    cmap=None,
    label: str = "Depth (m)",
) -> None:
    """Side-by-side comparison — one shared colorbar on the right."""
    import matplotlib.pyplot as plt

    path = _ensure_dir(path)
    n = len(fields)
    wet_vals = [f[m.astype(bool)] for f, m in zip(fields, masks) if m.any()]
    all_vals = np.concatenate(wet_vals) if wet_vals else np.array([0.0, 1.0])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if cmap is None:
        cmap = _cm_depth()
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig, axes = plt.subplots(
            1, n, figsize=(6 * n, 5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        axes_list = [axes] if n == 1 else list(axes)
        pcm = None
        for ax, field, mask, title in zip(axes_list, fields, masks, titles):
            masked = np.where(mask, field, np.nan)
            pcm = ax.pcolormesh(lon, lat, masked, transform=ccrs.PlateCarree(), **kw)  # type: ignore[union-attr]
            ax.add_feature(cfeature.LAND, facecolor="tan", zorder=2)  # type: ignore[union-attr]
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)  # type: ignore[union-attr]
            _apply_gridlines(ax)  # type: ignore[arg-type]
            ax.set_title(title)
    except ImportError:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        axes_list = [axes] if n == 1 else list(axes)
        pcm = None
        for ax, field, mask, title in zip(axes_list, fields, masks, titles):
            masked = np.where(mask, field, np.nan)
            pcm = ax.pcolormesh(lon, lat, masked, **kw)  # type: ignore[union-attr]
            ax.set_title(title)

    if pcm is not None:
        fig.colorbar(pcm, ax=axes_list, label=label, fraction=0.02, pad=0.04, aspect=40)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_section_profile(
    distance_km: npt.NDArray,
    depth_fine: npt.NDArray,
    depth_coarse: float,
    title: str,
    path: str | Path,
    inset_lon: Optional[float] = None,
    inset_lat: Optional[float] = None,
    inset_bounds: Optional[tuple[float, float, float, float]] = None,
    fine_sub: Optional[npt.NDArray] = None,
    fine_lons: Optional[npt.NDArray] = None,
    fine_lats: Optional[npt.NDArray] = None,
    cs_col: Optional[int] = None,
    cs_row: Optional[int] = None,
    coarse_corner_lons: Optional[npt.NDArray] = None,
    coarse_corner_lats: Optional[npt.NDArray] = None,
) -> None:
    """Cross-section depth profile with optional map inset and plan view.

    When *fine_sub* is provided (the 2-D fine-resolution sub-grid around the
    flagged interface), the figure uses a three-panel layout:
    - top-left: section depth profile with two dashed neighbour curves
    - top-right: Cartopy location inset map
    - bottom: plan view of fine bathymetry with coarse grid overlay

    Parameters
    ----------
    fine_sub : [ny_f, nx_f] array
        Fine-resolution depth sub-grid.  Positive = ocean.
    fine_lons : [nx_f] array
        Longitude coordinates of *fine_sub* columns.
    fine_lats : [ny_f] array
        Latitude coordinates of *fine_sub* rows.
    cs_col : int
        Column index of the cross-section in *fine_sub* (U-interface).
    cs_row : int
        Row index of the cross-section in *fine_sub* (V-interface).
    coarse_corner_lons : [nr+1, nc+1] array
        Coarse cell corner longitudes (from dst_grid.corner_lon subset).
    coarse_corner_lats : [nr+1, nc+1] array
        Coarse cell corner latitudes (from dst_grid.corner_lat subset).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    path = _ensure_dir(path)
    has_plan = (
        fine_sub is not None
        and fine_lons is not None
        and fine_lats is not None
        and (cs_col is not None or cs_row is not None)
    )
    has_inset = inset_lon is not None and inset_lat is not None

    if has_plan:
        # Three-panel layout: [profile | inset] on top, [plan view] on bottom
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            height_ratios=[1, 1.2],
            width_ratios=[1.6, 1],
            hspace=0.35, wspace=0.3,
        )
        ax      = fig.add_subplot(gs[0, 0])   # top-left: section profile
        ax_plan = fig.add_subplot(gs[1, :])   # bottom: plan view full width
        ax_ins_gs = gs[0, 1]                  # top-right: reserved for inset
    else:
        fig, ax = plt.subplots(figsize=(8, 4))

    # ---- Section profile ----
    ax.fill_between(distance_km, 0, depth_fine, where=depth_fine > 0,
                    color="steelblue", alpha=0.35, label="Fine (central section)")
    ax.plot(distance_km, depth_fine, color="steelblue", linewidth=1.4)

    # Neighbouring sections: ±step in the perpendicular direction
    if has_plan:
        assert fine_sub is not None  # for type-checker
        if cs_col is not None:
            step = max(1, fine_sub.shape[1] // 4)
            for sign, lbl in [(-1, "−"), (+1, "+")]:
                nb = cs_col + sign * step
                if 0 <= nb < fine_sub.shape[1]:
                    nb_depth = fine_sub[:, nb].astype(float)
                    nb_depth = np.where(nb_depth > 0, nb_depth, np.nan)
                    ax.plot(distance_km, nb_depth,
                            color="steelblue", linewidth=0.8,
                            linestyle="--", alpha=0.6,
                            label=f"Fine ({lbl}½ cell E/W)")
        else:
            assert cs_row is not None
            step = max(1, fine_sub.shape[0] // 4)
            for sign, lbl in [(-1, "−"), (+1, "+")]:
                nb = cs_row + sign * step
                if 0 <= nb < fine_sub.shape[0]:
                    nb_depth = fine_sub[nb, :].astype(float)
                    nb_depth = np.where(nb_depth > 0, nb_depth, np.nan)
                    ax.plot(distance_km, nb_depth,
                            color="steelblue", linewidth=0.8,
                            linestyle="--", alpha=0.6,
                            label=f"Fine ({lbl}½ cell N/S)")

    ax.axhline(depth_coarse, color="firebrick", linewidth=1.5, linestyle="--",
               label=f"Coarse cell mean: {depth_coarse:.1f} m")
    ax.set_xlabel("Distance along section (km)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9 if has_plan else 10)
    ax.legend(fontsize=8)

    # ---- Plan view panel ----
    if has_plan:
        assert fine_sub is not None and fine_lons is not None and fine_lats is not None

        R = 6371.0
        clat = float(fine_lats.mean())
        lon_km = (fine_lons - fine_lons.mean()) * np.pi / 180.0 * R * np.cos(np.radians(clat))
        lat_km = (fine_lats - fine_lats.mean()) * np.pi / 180.0 * R

        depth_masked = np.where(fine_sub > 0, fine_sub.astype(float), np.nan)
        lon_km_2d, lat_km_2d = np.meshgrid(lon_km, lat_km)
        vmax = float(np.nanmax(depth_masked)) if np.isfinite(depth_masked).any() else 1.0
        pcm = ax_plan.pcolormesh(
            lon_km_2d, lat_km_2d, depth_masked,
            cmap=_cm_depth(), vmin=0.0, vmax=vmax, shading="auto",
        )
        fig.colorbar(pcm, ax=ax_plan, label="Fine depth (m)", fraction=0.025, pad=0.03)

        # Coarse grid overlay — exact cell edges from corner coordinates
        if coarse_corner_lons is not None and coarse_corner_lats is not None:
            # Convert coarse corners to km (same reference as fine grid)
            ref_lon = float(fine_lons.mean())
            ref_lat = clat  # already computed above
            cc_lon_km = (
                (coarse_corner_lons - ref_lon)
                * np.pi / 180.0 * R * np.cos(np.radians(ref_lat))
            )
            cc_lat_km = (coarse_corner_lats - ref_lat) * np.pi / 180.0 * R
            _gkw = dict(color="white", linewidth=0.9, alpha=0.9, zorder=3)
            # Draw each row of corners as a polyline (constant-lat cell edges)
            for k in range(cc_lon_km.shape[0]):
                ax_plan.plot(cc_lon_km[k, :], cc_lat_km[k, :], **_gkw)
            # Draw each column of corners as a polyline (constant-lon cell edges)
            for k in range(cc_lon_km.shape[1]):
                ax_plan.plot(cc_lon_km[:, k], cc_lat_km[:, k], **_gkw)

        # Cross-section line and neighbours (drawn after grid so they are on top)
        if cs_col is not None:
            step = max(1, fine_sub.shape[1] // 4)
            ax_plan.axvline(lon_km[cs_col], color="firebrick", linewidth=1.5,
                            linestyle="-", label="Section", zorder=4)
            for sign in (-1, +1):
                nb = cs_col + sign * step
                if 0 <= nb < len(lon_km):
                    ax_plan.axvline(lon_km[nb], color="firebrick", linewidth=0.8,
                                    linestyle="--", alpha=0.7, zorder=4)
        else:
            assert cs_row is not None
            step = max(1, fine_sub.shape[0] // 4)
            ax_plan.axhline(lat_km[cs_row], color="firebrick", linewidth=1.5,
                            linestyle="-", label="Section", zorder=4)
            for sign in (-1, +1):
                nb = cs_row + sign * step
                if 0 <= nb < len(lat_km):
                    ax_plan.axhline(lat_km[nb], color="firebrick", linewidth=0.8,
                                    linestyle="--", alpha=0.7, zorder=4)

        ax_plan.set_xlabel("E–W distance from interface (km)")
        ax_plan.set_ylabel("N–S distance from interface (km)")
        ax_plan.set_title("Fine-resolution bathymetry (plan view)", fontsize=9)
        ax_plan.legend(fontsize=8, loc="upper right")

        # ---- Cartopy inset in top-right GridSpec cell ----
        if has_inset:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature

                ax_ins = fig.add_subplot(ax_ins_gs, projection=ccrs.PlateCarree())
                extent = (
                    list(inset_bounds) if inset_bounds else
                    [inset_lon - 8, inset_lon + 8, inset_lat - 5, inset_lat + 5]
                )
                ax_ins.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore[union-attr]
                ax_ins.add_feature(cfeature.LAND, facecolor="tan", zorder=1)  # type: ignore[union-attr]
                ax_ins.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)  # type: ignore[union-attr]
                ax_ins.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)  # type: ignore[union-attr]
                ax_ins.plot(  # type: ignore[union-attr]
                    inset_lon, inset_lat, "r+", markersize=10, markeredgewidth=2,
                    transform=ccrs.PlateCarree(), zorder=3,
                )
                _inset_gridlines(ax_ins, extent)
                ax_ins.set_title("section location", fontsize=8, pad=2)  # type: ignore[union-attr]
            except ImportError:
                pass

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # ---- Original single-panel layout + optional Cartopy inset ----
    if not has_inset:
        fig.tight_layout()

    if has_inset:
        fig.subplots_adjust(right=0.60)
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax_ins = fig.add_axes([0.63, 0.50, 0.33, 0.42],
                                  projection=ccrs.PlateCarree())
            extent = (
                list(inset_bounds) if inset_bounds else
                [inset_lon - 8, inset_lon + 8, inset_lat - 5, inset_lat + 5]
            )
            ax_ins.set_extent(extent, crs=ccrs.PlateCarree())
            ax_ins.add_feature(cfeature.LAND, facecolor="tan", zorder=1)
            ax_ins.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
            ax_ins.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
            ax_ins.plot(inset_lon, inset_lat, "r+", markersize=10, markeredgewidth=2,
                        transform=ccrs.PlateCarree(), zorder=3)
            _inset_gridlines(ax_ins, extent)
            ax_ins.set_title("section location", fontsize=7, pad=2)
        except ImportError:
            pass

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_straits(
    lon: npt.NDArray,
    lat: npt.NDArray,
    depth: npt.NDArray,
    mask: npt.NDArray,
    strait_records: list[dict],
    path: str | Path,
    domain_bounds: Optional[tuple[float, float, float, float]] = None,
) -> None:
    """Depth map with all flagged interfaces + regional context inset + HTML.

    Parameters
    ----------
    domain_bounds : (lon_min, lon_max, lat_min, lat_max)
        When supplied, a small inset is drawn in the lower-left corner showing
        the model domain in its regional geographic context (coastlines only,
        domain outlined with a red rectangle).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    path = _ensure_dir(path)
    colours = {"AREA_DEFICIT": "orange", "SILL_DEFICIT": "red", "BLOCKED": "black"}

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        masked = np.where(mask, depth, np.nan)
        pcm = ax.pcolormesh(lon, lat, masked, cmap=_cm_depth(),
                            transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="tan", zorder=2)  # type: ignore[union-attr]
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)  # type: ignore[union-attr]
        _apply_gridlines(ax)  # type: ignore[arg-type]
        transform = ccrs.PlateCarree()
    except ImportError:
        fig, ax = plt.subplots(figsize=(12, 7))
        masked = np.where(mask, depth, np.nan)
        pcm = ax.pcolormesh(lon, lat, masked, cmap=_cm_depth())  # type: ignore[union-attr]
        transform = None

    for rec in strait_records:
        c = colours.get(rec["category"], "purple")
        kw: dict = dict(color=c, marker="x", markersize=8, linewidth=2, zorder=5)
        if transform is not None:
            ax.plot(rec["lon"], rec["lat"], transform=transform, **kw)  # type: ignore[union-attr]
        else:
            ax.plot(rec["lon"], rec["lat"], **kw)  # type: ignore[union-attr]

    patches = [mpatches.Patch(color=c, label=k) for k, c in colours.items()]
    ax.legend(handles=patches, loc="upper right")  # type: ignore[union-attr]
    _add_colorbar(fig, ax, pcm, "Depth (m)")
    ax.set_title(
        f"Strait / connectivity concerns — {len(strait_records)} flagged interface(s)"
    )

    # Regional context inset: shows the domain as a red box on a wider map
    if domain_bounds is not None:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import matplotlib.patches as mpatch

            lon_min, lon_max, lat_min, lat_max = domain_bounds
            pad_lon = max((lon_max - lon_min) * 1.5, 5.0)
            pad_lat = max((lat_max - lat_min) * 1.5, 4.0)

            # Place inset in lower-left; tight_layout is called first so the
            # main axes position is settled before we pin the inset.
            fig.tight_layout()
            ax_ins = fig.add_axes([0.01, 0.01, 0.22, 0.28],
                                  projection=ccrs.PlateCarree())
            ax_ins.set_extent(
                [lon_min - pad_lon, lon_max + pad_lon,
                 lat_min - pad_lat, lat_max + pad_lat],
                crs=ccrs.PlateCarree(),
            )
            ins_extent = (
                lon_min - pad_lon, lon_max + pad_lon,
                lat_min - pad_lat, lat_max + pad_lat,
            )
            ax_ins.add_feature(cfeature.LAND, facecolor="tan", zorder=1)
            ax_ins.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
            ax_ins.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
            # Draw domain rectangle
            rect = mpatch.Rectangle(
                (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                linewidth=1.5, edgecolor="red", facecolor="none", zorder=3,
                transform=ccrs.PlateCarree(),
            )
            ax_ins.add_patch(rect)
            _inset_gridlines(ax_ins, ins_extent)
            ax_ins.set_title("domain", fontsize=7, pad=2)
        except ImportError:
            fig.tight_layout()
    else:
        fig.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Interactive version
    _save_straits_html(lon, lat, masked, strait_records, colours, path)


def plot_basins(
    lon: npt.NDArray,
    lat: npt.NDArray,
    labels: npt.NDArray,
    title: str,
    path: str | Path,
    nkeep: int = 1,
) -> None:
    """Plot connected ocean basins.

    Kept basins (the *nkeep* largest) are shown in blue tones.
    Removed (isolated) basins are shown in red/orange tones.
    Land / unmasked cells are shown as light grey.

    This answers the question "what does the basin plot show?": it reveals
    any isolated ocean pockets disconnected from the main basin that were
    masked out.  If only one colour appears, there were no isolated regions.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    path = _ensure_dir(path)

    # Rank basins by size; the largest nkeep are "kept"
    unique_ids, counts = np.unique(labels[labels > 0], return_counts=True)
    order = np.argsort(counts)[::-1]
    kept_ids = set(unique_ids[order[:nkeep]].tolist())

    blue_cm = plt.cm.Blues   # type: ignore[attr-defined]
    red_cm  = plt.cm.Reds    # type: ignore[attr-defined]
    kept_list    = [i for i in order if unique_ids[i] in kept_ids]
    removed_list = [i for i in order if unique_ids[i] not in kept_ids]

    # Build a scalar field: 0=land, 1..nkeep=kept, nkeep+1..=removed.
    # pcolormesh places each cell correctly at its geographic position;
    # imshow(extent=...) shifts by half a cell and fails for 2-D lon/lat.
    scalar = np.zeros(labels.shape, dtype=float)
    colours = [(0.88, 0.88, 0.88)]  # index 0 → land (grey)
    for rank, idx in enumerate(kept_list):
        scalar[labels == unique_ids[idx]] = rank + 1
        colours.append(blue_cm(0.5 + 0.4 * rank / max(len(kept_list), 1)))
    for rank, idx in enumerate(removed_list):
        scalar[labels == unique_ids[idx]] = len(kept_list) + rank + 1
        colours.append(red_cm(0.4 + 0.5 * rank / max(len(removed_list), 1)))

    n = len(colours)
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n), n)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.pcolormesh(lon, lat, scalar,  # type: ignore[union-attr]
                      cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)  # type: ignore[union-attr]
        _apply_gridlines(ax)  # type: ignore[arg-type]
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pcolormesh(lon, lat, scalar, cmap=cmap, norm=norm)  # type: ignore[union-attr]

    legend_patches = [mpatches.Patch(color=blue_cm(0.6), label=f"Kept ({len(kept_list)} basin(s))")]
    if removed_list:
        legend_patches.append(
            mpatches.Patch(color=red_cm(0.5), label=f"Removed ({len(removed_list)} isolated basin(s))")
        )
    legend_patches.append(mpatches.Patch(color=(0.88, 0.88, 0.88), label="Land"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)  # type: ignore[union-attr]

    n_removed = len(removed_list)
    subtitle = (
        f"Isolated basins removed: {n_removed}" if n_removed
        else "No isolated basins — full domain is connected"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)  # type: ignore[union-attr]
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rx0_diagnostics(
    rx0_before: npt.NDArray,
    rx0_after: npt.NDArray,
    corrections: npt.NDArray,
    lon: npt.NDArray,
    lat: npt.NDArray,
    mask: npt.NDArray,
    path_prefix: str | Path,
) -> None:
    """Histogram of rx0 before/after + static and interactive correction maps."""
    import matplotlib.pyplot as plt

    pp = Path(path_prefix)
    pp.parent.mkdir(parents=True, exist_ok=True)

    upper = max(float(rx0_before.max()), float(rx0_after.max()), 0.01) * 1.05
    bins = np.linspace(0, upper, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rx0_before[rx0_before > 0], bins=bins, alpha=0.6, label="Before", color="firebrick")
    ax.hist(rx0_after[rx0_after > 0], bins=bins, alpha=0.6, label="After", color="steelblue")
    ax.set_xlabel("rx0")
    ax.set_ylabel("Count")
    ax.set_title("rx0 distribution before / after smoothing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(pp) + "_rx0_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    corr_map = np.full(mask.shape, np.nan)
    corr_map[mask.astype(bool)] = np.abs(corrections)[mask.astype(bool)]
    plot_depth(
        lon, lat, corr_map, mask,
        title="Absolute depth correction from rx0 smoothing (m)",
        path=str(pp) + "_corrections.png",
        cmap=_cm_amp(),
        interactive=True,   # also write HTML
    )


# ---------------------------------------------------------------------------
# Interactive (plotly) helpers
# ---------------------------------------------------------------------------

def _save_diff_html(
    lon: npt.NDArray,
    lat: npt.NDArray,
    diff: npt.NDArray,
    title: str,
    png_path: Path,
    vmax: Optional[float],
) -> None:
    """Write a zoomable plotly Heatmap for a signed depth-difference field."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    lon_axis = lon[0, :] if lon.ndim == 2 else lon
    lat_axis = lat[:, 0] if lat.ndim == 2 else lat

    abs_max = float(np.nanmax(np.abs(diff[np.isfinite(diff)]))) if np.isfinite(diff).any() else 1.0
    z_range = vmax if vmax is not None else abs_max

    heatmap = go.Heatmap(
        z=diff,
        x=lon_axis,
        y=lat_axis,
        colorscale=_plotly_colorscale(_cm_correction()),
        zmin=-z_range,
        zmid=0.0,
        zmax=z_range,
        colorbar=dict(title="Δ depth (m)", thickness=15),
        hoverongaps=False,
        hovertemplate=(
            "lon: %{x:.3f}<br>lat: %{y:.3f}<br>diff: %{z:+.1f} m<extra></extra>"
        ),
    )
    fig = go.Figure(heatmap)
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis_scaleanchor="x",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.write_html(str(png_path.with_suffix(".html")))


def _save_depth_html(
    lon: npt.NDArray,
    lat: npt.NDArray,
    depth_masked: npt.NDArray,
    title: str,
    png_path: Path,
    log_scale: bool = False,
) -> None:
    """Write a zoomable plotly Heatmap alongside the PNG."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    # For 2D lon/lat grids extract 1D axes (or use the 2D arrays directly)
    if lon.ndim == 2:
        lon_axis = lon[0, :]
        lat_axis = lat[:, 0]
    else:
        lon_axis = lon
        lat_axis = lat

    if log_scale:
        # Plotly Heatmap doesn't support LogNorm natively; store log10 values
        # and customise the colorbar ticks to show original depths.
        pos = np.where(depth_masked > 0, depth_masked, np.nan)
        z_plot = np.log10(pos)
        valid = z_plot[np.isfinite(z_plot)]
        if valid.size:
            lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
            import math
            tick_vals = [10 ** e for e in range(math.floor(lo), math.ceil(hi) + 1)]
            tick_text = [f"{v:.0f} m" for v in tick_vals]
            log_tick_vals = [math.log10(v) for v in tick_vals]
        else:
            log_tick_vals, tick_text = [], []
        colorbar = dict(
            title="Depth (m)", thickness=15,
            tickvals=log_tick_vals, ticktext=tick_text,
        )
        hover = "lon: %{x:.3f}<br>lat: %{y:.3f}<br>depth: %{customdata:.1f} m<extra></extra>"
        heatmap = go.Heatmap(
            z=z_plot,
            x=lon_axis,
            y=lat_axis,
            customdata=pos,
            colorscale=_plotly_colorscale(_cm_depth()),
            colorbar=colorbar,
            hoverongaps=False,
            hovertemplate=hover,
        )
    else:
        z_plot = depth_masked
        heatmap = go.Heatmap(
            z=z_plot,
            x=lon_axis,
            y=lat_axis,
            colorscale=_plotly_colorscale(_cm_depth()),
            colorbar=dict(title="Depth (m)", thickness=15),
            hoverongaps=False,
            hovertemplate="lon: %{x:.3f}<br>lat: %{y:.3f}<br>depth: %{z:.1f} m<extra></extra>",
        )

    fig = go.Figure(heatmap)
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis_scaleanchor="x",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    html_path = png_path.with_suffix(".html")
    fig.write_html(str(html_path))


def _save_straits_html(
    lon: npt.NDArray,
    lat: npt.NDArray,
    depth_masked: npt.NDArray,
    records: list[dict],
    colours: dict[str, str],
    png_path: Path,
) -> None:
    """Write a zoomable plotly map of the strait analysis."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    lon_axis = lon[0, :] if lon.ndim == 2 else lon
    lat_axis = lat[:, 0] if lat.ndim == 2 else lat

    traces = [go.Heatmap(
        z=depth_masked, x=lon_axis, y=lat_axis,
        colorscale=_plotly_colorscale(_cm_depth()),
        colorbar=dict(title="Depth (m)", thickness=15),
        hoverongaps=False,
        hovertemplate="lon: %{x:.3f}<br>lat: %{y:.3f}<br>depth: %{z:.1f} m<extra></extra>",
        name="Depth",
    )]

    for cat, colour in colours.items():
        pts = [r for r in records if r.get("category") == cat]
        if not pts:
            continue
        hover = [
            f"lon={p['lon']:.3f}, lat={p['lat']:.3f}<br>"
            f"sill ratio={p.get('sill_ratio','?')}, "
            f"area ratio={p.get('area_ratio','?')}<br>"
            f"fix: {p.get('suggested_fix','')}"
            for p in pts
        ]
        traces.append(go.Scatter(
            x=[p["lon"] for p in pts],
            y=[p["lat"] for p in pts],
            mode="markers",
            marker=dict(symbol="x", size=10, color=colour,
                        line=dict(width=2, color=colour)),
            name=cat,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title="Strait / connectivity analysis (interactive)",
        xaxis_title="Longitude", yaxis_title="Latitude",
        yaxis_scaleanchor="x",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=20, t=50, b=80),
    )
    fig.write_html(str(png_path.with_suffix(".html")))
