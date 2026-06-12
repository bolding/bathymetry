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

    @staticmethod
    def _anchor(heading: str) -> str:
        """GitHub-Flavored Markdown anchor from a heading string."""
        import re
        return "#" + re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Table of contents
        toc = ["## Contents", ""]
        for sec in self._sections:
            toc.append(f"- [{sec['heading']}]({self._anchor(sec['heading'])})")
        toc.append("")

        lines: list[str] = [
            f"# {self.title}", "",
            f"*Generated: {self._created}*", "",
        ] + toc
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


def update_fixes_yaml(
    records: list[dict],
    path: str | Path,
    bridge_records: list[dict] | None = None,
    thalweg_fixes: list[dict] | None = None,
    phantom_island_fixes: list[dict] | None = None,
) -> None:
    """Merge new fix suggestions into fixes.yaml, preserving applied flags.

    Each entry has an ``applied`` field (default ``false``).  Set it to
    ``true`` in the file to have the fix applied on the next run.  Entries
    with ``applied: true`` are kept permanently as an audit trail even when
    the underlying deficit is resolved.  Unapplied entries that are no longer
    suggested are removed automatically.

    Example fixes.yaml entry::

        b001:
          lon: 10.123
          lat: 55.456
          action: open_cell
          depth: 15.0
          applied: false
          comment: "BLOCKED — no fine wet path"
    """
    import yaml as _yaml
    import re as _re
    import unicodedata as _ud

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _slug(name: str) -> str:
        ascii_name = _ud.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        return _re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")

    _key_prefix = {"BLOCKED": "b", "SILL_DEFICIT": "s", "AREA_DEFICIT": "a",
                   "LAND_BRIDGE": "lb"}
    _cat_desc = {
        "BLOCKED":      "No fine wet path — open_cell to reconnect",
        "SILL_DEFICIT": "Coarse sill too shallow — set_depth to fine-grid sill",
        "AREA_DEFICIT": "Cross-section under-represented — deepen to improve transport",
        "LAND_BRIDGE":  "Land cell (wet_frac > 0) between disconnected basins — open_cell",
    }

    # ── flat suggestions (strait / bridge fixes) ─────────────────────────────
    flat_groups: dict[str, list[dict]] = {
        "BLOCKED": [], "SILL_DEFICIT": [], "AREA_DEFICIT": [], "LAND_BRIDGE": [],
    }
    for r in records:
        cat = r["category"]
        if cat == "BLOCKED":
            flat_groups["BLOCKED"].append({"lon": r["lon"], "lat": r["lat"],
                                           "action": "open_cell", "depth": r["sill_depth_fine"],
                                           "comment": "BLOCKED — no fine wet path"})
        elif cat == "SILL_DEFICIT":
            flat_groups["SILL_DEFICIT"].append({"lon": r["lon"], "lat": r["lat"],
                                                "action": "set_depth", "value": r["sill_depth_fine"],
                                                "comment": f"SILL_DEFICIT — sill_ratio={r['sill_ratio']:.2f}"})
        elif cat == "AREA_DEFICIT":
            flat_groups["AREA_DEFICIT"].append({"lon": r["lon"], "lat": r["lat"],
                                                "action": "set_depth",
                                                "value": round(r["sill_depth_fine"] * 0.9, 1),
                                                "comment": f"AREA_DEFICIT — area_ratio={r['area_ratio']:.2f}"})
    for r in (bridge_records or []):
        flat_groups["LAND_BRIDGE"].append({
            "lon": r["lon"], "lat": r["lat"],
            "action": "open_cell", "depth": r["estimated_depth"],
            "comment": (f"LAND_BRIDGE — wet_fraction={r['wet_fraction']:.2f}, "
                        f"bridges {r['n_components']} basin(s), "
                        f"estimated depth {r['estimated_depth']} m"),
        })

    new_flat: dict[str, dict] = {}
    for cat, fixes in flat_groups.items():
        pfx = _key_prefix[cat]
        for idx, fix in enumerate(fixes, start=1):
            new_flat[f"{pfx}{idx:03d}"] = fix

    # ── thalweg suggestions: nested by waypoint ───────────────────────────────
    # new_tw[group_key] = {"wp_name": ..., "entries": [fix, ...]}
    new_tw: dict[str, dict] = {}
    for r in (thalweg_fixes or []):
        wp_name = r.get("thalweg_name", "thalweg")
        gk = f"tw_{_slug(wp_name)}"
        if gk not in new_tw:
            new_tw[gk] = {"wp_name": wp_name, "entries": []}
        new_tw[gk]["entries"].append({
            "lon":     r["lon"],
            "lat":     r["lat"],
            "action":  r["action"],
            "value":   r["value"],
            "comment": r.get("comment", "THALWEG — coarse cell too shallow"),
        })

    # ── read existing file ────────────────────────────────────────────────────
    existing_flat: dict[str, dict] = {}
    # Full existing thalweg group data — preserved verbatim when a group is
    # not regenerated (e.g. step-4d call without thalweg_fixes), so that
    # user-set applied flags and entries are never silently dropped.
    existing_tw: dict[str, dict] = {}  # gk → {applied, wp_name, entries}
    existing_pi_applied: bool = False   # preserve user's applied flag for pi group
    if path.exists():
        with open(path) as f:
            data = _yaml.safe_load(f) or {}
        raw = data.get("fixes") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if not isinstance(v, dict):
                    continue
                if "lon" in v and "lat" in v:
                    existing_flat[str(k)] = dict(v)
                elif str(k) == "phantom_islands" and "applied" in v:
                    existing_pi_applied = bool(v.get("applied", False))
                elif str(k).startswith("tw_") and "applied" in v:
                    # Reconstruct entry list from numbered sub-keys
                    _entries = [dict(sv) for sk, sv in v.items()
                                if sk != "applied" and isinstance(sv, dict)
                                and "lon" in sv and "lat" in sv]
                    # Recover a display name by un-slugging the key
                    _wp_display = str(k)[3:].replace("_", " ").title()
                    existing_tw[str(k)] = {
                        "applied": bool(v.get("applied", False)),
                        "wp_name": _wp_display,
                        "entries": _entries,
                    }

    # ── merge flat entries ────────────────────────────────────────────────────
    _tol = 0.05
    matched_new: set[str] = set()
    merged_flat: dict[str, dict] = {}

    for ex_key, ex_entry in existing_flat.items():
        ex_lon = float(ex_entry.get("lon", 0))
        ex_lat = float(ex_entry.get("lat", 0))
        best_key: str | None = None
        best_d = _tol
        for new_key, new_entry in new_flat.items():
            if new_key in matched_new:
                continue
            d = abs(float(new_entry["lon"]) - ex_lon) + abs(float(new_entry["lat"]) - ex_lat)
            if d < best_d:
                best_d, best_key = d, new_key
        if best_key:
            merged_flat[ex_key] = {**new_flat[best_key],
                                   "applied": bool(ex_entry.get("applied", False))}
            matched_new.add(best_key)
        elif ex_entry.get("applied", False):
            merged_flat[ex_key] = ex_entry
    for new_key, new_entry in new_flat.items():
        if new_key not in matched_new:
            merged_flat[new_key] = {**new_entry, "applied": False}

    # ── write ─────────────────────────────────────────────────────────────────
    def _write_flat_entry(fh, key: str, entry: dict) -> None:
        fh.write(f"  {key}:\n")
        fh.write(f"    lon:     {entry['lon']}\n")
        fh.write(f"    lat:     {entry['lat']}\n")
        fh.write(f"    action:  {entry['action']}\n")
        if "value" in entry:
            fh.write(f"    value:   {entry['value']}\n")
        if "depth" in entry:
            fh.write(f"    depth:   {entry['depth']}\n")
        fh.write(f"    applied: {'true' if entry.get('applied') else 'false'}\n")
        fh.write(f"    comment: \"{entry.get('comment', '')}\"\n")

    def _write_tw_group(fh, gk: str, wp_name: str,
                        entries: list[dict], applied: bool) -> None:
        fh.write(f"\n  # --- THALWEG: {wp_name} ---\n")
        fh.write(f"  {gk}:\n")
        fh.write(f"    applied: {'true' if applied else 'false'}"
                 f"   # set true to apply all {wp_name} fixes\n")
        for idx, e in enumerate(entries, start=1):
            fh.write(f"    \"{idx:03d}\":\n")
            fh.write(f"      lon:     {e['lon']}\n")
            fh.write(f"      lat:     {e['lat']}\n")
            fh.write(f"      action:  {e['action']}\n")
            if "value" in e:
                fh.write(f"      value:   {e['value']}\n")
            if "depth" in e:
                fh.write(f"      depth:   {e['depth']}\n")
            fh.write(f"      comment: \"{e.get('comment', '')}\"\n")

    def _write_pi_group(fh, entries: list[dict], applied: bool) -> None:
        fh.write("\n  # --- PHANTOM ISLANDS: ocean cells that are mostly land in the fine grid ---\n")
        fh.write("  phantom_islands:\n")
        fh.write(f"    applied: {'true' if applied else 'false'}"
                 "   # set true to mask all detected phantom island cells\n")
        for idx, e in enumerate(entries, start=1):
            notes = []
            if e.get("cluster_size", 1) > 1:
                notes.append(f"{e['cluster_size']}-cell cluster")
            if e.get("fine_cells"):
                notes.append(f"{e['fine_cells']} fine px")
            note_str = ("; " + ", ".join(notes)) if notes else ""
            fh.write(f"    \"{idx:03d}\":\n")
            fh.write(f"      lon:          {e['lon']}\n")
            fh.write(f"      lat:          {e['lat']}\n")
            fh.write(f"      action:       mask_cell\n")
            fh.write(f"      wet_fraction: {e['wet_fraction']:.3f}\n")
            fh.write(f"      comment: \"phantom island — "
                     f"wf={e['wet_fraction']:.2f}, depth={e['depth']:.1f} m"
                     f"{note_str}\"\n")

    with open(path, "w") as fh:
        fh.write("# fixes.yaml — edit applied: true/false, then re-run with --accept-fixes.\n")
        fh.write("# Thalweg groups: set applied: true on the group key to apply all fixes in it.\n")
        fh.write("# phantom_islands: set applied: true on the group key to mask all flagged cells.\n\n")
        fh.write("fixes:\n")

        cur_cat: str | None = None
        for key, entry in merged_flat.items():
            cat = next((c for c, p in _key_prefix.items() if key.startswith(p)), "OTHER")
            if cat != cur_cat:
                fh.write(f"\n  # --- {cat}: {_cat_desc.get(cat, '')} ---\n")
                cur_cat = cat
            _write_flat_entry(fh, key, entry)

        # Write fresh thalweg groups (with preserved applied flags).
        for gk, grp in new_tw.items():
            applied = existing_tw.get(gk, {}).get("applied", False)
            _write_tw_group(fh, gk, grp["wp_name"], grp["entries"], applied)

        # Re-emit existing groups not in new_tw so user edits are never lost.
        for gk, grp in existing_tw.items():
            if gk not in new_tw:
                _write_tw_group(fh, gk, grp["wp_name"], grp["entries"],
                                grp["applied"])

        # Phantom islands group — always rewritten from fresh detection.
        if phantom_island_fixes:
            _write_pi_group(fh, phantom_island_fixes, existing_pi_applied)


save_fixes_yaml = update_fixes_yaml  # backward-compat alias


def mark_all_applied(path: str | Path) -> int:
    """Set applied: true for every entry in fixes.yaml and write back.

    Called after --accept-fixes so the audit trail reflects which fixes
    were actually applied in this run.  Returns the number of entries updated.
    """
    import yaml as _yaml

    path = Path(path)
    if not path.exists():
        return 0
    with open(path) as f:
        data = _yaml.safe_load(f) or {}
    raw = data.get("fixes") or {}
    if not isinstance(raw, dict):
        return 0
    # Re-read via update_fixes_yaml to preserve all formatting logic,
    # but we need a simpler in-place rewrite here.
    count = 0
    with open(path) as f:
        lines = f.readlines()
    out: list[str] = []
    for line in lines:
        if line.lstrip().startswith("applied:"):
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "applied: true\n")
            count += 1
        else:
            out.append(line)
    with open(path, "w") as f:
        f.writelines(out)
    return count


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
# Coastline / land helpers
# ---------------------------------------------------------------------------

def _add_land_feature(
    ax,
    scale: str = "10m",
    land_color: str = "tan",
    land_zorder: int = 2,
    coast_zorder: int = 3,
    linewidth: float = 0.5,
    fill_land: bool = True,
) -> None:
    """Add land polygon and coastline to a Cartopy axes.

    *scale* can be a NaturalEarth resolution (``"10m"``, ``"50m"``,
    ``"110m"``) or a GSHHG scale prefixed with ``"gshhg-"``
    (``"gshhg-f"``, ``"gshhg-h"``, ``"gshhg-i"``, ``"gshhg-l"``,
    ``"gshhg-c"`` for full / high / intermediate / low / coarse).
    GSHHG is significantly finer than NaturalEarth and recommended for
    high-resolution regional or fjord domains.

    Set *fill_land=False* to draw only the coastline (no land fill).
    Set *scale="none"* to skip all coastline/land features entirely.
    """
    if scale == "none":
        ax.set_facecolor("white")
        return
    import cartopy.feature as cfeature  # noqa: PLC0415
    import cartopy.io.shapereader as shapereader  # noqa: PLC0415

    if scale.startswith("gshhg"):
        gshhg_scale = scale.split("-", 1)[1] if "-" in scale else "h"
        # Pre-check the data is available — falls back to NE 10m if not.
        # The NGDC download mirror is sometimes offline; the SOEST mirror at
        # https://www.soest.hawaii.edu/pwessel/gshhg/ is a reliable fallback.
        # To install manually: download gshhg-shp-2.3.7.zip from SOEST and
        # extract its GSHHS_shp/ tree into ~/.local/share/cartopy/shapefiles/gshhs/
        try:
            shapereader.gshhs(gshhg_scale, 1)  # raises if not cached and download fails
        except Exception:
            import warnings  # noqa: PLC0415
            warnings.warn(
                f"GSHHG '{scale}' not available (download failed?) — "
                "falling back to NaturalEarth 10m coastline.  "
                "Install manually: download gshhg-shp-2.3.7.zip from "
                "https://www.soest.hawaii.edu/pwessel/gshhg/ and extract "
                "GSHHS_shp/ into ~/.local/share/cartopy/shapefiles/gshhs/",
                stacklevel=3,
            )
            if fill_land:
                ax.add_feature(cfeature.LAND.with_scale("10m"),
                               facecolor=land_color, zorder=land_zorder)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                           linewidth=linewidth, zorder=coast_zorder)
            return
        facecolor = land_color if fill_land else "none"
        feat = cfeature.GSHHSFeature(
            scale=gshhg_scale,
            levels=[1],
            facecolor=facecolor,
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_feature(feat, zorder=land_zorder)
    else:
        if fill_land:
            ax.add_feature(cfeature.LAND.with_scale(scale),
                           facecolor=land_color, zorder=land_zorder)
        ax.add_feature(cfeature.COASTLINE.with_scale(scale),
                       linewidth=linewidth, zorder=coast_zorder)


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
    coastline_scale: str = "10m",
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

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, diff,  # type: ignore[union-attr]
                           cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        plt.colorbar(im, ax=ax, label="Depth difference (m)", fraction=0.046, pad=0.04)
        _add_land_feature(ax, coastline_scale, fill_land=False)  # type: ignore[union-attr]
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
    coastline_scale: str = "10m",
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

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.pcolormesh(lon, lat, comp,  # type: ignore[union-attr]
                      cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        _add_land_feature(ax, coastline_scale, fill_land=False)  # type: ignore[union-attr]
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
    coastline_scale: str = "10m",
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
    if coastline_scale == "none":
        import copy
        cmap = copy.copy(cmap)
        cmap.set_bad("white")
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

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon, lat, masked, transform=ccrs.PlateCarree(),
                            rasterized=True, **kw)  # type: ignore[union-attr]
        _add_land_feature(ax, coastline_scale)  # type: ignore[union-attr]
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
    coastline_scale: str = "10m",
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

        fig, axes = plt.subplots(
            1, n, figsize=(6 * n, 5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        axes_list = [axes] if n == 1 else list(axes)
        pcm = None
        for ax, field, mask, title in zip(axes_list, fields, masks, titles):
            masked = np.where(mask, field, np.nan)
            pcm = ax.pcolormesh(lon, lat, masked, transform=ccrs.PlateCarree(), **kw)  # type: ignore[union-attr]
            _add_land_feature(ax, coastline_scale)  # type: ignore[union-attr]
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

        # Grey background so coarse grid lines are visible over land/NaN areas
        ax_plan.set_facecolor("#c8c8c8")

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
            _gkw = dict(color="black", linewidth=1.2, alpha=0.8, zorder=3)
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
    coastline_scale: str = "10m",
    phantom_island_records: Optional[list[dict]] = None,
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

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        masked = np.where(mask, depth, np.nan)
        pcm = ax.pcolormesh(lon, lat, masked, cmap=_cm_depth(),
                            transform=ccrs.PlateCarree())
        _add_land_feature(ax, coastline_scale)  # type: ignore[union-attr]
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

    pi_colour = "cyan"
    if phantom_island_records:
        pi_kw: dict = dict(color=pi_colour, marker="D", markersize=7,
                           linewidth=0, markeredgecolor="navy",
                           markeredgewidth=0.8, zorder=6)
        for rec in phantom_island_records:
            if transform is not None:
                ax.plot(rec["lon"], rec["lat"], transform=transform, **pi_kw)  # type: ignore[union-attr]
            else:
                ax.plot(rec["lon"], rec["lat"], **pi_kw)  # type: ignore[union-attr]

    patches = [mpatches.Patch(color=c, label=k) for k, c in colours.items()]
    if phantom_island_records:
        patches.append(mpatches.Patch(color=pi_colour, label="PHANTOM_ISLAND"))
    ax.legend(handles=patches, loc="upper right")  # type: ignore[union-attr]
    _add_colorbar(fig, ax, pcm, "Depth (m)")
    pi_note = (f", {len(phantom_island_records)} phantom island(s)"
               if phantom_island_records else "")
    ax.set_title(
        f"Strait / connectivity concerns — {len(strait_records)} flagged interface(s){pi_note}"
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
            _add_land_feature(ax_ins, "50m", land_zorder=1, coast_zorder=2)
            ax_ins.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
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
    _save_straits_html(lon, lat, masked, strait_records, colours, path,
                       phantom_island_records=phantom_island_records)


def plot_basins(
    lon: npt.NDArray,
    lat: npt.NDArray,
    labels: npt.NDArray,
    title: str,
    path: str | Path,
    nkeep: int = 1,
    coastline_scale: str = "10m",
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

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.pcolormesh(lon, lat, scalar,  # type: ignore[union-attr]
                      cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        _add_land_feature(ax, coastline_scale, fill_land=False, linewidth=0.6)  # type: ignore[union-attr]
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

def _1d_axes(lon: npt.NDArray, lat: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Extract 1-D lon/lat axes from 1-D or 2-D coordinate arrays."""
    lon_axis = lon[0, :] if lon.ndim == 2 else lon
    lat_axis = lat[:, 0] if lat.ndim == 2 else lat
    return lon_axis, lat_axis
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

    lon_axis, lat_axis = _1d_axes(lon, lat)

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

    lon_axis, lat_axis = _1d_axes(lon, lat)

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
    phantom_island_records: Optional[list[dict]] = None,
) -> None:
    """Write a zoomable plotly map of the strait analysis."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    lon_axis, lat_axis = _1d_axes(lon, lat)

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

    if phantom_island_records:
        hover_pi = [
            f"lon={p['lon']:.4f}, lat={p['lat']:.4f}<br>"
            f"wf={float(p.get('wet_fraction', 0)):.2f}, depth={float(p.get('depth', 0)):.1f} m<br>"
            f"fine cells={p.get('fine_cells', '?')}"
            + (f", cluster_size={p['cluster_size']}" if p.get("cluster_size", 1) > 1 else "")
            for p in phantom_island_records
        ]
        traces.append(go.Scatter(
            x=[p["lon"] for p in phantom_island_records],
            y=[p["lat"] for p in phantom_island_records],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="cyan",
                        line=dict(width=1, color="navy")),
            name="PHANTOM_ISLAND",
            text=hover_pi,
            hovertemplate="%{text}<extra></extra>",
        ))

    n_pi = len(phantom_island_records) if phantom_island_records else 0
    pi_note = f", {n_pi} phantom island(s)" if n_pi else ""
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Strait / connectivity analysis (interactive){pi_note}",
        xaxis_title="Longitude", yaxis_title="Latitude",
        yaxis_scaleanchor="x",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=20, t=50, b=80),
    )
    fig.write_html(str(png_path.with_suffix(".html")))


def _plot_thalweg_failed(record: dict[str, Any], coarse_ds: Any,
                         png_path: str | Path,
                         fine_ds: Any = None,
                         coastline_scale: str = "10m") -> None:
    """Single-panel map showing why a thalweg failed.

    The coarse depth field is shown as the main background.  When *fine_ds*
    is supplied, the fine-resolution wet mask is overlaid as a semi-transparent
    blue layer — this immediately reveals whether the gap is a GEBCO dry
    barrier or a stop-coordinate placement issue.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cmocean
    except ImportError:
        return

    name   = record.get("name", "Thalweg")
    bbox   = record.get("bbox")         # [lo_min, lo_max, la_min, la_max] or None
    stops  = record.get("stops_lonlat") or []
    reason = record.get("failed_reason", "unknown reason")

    def _get_coord(ds: Any, candidates: list[str]) -> Any:
        for c in candidates:
            if c in ds.coords or c in ds:
                arr = ds[c].values if hasattr(ds[c], "values") else ds[c]
                if arr.ndim >= 1:
                    return arr
        return None

    _clon_raw = _get_coord(coarse_ds, ["lon", "longitude", "lont", "nav_lon"])
    _clat_raw = _get_coord(coarse_ds, ["lat", "latitude", "latt", "nav_lat"])
    _cdepth   = coarse_ds["depth"].values if hasattr(coarse_ds["depth"], "values") else coarse_ds["depth"]
    _cmask    = coarse_ds["mask"].values  if hasattr(coarse_ds["mask"],  "values") else coarse_ds["mask"]
    coarse_depth_bg = np.where(_cmask.astype(bool), _cdepth, np.nan)
    if _clon_raw is not None and _clat_raw is not None:
        if _clon_raw.ndim == 1 and _clat_raw.ndim == 1:
            coarse_lon2d, coarse_lat2d = np.meshgrid(_clon_raw, _clat_raw)
        else:
            coarse_lon2d, coarse_lat2d = _clon_raw, _clat_raw
    else:
        coarse_lon2d = coarse_lat2d = None

    geo = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(1, 1, 1, projection=geo)

    # extent: use bbox if given, else full coarse domain
    if bbox is not None:
        lo0, lo1, la0, la1 = bbox
        margin = max((lo1 - lo0) * 0.25, (la1 - la0) * 0.25, 0.3)
        ext = [lo0 - margin, lo1 + margin, la0 - margin, la1 + margin]
    elif coarse_lon2d is not None:
        ext = [float(coarse_lon2d.min()), float(coarse_lon2d.max()),
               float(coarse_lat2d.min()), float(coarse_lat2d.max())]
    else:
        ext = [-180, 180, -90, 90]
    ax.set_extent(ext, crs=geo)

    # Cartopy LAND goes behind all data so it fills gaps without hiding channels.
    _add_land_feature(ax, coastline_scale, land_color="#e8dcc8", land_zorder=1, coast_zorder=6)

    # Primary background: fine-resolution GEBCO depth when available (most
    # informative for diagnosing path failures); fall back to coarse depth.
    _colorbar_drawn = False
    if fine_ds is not None:
        try:
            f_lon = fine_ds.lon.values
            f_lat = fine_ds.lat.values
            f_depth_raw = fine_ds["depth"].values.astype(float)
            f_land  = fine_ds["land"].values.astype(bool)
            f_depth = np.where(f_land, np.nan, f_depth_raw)
            lo0_, lo1_, la0_, la1_ = ext
            lon_mask = (f_lon >= lo0_) & (f_lon <= lo1_)
            lat_mask = (f_lat >= la0_) & (f_lat <= la1_)
            f_lon_c   = f_lon[lon_mask]
            f_lat_c   = f_lat[lat_mask]
            f_depth_c = f_depth[np.ix_(lat_mask, lon_mask)]
            if f_lon_c.size and f_lat_c.size and np.isfinite(f_depth_c).any():
                f_lon2d, f_lat2d = np.meshgrid(f_lon_c, f_lat_c)
                vmax_f = float(np.nanmax(f_depth_c))
                pcm = ax.pcolormesh(f_lon2d, f_lat2d, f_depth_c,
                                    cmap=cmocean.cm.deep, vmin=0, vmax=vmax_f,
                                    shading="auto", transform=geo, zorder=2)
                plt.colorbar(pcm, ax=ax, label="Fine depth (m)", shrink=0.75, pad=0.02)
                _colorbar_drawn = True
        except Exception:
            pass  # best-effort; fall through to coarse

    if not _colorbar_drawn and coarse_lon2d is not None:
        lo0_, lo1_, la0_, la1_ = ext
        in_ext = (
            (coarse_lon2d >= lo0_) & (coarse_lon2d <= lo1_) &
            (coarse_lat2d >= la0_) & (coarse_lat2d <= la1_)
        )
        local_depths = coarse_depth_bg[in_ext]
        vmax = float(np.nanmax(local_depths)) if np.isfinite(local_depths).any() else 1.0
        pcm = ax.pcolormesh(coarse_lon2d, coarse_lat2d, coarse_depth_bg,
                            cmap=cmocean.cm.deep, vmin=0, vmax=vmax,
                            shading="auto", transform=geo, zorder=2)
        plt.colorbar(pcm, ax=ax, label="Coarse depth (m)", shrink=0.75, pad=0.02)
    _gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey",
                       alpha=0.5, x_inline=False, y_inline=False)
    _gl.top_labels   = False
    _gl.right_labels = False

    # bbox rectangle
    if bbox is not None:
        lo0, lo1, la0, la1 = bbox
        rect = mpatches.FancyBboxPatch(
            (lo0, la0), lo1 - lo0, la1 - la0,
            boxstyle="square,pad=0", linewidth=2, edgecolor="crimson",
            facecolor="none", linestyle="--", transform=geo, zorder=5,
        )
        ax.add_patch(rect)

    # stops
    colours = ["limegreen", "dodgerblue", "gold", "magenta"]
    markers = ["^", "s", "D", "o"]
    for si, (slo, sla) in enumerate(stops):
        ax.plot(slo, sla, marker=markers[si % len(markers)], ms=9,
                color=colours[si % len(colours)], markeredgecolor="k",
                markeredgewidth=0.5, linestyle="none",
                transform=geo, zorder=9,
                label=f"Stop {si+1}: ({slo:.2f},{sla:.2f})")

    ax.set_title(f"{name} — FAILED\n{reason}", fontsize=10, color="crimson")
    if stops:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)
    fig.tight_layout()
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_thalweg_comparison(
    record: dict[str, Any],
    fine_ds: Any,
    coarse_ds: Any,
    png_path: str | Path,
    coastline_scale: str = "10m",
) -> None:
    """Two-panel thalweg comparison: map (left) + depth profile (right).

    Parameters
    ----------
    record:
        One entry from the list returned by thalweg.compute_strait_thalwegs /
        boundary_thalwegs / waypoint_thalwegs.  Must contain keys
        ``fine`` and ``coarse`` (each a dict with arrays lon/lat/dist_km/depth)
        and optionally ``name``, ``sill_depth_m``, ``sill_lon``, ``sill_lat``.
    fine_ds, coarse_ds:
        xarray Datasets with variables ``depth`` and ``mask`` (or ``land``); must have 2-D
        lon/lat coordinate arrays accessible as ``lon`` / ``lat``.
    png_path:
        Output PNG path (saved at 150 dpi).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean

    name  = record.get("name", "Thalweg")

    if record.get("failed"):
        _plot_thalweg_failed(record, coarse_ds, png_path, fine_ds=fine_ds,
                             coastline_scale=coastline_scale)
        return

    fine  = record["fine"]
    coarse = record["coarse"]

    # ── helper ──────────────────────────────────────────────────────────────
    def _get_coord(ds: Any, candidates: list[str]) -> Any:
        for c in candidates:
            if c in ds.coords or c in ds:
                arr = ds[c].values if hasattr(ds[c], "values") else ds[c]
                if arr.ndim >= 1:
                    return arr
        return None

    # ── coarse grid arrays (map background) ──────────────────────────────────
    _clon_raw = _get_coord(coarse_ds, ["lon", "longitude", "lont", "nav_lon"])
    _clat_raw = _get_coord(coarse_ds, ["lat", "latitude", "latt", "nav_lat"])
    if _clon_raw is not None and _clat_raw is not None:
        if _clon_raw.ndim == 1 and _clat_raw.ndim == 1:
            coarse_lon2d, coarse_lat2d = np.meshgrid(_clon_raw, _clat_raw)
        else:
            coarse_lon2d, coarse_lat2d = _clon_raw, _clat_raw
    else:
        coarse_lon2d = coarse_lat2d = None
    _cdepth = coarse_ds["depth"].values if hasattr(coarse_ds["depth"], "values") else coarse_ds["depth"]
    _cmask  = coarse_ds["mask"].values  if hasattr(coarse_ds["mask"],  "values") else coarse_ds["mask"]
    coarse_depth_bg = np.where(_cmask.astype(bool), _cdepth, np.nan)

    # ── figure layout ────────────────────────────────────────────────────────
    geo = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1], wspace=0.35)
    ax_map  = fig.add_subplot(gs[0], projection=geo)
    ax_prof = fig.add_subplot(gs[1])

    # ── map extent — zoom to path bbox, clipped to coarse domain ────────────
    p_lon_min  = float(np.nanmin(fine["lon"]))
    p_lon_max  = float(np.nanmax(fine["lon"]))
    p_lat_min  = float(np.nanmin(fine["lat"]))
    p_lat_max  = float(np.nanmax(fine["lat"]))
    span       = max(p_lon_max - p_lon_min, p_lat_max - p_lat_min, 0.5)
    margin     = span * 0.30
    if coarse_lon2d is not None:
        c_lon_min = float(np.nanmin(coarse_lon2d))
        c_lon_max = float(np.nanmax(coarse_lon2d))
        c_lat_min = float(np.nanmin(coarse_lat2d))
        c_lat_max = float(np.nanmax(coarse_lat2d))
        ext = [max(p_lon_min - margin, c_lon_min),
               min(p_lon_max + margin, c_lon_max),
               max(p_lat_min - margin, c_lat_min),
               min(p_lat_max + margin, c_lat_max)]
    else:
        ext = [p_lon_min - margin, p_lon_max + margin,
               p_lat_min - margin, p_lat_max + margin]
    ax_map.set_extent(ext, crs=geo)  # type: ignore[union-attr]

    # ── shared depth colormap — scale to depths visible within the map extent
    if coarse_lon2d is not None:
        _in_ext = (
            (coarse_lon2d >= ext[0]) & (coarse_lon2d <= ext[1]) &
            (coarse_lat2d >= ext[2]) & (coarse_lat2d <= ext[3])
        )
        _vis = coarse_depth_bg[_in_ext]
        _vis_max = float(np.nanmax(_vis)) if _vis.size > 0 and np.any(np.isfinite(_vis)) else 1.0
    else:
        _vis_max = 1.0
    depth_vmax = max(1.0, float(np.nanmax(fine["depth"])), _vis_max)
    cmap = cmocean.cm.deep
    norm = plt.Normalize(vmin=0, vmax=depth_vmax)

    # ── left panel: map — coarse grid as background ──────────────────────────
    if coarse_lon2d is not None and coarse_lat2d is not None:
        pcm = ax_map.pcolormesh(  # type: ignore[union-attr]
            coarse_lon2d, coarse_lat2d, coarse_depth_bg,
            cmap=cmap, norm=norm, shading="auto", transform=geo,
        )
        plt.colorbar(pcm, ax=ax_map, label="Depth (m)", shrink=0.75, pad=0.02)

    _add_land_feature(ax_map, coastline_scale, land_color="#e8dcc8", land_zorder=2, coast_zorder=3)  # type: ignore[union-attr]
    _gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color="grey",   # type: ignore[union-attr]
                           alpha=0.5, x_inline=False, y_inline=False)
    _gl.top_labels   = False
    _gl.right_labels = False

    # fine thalweg path
    ax_map.plot(fine["lon"], fine["lat"], color="white", lw=2.0,   # type: ignore[union-attr]
                transform=geo, zorder=4)
    ax_map.plot(fine["lon"], fine["lat"], color="steelblue", lw=1.2,  # type: ignore[union-attr]
                transform=geo, label="Fine thalweg", zorder=5)

    # tiny white dot at each unique coarse cell centre the path visits —
    # shows grid resolution without covering the pcolormesh background.
    if coarse_lon2d is not None:
        try:
            from scipy.spatial import cKDTree as _KDTree  # type: ignore[import-untyped]
            _c_pts = np.column_stack([coarse_lon2d.ravel(), coarse_lat2d.ravel()])
            _p_pts = np.column_stack([np.asarray(fine["lon"]),
                                      np.asarray(fine["lat"])])
            _, _cidx = _KDTree(_c_pts).query(_p_pts)
            _cidx_u  = np.unique(_cidx)
            ax_map.scatter(  # type: ignore[call-arg,union-attr]
                _c_pts[_cidx_u, 0], _c_pts[_cidx_u, 1],
                c="white", s=1.5, zorder=8, alpha=0.7, transform=geo,
            )
        except ImportError:
            pass

    # start / end markers — prefer stored boundary detection coordinates so
    # markers land at the actual domain edge even when the path is clipped.
    _start_lon = record.get("start_lon", float(fine["lon"][0]))
    _start_lat = record.get("start_lat", float(fine["lat"][0]))
    _end_lon   = record.get("end_lon",   float(fine["lon"][-1]))
    _end_lat   = record.get("end_lat",   float(fine["lat"][-1]))
    ax_map.plot(  # type: ignore[union-attr]
        _start_lon, _start_lat,
        marker="^", ms=9, color="limegreen", markeredgecolor="k",
        markeredgewidth=0.5, linestyle="none",
        transform=geo, label="Start", zorder=9,
    )
    ax_map.plot(  # type: ignore[union-attr]
        _end_lon, _end_lat,
        marker="s", ms=9, color="crimson", markeredgecolor="k",
        markeredgewidth=0.5, linestyle="none",
        transform=geo, label="End", zorder=9,
    )

    # sill marker — derive position from fine profile (works for all modes)
    sill_lon = record.get("sill_lon")
    sill_lat = record.get("sill_lat")
    if sill_lon is None or sill_lat is None:
        sill_dist = fine.get("sill_dist_km", float("nan"))
        if np.isfinite(sill_dist):
            si = int(np.argmin(np.abs(fine["dist_km"] - sill_dist)))
            sill_lon, sill_lat = float(fine["lon"][si]), float(fine["lat"][si])
    if sill_lon is not None and sill_lat is not None:
        ax_map.plot(  # type: ignore[union-attr]
            sill_lon, sill_lat,
            marker="o", linestyle="none", ms=11,
            markerfacecolor="none", markeredgecolor="gold", markeredgewidth=2,
            transform=geo, label="Sill", zorder=7,
        )

    ax_map.set_title(f"{name} — map")  # type: ignore[union-attr]
    ax_map.legend(fontsize=7, loc="best", ncol=2, framealpha=0.85)  # type: ignore[union-attr]

    # ── right panel: depth profile ───────────────────────────────────────────
    ax_prof.plot(fine["dist_km"],   fine["depth"],   color="steelblue",
                 lw=1.5, label="Fine")

    # Pre-fix raw coarse (only present when fixes have been applied)
    _prefixes = record.get("prefixes_coarse")
    if _prefixes is not None:
        ax_prof.scatter(
            np.asarray(_prefixes["dist_km"], dtype=float),
            np.asarray(_prefixes["depth"],   dtype=float),
            color="lightcoral", s=3, zorder=3,
            label=f"{_prefixes.get('label', 'pre-fix raw')} coarse",
        )

    # Raw coarse after fixes (always present)
    _raw_label = "Fixed raw coarse" if _prefixes is not None else "Raw coarse"
    ax_prof.scatter(coarse["dist_km"], coarse["depth"], color="darkorange",
                    s=4, zorder=4, label=_raw_label)

    _smooth_colors = ["forestgreen", "crimson", "purple", "saddlebrown"]
    for _si, (_lbl, _sc) in enumerate(record.get("smooth_coarse", {}).items()):
        _sc_dep = np.asarray(_sc["depth"], dtype=float)
        _sc_dist = np.asarray(_sc["dist_km"], dtype=float)
        _col = _smooth_colors[_si % len(_smooth_colors)]
        ax_prof.plot(_sc_dist, _sc_dep, color=_col, lw=1.4,
                     linestyle="--", zorder=5,
                     label=f"{_lbl} coarse")

    sill_d = record.get("sill_depth_m")
    if sill_d is not None:
        ax_prof.axhline(sill_d, color="red", ls="--", lw=1,
                        label=f"Fine sill {sill_d:.0f} m")

    sill_d_coarse = record.get("coarse_sill_depth_m")
    if sill_d_coarse is not None:
        ax_prof.axhline(sill_d_coarse, color="darkorange", ls="--", lw=1,
                        label=f"Coarse sill {sill_d_coarse:.0f} m")

    sill_dist = record.get("sill_dist_km")
    if sill_dist is not None:
        ax_prof.axvline(sill_dist, color="grey", ls=":", lw=1)

    ax_prof.invert_yaxis()
    ax_prof.set_xlabel("Along-path distance (km)")
    ax_prof.set_ylabel("Depth (m)")
    ax_prof.set_title(f"{name} — depth profile")
    ax_prof.legend(fontsize=7)

    fig.suptitle(name, fontsize=11, y=1.01)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_thalweg_html(record: dict, html_path: str) -> None:
    """Save a zoomable interactive plotly version of the thalweg comparison.

    Two panels: left = geographic scatter (path + sill), right = depth profile.
    Saved as a standalone HTML file.
    """
    import numpy as np
    from pathlib import Path
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fine   = record["fine"]
    coarse = record["coarse"]
    name   = record.get("name", "Thalweg")

    sill_dist = fine.get("sill_dist_km", float("nan"))
    sill_idx  = (int(np.argmin(np.abs(fine["dist_km"] - sill_dist)))
                 if np.isfinite(sill_dist) else None)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "scattergeo"}, {"type": "xy"}]],
        subplot_titles=[f"{name} — map", f"{name} — depth profile"],
    )

    # Left: geographic path
    fig.add_trace(go.Scattergeo(
        lon=fine["lon"], lat=fine["lat"],
        mode="lines",
        line={"color": "steelblue", "width": 2},
        name="Fine thalweg",
    ), row=1, col=1)

    # Coarse depth scatter on map
    c_valid = np.isfinite(coarse["depth"])
    if c_valid.any():
        fig.add_trace(go.Scattergeo(
            lon=fine["lon"][c_valid], lat=fine["lat"][c_valid],
            mode="markers",
            marker={"color": coarse["depth"][c_valid], "colorscale": "Oranges",
                    "size": 5, "reversescale": True,
                    "colorbar": {"title": "Coarse depth (m)", "x": 0.48}},
            name="Coarse depth",
        ), row=1, col=1)

    # Sill marker
    if sill_idx is not None:
        fig.add_trace(go.Scattergeo(
            lon=[float(fine["lon"][sill_idx])],
            lat=[float(fine["lat"][sill_idx])],
            mode="markers",
            marker={"symbol": "circle-open", "size": 14,
                    "color": "crimson", "line": {"width": 2}},
            name="Sill",
        ), row=1, col=1)

    fig.update_geos(
        lonaxis_range=[float(fine["lon"].min()) - 1, float(fine["lon"].max()) + 1],
        lataxis_range=[float(fine["lat"].min()) - 0.5, float(fine["lat"].max()) + 0.5],
        showcoastlines=True, coastlinecolor="black", coastlinewidth=0.8,
        showland=True, landcolor="#e8dcc8",
        showocean=True, oceancolor="#cde8f6",
        resolution=50,
        row=1, col=1,
    )

    # Right: depth profile
    fig.add_trace(go.Scatter(
        x=fine["dist_km"], y=fine["depth"],
        mode="lines", line={"color": "steelblue", "width": 1.5},
        name="Fine depth",
    ), row=1, col=2)

    c_valid2 = np.isfinite(coarse["depth"])
    if c_valid2.any():
        fig.add_trace(go.Scatter(
            x=coarse["dist_km"][c_valid2], y=coarse["depth"][c_valid2],
            mode="markers", marker={"color": "darkorange", "size": 3},
            name="Coarse depth",
        ), row=1, col=2)

    if sill_idx is not None:
        fig.add_hline(y=fine["depth"][sill_idx], line_dash="dash",
                      line_color="crimson", annotation_text=f"Sill {fine['depth'][sill_idx]:.0f} m",
                      row=1, col=2)
        fig.add_vline(x=float(fine["dist_km"][sill_idx]), line_dash="dot",
                      line_color="grey", row=1, col=2)

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=2)
    fig.update_xaxes(title_text="Along-path distance (km)", row=1, col=2)
    fig.update_layout(title_text=name, height=500, width=1100)

    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
