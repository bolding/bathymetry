# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Commands

```bash
# Install (requires esmpy / xesmf from conda)
pip install -e .

# Run the pipeline from a YAML config
bathymetry-regrid --config example_northsea.yaml

# Override config values on the CLI
bathymetry-regrid --config example_northsea.yaml --smooth-rx0 0.15 --name v2

# Dry-run: check what files exist, no download
bathymetry-regrid --config example_northsea.yaml --dryrun

# Skip the expensive regrid step (uses cached raw result from first run)
bathymetry-regrid --config example_northsea.yaml --skip-regrid

# Apply fixes from fixes.yaml:
#   --apply-all-fixes  applies ALL entries regardless of applied flag
#   --accept-fixes     applies only entries / groups with applied: true
bathymetry-regrid --config example_northsea.yaml --skip-regrid --apply-all-fixes
bathymetry-regrid --config example_northsea.yaml --skip-regrid --accept-fixes

# Explicit fixes file (any path)
bathymetry-regrid --config example_northsea.yaml --fixes-file report/my/fixes.yaml

# Equidistant cells: provide dlat, let dlon be computed from cos(lat_center)
# e.g. dlat=0.05° at 55°N → dlon≈0.0872°, grid 172×200 instead of 300×200
bathymetry-regrid --config example_northsea.yaml --dlat 0.05 --equidistant

# Run directly (no install needed if lib/ is on sys.path)
python cli/regrid.py --config example_northsea.yaml
```

## Architecture

```
cli/regrid.py          CLI entry point (entry point: bathymetry-regrid)
lib/
  grid.py              SphericalGrid, CartesianGrid (corner arrays for ESMF)
  reader.py            GEBCO and EMODnet GeoTIFF readers
  interpolate.py       xESMF conservative regridding; weight-file caching
  analysis.py          find_straits(), apply_fixes(), mask_regions(), flood-fill
  smooth.py            rx0 slope-factor smoothing via linprog
  thalweg.py           fine-vs-coarse thalweg extraction, depth-fix suggestions
  report.py            Markdown report, ASCII tables, Cartopy + plotly plots
  boundary.py          open-boundary T-grid coordinate CSV writer
```

### Pipeline steps (cli/regrid.py)

| Step | What |
|------|------|
| 1 | Build target grid |
| 2 | Read + clip source bathymetry |
| 3 | xESMF conservative regrid (may take a minute; result cached as `{name}_raw_regrid.nc`) |
| 4a | Apply user fixes (`fixes:` in config, `--accept-fixes`, `--apply-all-fixes`, `--fixes-file`) |
| 4b | Apply explicit `mask_regions:` (rectangle, polygon, point, ij_rectangle, ij_point) |
| 4c | Remove isolated ocean cells (flood-fill; keep *nkeep* largest basins) |
| 4c-ii | Detect LAND_BRIDGE cells (wet_fraction > 0 cells between disconnected basins) |
| 4d | Flag narrow / blocked interfaces (BLOCKED, SILL_DEFICIT, AREA_DEFICIT); write `fixes.yaml` |
| 4e | Thalweg analysis — fine-vs-coarse depth profiles, depth-fix suggestions (optional) |
| 5 | rx0 Haney slope smoothing via linear programming (optional) |
| 6 | Write output NetCDF + final plots |

### Fixes workflow

After step 4d and 4e, `fixes.yaml` is written to the report directory.
It is a **dict** (not a list), with one entry per suggested fix:

```yaml
fixes:

  # --- BLOCKED: No fine wet path ---
  b001:
    lon: 10.751275
    lat: 54.933333
    action: open_cell
    depth: 0.0
    applied: false
    comment: "BLOCKED — no fine wet path"

  # --- THALWEG: Great Belt ---
  tw_great_belt:
    applied: false   # set true to apply all Great Belt fixes
    "001":
      lon: 10.92562
      lat: 54.6
      action: set_depth
      value: 32.0
      comment: "thalweg: Great Belt; deficit=12.1 m (37.9%)"
```

Flat entries (b/s/a/lb prefixes) each have their own `applied` flag.
Thalweg entries are grouped by waypoint under a `tw_<slug>:` key; a single
`applied: true/false` controls the whole group.

**Two-run workflow:**
1. Full run → produces `fixes.yaml` with all entries `applied: false`
2. Open `fixes.yaml`, set `applied: true` on entries/groups you want
3. `--accept-fixes --skip-regrid` → applies only those marked true

Or: `--apply-all-fixes --skip-regrid` to apply everything at once.

**Traceability:** `update_fixes_yaml` never drops an existing entry.  Flat
entries with `applied: true` survive even when no longer suggested.  Thalweg
groups (including their sub-entries) are preserved verbatim when the step-4d
call writes the file without thalweg data (e.g. during `--skip-regrid`).

### Protecting fixed cells during Haney smoothing

When fixes are applied, every `set_depth` / `open_cell` cell is recorded in a
boolean `pin_mask`.  This mask is passed to `smooth_rx0()` as a lower-bound
constraint on the LP variables (correction ≥ 0 for pinned cells), so the LP
solver cannot shallow a fixed cell — it must instead deepen the neighbours to
satisfy rx0.  This guarantees the smoothed field satisfies rx0 ≤ target
everywhere, which is verified by an rx0 summary table printed just before "Done":

```
INFO     17:23:55  ── rx0 summary ──────────────────────────────────────────────────
INFO     17:23:55  depth (fixed raw)  │ 0.8312
INFO     17:23:55  depth_rx0_0p20     │ 0.2000  (target <= 0.20)  [OK]
INFO     17:23:55  ─────────────────────────────────────────────────────────────────
```

The raw field rx0 can be large (fixed cells have steep neighbours before
smoothing — that is what the LP resolves).  The smoothed field must show `[OK]`.

Logging format: `%(levelname)-8s %(asctime)s  %(message)s` with `datefmt="%H:%M:%S"`.

### lib/ vs cli/ convention (same as stats repo)

- `lib/` — importable modules only. No shebang, no argparse, no `__main__`.
- `cli/` — entry-point scripts. Always: shebang, `main()`, `if __name__ == '__main__': main()`.
- `cli/__init__.py` is absent here; lib/ is on sys.path via `pyproject.toml`
  `package-dir = {"" = "lib", "cli" = "cli"}`.

### Coordinate / grid conventions

- `SphericalGrid`: `lon_min/max`, `lat_min/max`, `dlon`, `dlat`, `rotation_deg`
- `CartesianGrid`: `x_min/max`, `y_min/max`, `dx`, `dy`, `crs`, `rotation_deg`
- `RotatedPoleGrid`: `pole_lon/lat` or `lon/lat_center`, `rlon/rlat_min/max`, `drot`, `axis_rotation`
- `SuperGrid`: `file` (path to supergrid NetCDF), optional `x_var`/`y_var` (auto-detected).
  Reads a `(2·ny+1) × (2·nx+1)` file (MOM6 `ocean_hgrid.nc` or pyGETM style).
  Extracts T-centres at `[1::2, 1::2]` and Q-corners at `[0::2, 0::2]` for ESMF.
  Also exposes `u_lon/u_lat` at `[1::2, 0::2]` and `v_lon/v_lat` at `[0::2, 1::2]`.
- Non-zero `rotation_deg` → corner arrays become 2-D; handled by ESMF and plotly.
- All Cartopy inset plots use `_inset_gridlines(ax, extent)` — auto-picks tick
  spacing (0.5° / 1° / 2° / 5° / 10°) from the extent span.

### Arakawa C-grid staggered depths (always written)

Every output NetCDF includes three depth variables:

| Variable  | Description | Shape |
|-----------|-------------|-------|
| `depth`   | T-point depth (NaN = land) | [ny, nx] |
| `depth_u` | Eastern-face depth = `min(depth[i,j], depth[i,j+1])` | [ny, nx] |
| `depth_v` | Northern-face depth = `min(depth[i,j], depth[i+1,j])` | [ny, nx] |

`depth_u[:, -1]` and `depth_v[-1, :]` use the boundary T-point depth.
NaN propagates: a face is land if either bordering T-cell is land.
`compute_cgrid_depth(depth_t)` in `lib/interpolate.py` performs the computation.

### Equidistant spherical grids (`--equidistant` / `grid.equidistant: true`)

Provide one of `dlat` or `dlon`; the other is computed as:

```
dlon = dlat / cos(lat_center)    # given dlat → wider lon spacing at high latitudes
dlat = dlon * cos(lat_center)    # given dlon → narrower lat spacing
```

where `lat_center = (lat_min + lat_max) / 2`.  This makes E-W and N-S physical
cell sizes approximately equal.  `nx = round((lon_max - lon_min) / dlon)` is
smaller than for a naive equal-degree grid — the startup message shows the
resulting dimensions so the user can confirm before the run proceeds.

### Report / plot helpers (lib/report.py)

- `plot_depth(ds, var, title, subtitle, log_scale=False)` — two-line title via
  `set_title("\n".join)`; do not use `ax.text` for subtitles (it overlaps the
  Cartopy title).  `log_scale=True` uses `matplotlib.colors.LogNorm`; the
  interactive plotly HTML stores `log10(depth)` with original-depth tick labels.
  Enabled via `output.log_depth_scale: true` in the YAML config.
- `update_fixes_yaml(records, path, bridge_records, thalweg_fixes)` — merges
  strait, bridge and thalweg fix suggestions into `fixes.yaml`.  Preserves
  existing `applied` flags (flat entries) and group-level `applied` flags
  (thalweg groups).  When called without `thalweg_fixes` (step 4d), existing
  thalweg groups are re-emitted verbatim so user edits are not lost.
- `mark_all_applied(path)` — sets `applied: true` for every entry; called
  after `--apply-all-fixes`.
- `_inset_gridlines(ax, extent)` — call this on all inset Cartopy axes.

### Thalweg depth profile plot (4 series)

Each thalweg panel shows:

| Series | Colour | Description |
|--------|--------|-------------|
| Fine | Solid blue | Fine-resolution depth along the thalweg path |
| Pre-fix raw coarse | Light coral dots | Coarse depth before any fixes (only when `--accept-fixes` / `--apply-all-fixes` was used) |
| Fixed raw coarse | Orange dots | Coarse depth after fixes, before Haney smoothing |
| `depth_rx0_*` coarse | Dashed green | Coarse depth after Haney smoothing (one series per rx0 variant) |

The dashed style and higher z-order (5) ensure the smoothed line is visible
even where it overlaps the orange dots (which occurs when re-pinning kept the
fixed depth unchanged through smoothing).

## Known issues / invariants

- xESMF weight files are cached in `regrid_weights/` — delete to force recompute.
- `tqdm` is **not** a dependency — progress bar was removed (strait detection is fast).
- Pyright reports false positives on `set_title`, `tight_layout`, `savefig` and
  `vmin`/`vmax` in `report.py` — these are pre-existing stubs issues, not real bugs.
