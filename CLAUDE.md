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
| 3 | xESMF conservative regrid (may take a minute; result cached as `regrid_weights/{name}_raw_regrid.nc`).  `bbox_depth_percentile` post-pass applied here and baked into the cache. |
| 3b | Second source comparison (optional) |
| 4a | Apply user fixes (`fixes:` in config, `--accept-fixes`, `--apply-all-fixes`, `--fixes-file`) |
| 4b | Apply explicit `mask_regions:` (rectangle, polygon, point, ij_rectangle, ij_point) |
| 4c | Remove isolated ocean cells (flood-fill; keep *nkeep* largest basins, or explicit `keep_basins` list) |
| 4c-i | Phantom island detection — runs after basin removal so isolated open-water cells are already masked.  Ocean cells with `wet_fraction < phantom_island_max_wet_fraction` whose fine-grid land pixels belong only to small components (< `phantom_island_max_fine_cells`) are flagged.  Multi-cell clusters grouped by shared fine-grid component (Union-Find).  Results written as `phantom_islands` group in `fixes.yaml`. |
| 4c-ii | Detect LAND_BRIDGE cells (wet_fraction > 0 cells between disconnected basins) |
| 4c-iii | Boundary cross-section matching (optional; `nudge_boundaries:` in config) |
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
      # With bbox_depth_percentile: 75 on the waypoint, the comment would read:
      # "thalweg: Great Belt; deficit=12.1 m (37.9%) p75; fine max=45.0 m"
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
boolean `pin_mask`.  Cells deepened by `nudge_boundaries:` (step 4c-iii) are
OR-ed into the same mask as `nudge_pin_mask`.  The combined mask is passed to
`smooth_rx0()` as a lower-bound constraint on the LP variables (correction ≥ 0
for pinned cells), so the LP solver cannot shallow any fixed or nudged cell —
it must instead deepen the neighbours to satisfy rx0.  This guarantees the smoothed field satisfies rx0 ≤ target
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
  — or alternatively `center_lon`, `center_lat`, `x_size` (km), `y_size` (km) to
  specify the domain by its geographic centre and extent.  `cli/regrid.py` converts
  to `x_min/max/y_min/max` via pyproj before constructing the grid.
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
- `update_fixes_yaml(records, path, bridge_records, thalweg_fixes, phantom_island_fixes)` —
  merges strait, bridge, thalweg, and phantom-island fix suggestions into `fixes.yaml`.
  Preserves existing `applied` flags (flat entries) and group-level `applied` flags
  (thalweg / phantom_islands groups).  When called without `thalweg_fixes` (step 4d),
  existing thalweg groups are re-emitted verbatim so user edits are not lost.
- `mark_all_applied(path)` — sets `applied: true` for every entry; called
  after `--apply-all-fixes`.
- `_inset_gridlines(ax, extent)` — call this on all inset Cartopy axes.

### Phantom island detection (step 3b)

`detect_phantom_islands(dst, ...)` in `lib/analysis.py` finds ocean cells that
are mostly land in the fine-resolution source — typically small islands that the
conservative regrid kept as 2 m ocean cells instead of land.

**Algorithm (full run — fine source available):**
1. Candidate cells: ocean (mask=1) with `wet_fraction < max_wet_fraction` (default 0.5).
2. For each candidate, find the fine-grid connected land-component IDs in its footprint
   (via `scipy.ndimage.label` + KDTree bounding-box search).
3. Candidates that touch any component with ≥ `max_island_fine_cells` pixels are excluded
   — those pixels belong to the mainland or a large island, not a phantom.
   Candidates with zero fine land pixels are also excluded.
4. Remaining candidates are grouped by shared fine-grid component (Union-Find): two coarse
   cells are in the same cluster when they contain pixels from the same fine component.
5. Clusters exceeding `max_cluster_size` (default 5 with src) are discarded as a safety cap.

The coarse-grid neighbourhood check (`search_radius`) is **not used** in the full-run path —
it is the wrong discriminator for fjords where every ocean cell is near land at coarse resolution.

**Fallback (--skip-regrid, no fine source):**
1. Candidate cells as above.
2. Neighbourhood check: dilate coarse land mask by `search_radius`; exclude candidates
   within that radius (they are coastal cells, not isolated islands).
3. Coarse 4-connectivity grouping; `max_cluster_size` cap defaults to 1.

**`fixes.yaml` structure:**  All detected cells are written under a single
`phantom_islands:` group with one `applied: false/true` flag controlling the whole
group.  The `mask_cell` action (alias for `close_cell`) sets depth=NaN, mask=0.
The comment includes `wet_fraction`, depth, fine pixel count, and cluster size.

**Report and log:**  Each detected island is listed in the log with lon, lat,
wet_fraction, depth, and fine pixel count.  A "Phantom island detection" section
is written to the Markdown report (table of all flagged cells + warning if any found).
Phantom islands appear on the straits/connectivity plot (step 4d) as cyan diamonds —
both in the static PNG and the zoomable plotly HTML — alongside BLOCKED/SILL_DEFICIT
markers.  Hover text shows lon, lat, wet_fraction, depth, and fine cell count.

**Config keys** (under `analysis:`):`
| Key | Default | Meaning |
|-----|---------|---------|
| `phantom_island_max_wet_fraction` | 0.5 | Maximum wet_fraction to be a candidate |
| `phantom_island_max_fine_cells` | 1000 | Fine-grid component size threshold: larger = mainland, not island |
| `phantom_island_search_radius` | 2 | Coarse neighbourhood radius (fallback only, ignored when fine src available) |
| `phantom_island_max_cluster_size` | 1 | Max coarse-cell cluster size (auto-raised to 5 with fine src) |

### Thalweg analysis

#### Source data (no coastline mask)

Thalweg path-finding always uses **raw GEBCO** — without the NaturalEarth
coastline mask that is applied to the main pipeline source.  NE land polygons
clip narrow channel cells (e.g. Little Belt) as land, which disconnects the
max-bottleneck MST and causes a false "no wet path" failure.  When
`regridding.coastline_mask` is set, the thalweg source is reloaded from the
original file without the mask.

#### Auto-corridor detection (`auto_corridors`)

`detect_auto_corridors(dst, user_waypoints, ...)` in `lib/thalweg.py` finds
narrow straits automatically without requiring manual waypoints:

1. `_find_articulation_points_2d(mask)` — iterative Tarjan's DFS on the 4-connected
   coarse wet-cell graph.  Returns a boolean mask of articulation points (cells
   whose removal disconnects the graph).  Iterative to avoid Python recursion limit.
2. Nearby APs are merged via dilation + relabelling (controlled by `auto_corridor_merge_dist`).
3. For each corridor: remove the AP cells, flood-fill to find the two (or more)
   disconnected components; pick begin/end as the deepest cell within
   `auto_corridor_margin` cells of each component edge.
4. Components smaller than `auto_corridor_min_basin_cells` are skipped.
5. Corridors whose bbox overlaps an existing manual waypoint (or whose centroid
   is within a proximity threshold) are skipped to avoid duplicates.
6. Returns `list[dict]` with `{name, begin, end, bbox, auto: True}`.

Auto-corridors are appended to `user_waypoints_cfg` before calling
`waypoint_thalwegs()`, so they run through the identical pipeline as manual
waypoints.  The `auto: True` flag is for bookkeeping only.

#### `bbox_depth_percentile` (per-waypoint and global)

Set on a waypoint (or globally under `thalweg:`) to replace the conservative
area-average depth in coarse cells inside the bbox with the Nth percentile of
the fine-grid depths within each cell.  This is **bidirectional** — it can
deepen a cell that the area-average made too shallow (land-fraction dilution)
*and* shallow a cell that was pulled too deep by an isolated deep hole.

```yaml
thalweg:
  # bbox_depth_percentile: 75   # global fallback for all waypoints
  waypoints:
    - name: "Little Belt"
      bbox: [9.3, 10.2, 55.0, 55.7]
      bbox_depth_percentile: 75   # per-waypoint; overrides global
```

The result is baked into the raw-regrid cache (`regrid_weights/{name}_raw_regrid.nc`).
**Delete the cache and rerun without `--skip-regrid`** whenever you change this value.

#### Fix-value percentile

When `bbox_depth_percentile` is set on a waypoint, `suggest_depth_fixes` uses
the same percentile to compute the suggested fix *value* for each coarse cell
(instead of the maximum fine depth).  This prevents an isolated deep hole from
producing a fix that sets the coarse cell to an unrealistically deep value.

- The fix value becomes `np.percentile(fine_thalweg_depths_in_cell, pct)`.
- The comment in `fixes.yaml` shows both the percentile value and the actual
  maximum (`fine max=Xm`) so the clipping is visible.
- Deficit thresholds are evaluated against the percentile value, not the max.
- For cells visited by multiple thalwegs with different percentiles the most
  conservative (lowest) percentile is used.

#### Coastline resolution for all depth plots

`output.coastline_scale: "10m"` (default) controls the coastline dataset used in
**all** depth, basin, strait, thalweg, diff, and source-comparison plots.

NaturalEarth values: `"10m"` | `"50m"` | `"110m"`.
GSHHG (auto-downloaded via Cartopy): `"gshhg-f"` | `"gshhg-h"` | `"gshhg-i"` | `"gshhg-l"` | `"gshhg-c"`
(full / high / intermediate / low / coarse).  GSHHG is significantly finer than
NaturalEarth and recommended for high-resolution regional or fjord domains where
NE 10m coastline is too coarse relative to the grid resolution.
Use `"none"` to suppress all coastline/land features on every plot.

`output.final_coastline: true` (default) — set `false` to suppress coastline on the
step-6 final depth plots only (all diagnostic plots keep the `coastline_scale`).
When `false`, NaN (land) cells render as solid white and the colormap `set_bad`
colour is set to white to avoid black edge artifacts at the NaN/ocean boundary.

The `_add_land_feature(ax, scale, ...)` helper in `lib/report.py` dispatches on the
`"gshhg-"` prefix; `fill_land=False` draws only the coastline outline (used for
diff/source-comparison/basin plots where the pcolormesh already colours the domain).
`scale="none"` sets `ax.set_facecolor("white")` and returns immediately.

#### Failed-thalweg diagnostic plot

When a waypoint thalweg fails (no wet MST path), `_plot_thalweg_failed` writes
a PNG showing the **fine GEBCO depth** as background (not the coarse grid), the
waypoint bbox, and the start/end/via markers.  The Cartopy LAND feature is at
`zorder=1` (behind the fine depth overlay) so it does not obscure the channel.
The colorbar uses the local bbox depth range, not the global domain max.

#### Depth profile plot (4 series)

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

### Boundary cross-section matching (`nudge_boundaries:`)

Nudges inner-grid depths near open boundaries so the cross-sectional area of
each outer-model boundary cell is preserved.  Applied at step 4c-iii — after
basin removal, before strait detection.

```yaml
nudge_boundaries:
  enabled: true          # set false to keep config but skip the step (default true)
  outer_file: /path/to/outer_bathymetry.nc   # required when enabled
  boundaries_file: northsea_1d15deg_bdy.csv  # *_bdy.csv from --write-boundaries
  boundaries: [N, S, E, W]   # fallback sides when no file; default all four
  width: 10                  # taper width in cells
  shape: cosine              # cosine | linear | exponential
  min_depth: 2.0             # floor for nudged depth (m)
  max_scale: 3.0             # clamp scale factor to [1/max_scale, max_scale]
  outer_depth_var: depth     # variable name; auto-detected if omitted
```

**Boundary cell identification** (both can be used simultaneously):
- `boundaries_file`: reads the `*_bdy.csv` produced by `--write-boundaries`
  (two-row header "T-grid" / "lon,lat", then one row per wet boundary cell).
  Each lon/lat is snapped to the nearest inner-grid cell via KDTree.  Only
  those cells are treated as open boundaries — useful when not all four edges
  carry an open boundary condition.
- `boundaries`: scans every wet cell on the listed grid edges (N/S/E/W).
  This is the default when no `boundaries_file` is given.

**Algorithm:**
1. Load outer model NetCDF; auto-detect depth variable and lon/lat coordinates.
2. Build KDTree over outer wet cells; query nearest outer cell for each inner
   boundary cell.
3. For each group of inner cells that map to the same outer cell, compute
   `scale = A_outer / A_inner` where `A = depth × cell_width_deg_coslat`.
   Scale is clamped to `[1/max_scale, max_scale]`.
4. BFS inland from boundary cells to propagate each scale factor over
   `width` cells.
5. Apply: `depth_nudged = depth_in × (w × scale + (1 − w))` where `w` is
   the taper weight (1 at boundary, 0 at `width` cells inland).
6. Cells that are **deepened** are added to `soft_pin_mask` so the Haney LP
   cannot shallow them — it must deepen their neighbours instead.

`apply_boundary_crosssection_match()` in `lib/interpolate.py` implements steps 1–6
and returns `(depth_nudged, soft_pin_mask)`.

## Known issues / invariants

- xESMF weight files are cached in `regrid_weights/` — delete to force recompute.
- Raw-regrid result cached as `regrid_weights/{name}_raw_regrid.nc` (includes
  `bbox_depth_percentile` post-pass).  Delete it when changing `bbox_depth_percentile`
  or the grid spec; `--skip-regrid` will error with a shape-mismatch message if stale.
- `tqdm` is **not** a dependency — progress bar was removed (strait detection is fast).
- Pyright reports false positives on `set_title`, `tight_layout`, `savefig` and
  `vmin`/`vmax` in `report.py`, and on `lib/` imports in `cli/regrid.py` — these are
  pre-existing stubs / path issues, not real bugs.
