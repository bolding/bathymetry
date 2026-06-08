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

# Apply suggested strait fixes automatically
bathymetry-regrid --config example_northsea.yaml --accept-fixes
bathymetry-regrid --config example_northsea.yaml --fixes-file report/my/fixes_suggested.yaml

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
  report.py            Markdown report, ASCII tables, Cartopy + plotly plots
```

### Pipeline steps (cli/regrid.py)

| Step | What |
|------|------|
| 1 | Build target grid |
| 2 | Read + clip source bathymetry |
| 3 | xESMF conservative regrid (may take a minute) |
| 4a | Detect narrow straits → writes `fixes_suggested.yaml` sorted by cause |
| 4b | Apply user fixes (`fixes:` in config, `--accept-fixes`, `--fixes-file`) |
| 4c | Apply `mask_regions:` |
| 4d | Remove isolated ocean cells (flood-fill) |
| 5 | rx0 smoothing (optional) |
| 6 | Write NetCDF + final plots |

### Fixes workflow

After step 4a, `fixes_suggested.yaml` is written to the report directory.
Entries are grouped and sorted by category (BLOCKED → SILL_DEFICIT → AREA_DEFICIT).
On the next run, load them with `--accept-fixes` (reads from the default path)
or `--fixes-file <path>` (explicit). No manual copy-paste needed.

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
- `save_fixes_yaml(records, path)` — groups entries by category, writes one
  comment-block header per group so the user sees BLOCKED / SILL_DEFICIT /
  AREA_DEFICIT in priority order.
- `_inset_gridlines(ax, extent)` — call this on all inset Cartopy axes.

## Known issues / invariants

- xESMF weight files are cached in `regrid_weights/` — delete to force recompute.
- `fixes_suggested.yaml` `_note` keys are silently ignored by `apply_fixes()`;
  no need to strip them before passing via `--accept-fixes`.
- `tqdm` is **not** a dependency — progress bar was removed (strait detection is fast).
