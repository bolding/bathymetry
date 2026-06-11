# bathymetry

Conservative interpolation of fine-resolution bathymetry (GEBCO, EMODnet) onto
coarser model grids, with automated diagnostics for narrow straits and isolated
ocean cells.

## Motivation

Simple averaging of a fine bathymetry onto a coarser grid destroys the volume
and cross-sectional area of narrow straits.  This is particularly critical for:

- **Transport capacity** — the cross-sectional area (width × depth) controls
  volume exchange between adjacent basins.
- **Dense bottom-water overflows** — the *sill depth* (deepest point of the
  narrowest cross-section) determines whether dense water can spill across a
  ridge (e.g. Denmark Strait, Faroe Bank Channel).

This tool uses xESMF first-order conservative regridding with FRACAREA
normalisation, followed by automated diagnostics that flag both geometric
narrowing and sill-depth under-representation.

## Installation

The project requires `esmpy` / `xesmf` and optionally `rioxarray` (for
EMODnet), which are easiest to install via conda.  The remaining pure-Python
dependencies are installed by pip.

```bash
# Inside the stats (or any) conda environment that already has esmpy + xesmf:
cd /path/to/bathymetry
pip install -e .
```

To verify:

```bash
bathymetry-regrid --help
```

### Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| xarray, netCDF4 | NetCDF I/O | pip |
| numpy, scipy | Numerics, LP solver | pip |
| matplotlib, cmocean | Plots and colormaps | pip / conda |
| cartopy | Geographic map projections | conda |
| pyproj | CRS transforms (Cartesian grids) | pip / conda |
| xesmf ≥ 0.9 | Conservative regridding (wraps ESMF) | conda |
| esmpy ≥ 8.9 | ESMF backend (pulled in by xesmf) | conda |
| plotly | Interactive / zoomable HTML figures | pip |
| pyyaml | YAML configuration files | pip |
| rioxarray | EMODnet GeoTIFF reading (optional) | conda / pip |

## Project layout

```
bathymetry/
├── pyproject.toml
├── README.md
├── config/
│   ├── northsea_1d15deg.yaml      ← North Sea 1/15° operational config
│   ├── northsea_1d20deg.yaml      ← North Sea 1/20° operational config
│   ├── example_northsea_rotated_pole.yaml
│   └── example_supergrid.yaml
├── lib/                           ← flat library modules (on sys.path after install)
│   ├── grid.py        SphericalGrid, CartesianGrid, RotatedPoleGrid, SuperGrid
│   ├── reader.py      GEBCO (local + auto-download) and EMODnet readers
│   ├── interpolate.py xESMF conservative regridding with weight-file caching
│   ├── analysis.py    strait detection, mask regions, isolated-cell masking
│   ├── smooth.py      rx0 slope-factor smoothing (LP, ported from GETM)
│   ├── boundary.py    open-boundary T-grid coordinate CSV writer
│   ├── thalweg.py     fine-vs-coarse thalweg extraction and comparison
│   └── report.py      Markdown report, ASCII tables, cartopy + plotly plots
└── cli/
    └── regrid.py      CLI entry point (bathymetry-regrid)
```

## Quick start

### From a YAML config (recommended)

```bash
bathymetry-regrid --config config/northsea_1d20deg.yaml
```

Override any setting on the command line:

```bash
bathymetry-regrid --config config/northsea_1d20deg.yaml --smooth-rx0 0.15 --name northsea_v2
```

### Spherical grid — North Sea, command-line only

```bash
bathymetry-regrid \
    --source /server/data/GEBCO/GEBCO_2023.nc \
    --name northsea_0p05deg \
    --grid spherical \
    --lon-min 0 --lon-max 15 --lat-min 50 --lat-max 60 \
    --dlat 0.05 \
    --min-depth 2 \
    --smooth-rx0 0.2 \
    --output northsea.nc \
    --report-dir ./report/northsea/
```

By default (`equidistant=true`) `dlon` is derived from `dlat` and the central
latitude so that cells are approximately square in physical distance.  At a
central latitude of 55 °N, `cos(55°) ≈ 0.574`, so `dlon ≈ 0.05 / 0.574 ≈ 0.0872°`.
The computed values are printed at startup:

```
[equidistant] lat_center=55.00°  dlon=0.087126°  dlat=0.05°  → grid 172 × 200 (lon × lat)
```

To use an explicit equal-degree grid instead (cells narrowing towards the poles),
supply both spacings and disable equidistant:

```bash
bathymetry-regrid ... --dlon 0.05 --dlat 0.05 --no-equidistant
```

In YAML (equidistant is the default — just omit `dlon`):

```yaml
grid:
  type: spherical
  lon_min: 0.0
  lon_max: 15.0
  lat_min: 50.0
  lat_max: 60.0
  dlat: 0.05          # dlon computed automatically from dlat / cos(lat_center)
  # dlon: 0.05        # uncomment + equidistant: false for an explicit equal-degree grid
```

### Rotated spherical grid

```bash
bathymetry-regrid --config my_run.yaml --rotation 30
```

### Cartesian grid — UTM zone 32N from EMODnet

```bash
bathymetry-regrid \
    --source emodnet \
    --name baltic_utm_1km \
    --grid cartesian \
    --x-min 400000 --x-max 900000 --y-min 6000000 --y-max 6500000 \
    --dx 1000 --dy 1000 --crs EPSG:32632 \
    --min-depth 1
```

## YAML configuration

All parameters can be set in a YAML file and selectively overridden on the CLI.
Fully annotated configs are provided in the `config/` directory.

```yaml
# Short identifier — used in all output file and directory names.
# E.g. name: northsea_0p05deg  →  northsea_0p05deg.nc,
#                                   report/northsea_0p05deg/northsea_0p05deg_*.png
name: northsea_0p05deg

# "emodnet"   → download on-the-fly from EMODnet WCS
# "gebco"     → download GEBCO 2025 ZIP from CEDA, cache at ~/.cache/gebco/gebco_2025.nc
# "/path/..."  → local GEBCO-style NetCDF
source: /server/data/GEBCO/GEBCO_2023.nc

# Optional second source — produces a side-by-side comparison map and depth-diff plot
# source2: /server/data/EMODnet/emodnet_2024.nc

grid:
  type: spherical        # spherical | cartesian | rotated_pole
  lon_min: 0.0
  lon_max: 15.0
  lat_min: 50.0
  lat_max: 60.0
  dlat: 0.05             # dlon computed automatically (equidistant=true is the default)
  # dlon: 0.05           # set dlon explicitly only when equidistant: false
  rotation: 0.0          # degrees CCW (non-zero → rotated curvilinear grid)
  # interfaces: false    # true → lon/lat bounds are cell corners (interfaces)
  #                      # false (default) → T-point positions; lon_max/lat_max
  #                      #   are the last T-points, so the grid includes them exactly.
  # equidistant: false   # uncomment to supply both dlon and dlat explicitly

regridding:
  cache_dir: ./regrid_weights   # xESMF weight files cached here
  min_depth: 2.0                # shallow-clamp after regridding (m)
  min_wet_fraction: 0.05
  coastline_mask: "10m"         # overlay NE land polygons on source; omit to skip
  # pad_deg: 1.0                # extra source margin beyond grid extent (expert)

analysis:
  nkeep_basins: 1          # keep the N largest basins (default 1 = largest only)
  # keep_basins: [1]       # alternatively: explicit list of size-rank numbers to keep
  #                        # e.g. [5] keeps only the 5th largest; [1, 3] keeps 1st and 3rd
  #                        # basins are numbered in the 04c_basins.png plot
  wet_frac_threshold: 0.3
  sill_ratio_threshold: 0.7
  area_ratio_threshold: 0.5

smooth:
  rx0: 0.2               # Haney slope-factor target; omit / null to skip

output:
  file: northsea_0p05deg.nc
  report_dir: ./report/northsea_0p05deg

# After the first run, fixes.yaml is written to the report directory.
# Set applied: true on entries you want, then re-run with --accept-fixes.
# Use --apply-all-fixes to accept everything at once.
# To hand-pick individual fixes, embed them here directly.
# Actions: set_depth, open_cell, close_cell
# fixes:
#   - lon: 5.3
#     lat: 55.7
#     action: set_depth
#     value: 22.0

# Thalweg analysis (step 4e) — compare fine-vs-coarse depth profiles.
# Enabled automatically when this section is present.
# thalweg:
#   boundary_thalwegs: false      # Mode B: auto-start from domain edges
#   bbox_depth_percentile: 75     # global percentile for all waypoint bboxes
#   auto_corridors: true          # Mode D: auto-detect narrow chokepoints
#   auto_corridor_min_basin_cells: 20  # ignore basins smaller than this
#   auto_corridor_margin: 3       # search radius for begin/end (cells)
#   auto_corridor_merge_dist: 2   # merge nearby chokepoints
#   waypoints:
#     - name: "Great Belt"
#       begin: [10.2, 55.3]
#       end:   [11.0, 55.9]
#       bbox:  [10.0, 12.0, 54.5, 56.0]
#       bbox_depth_percentile: 75   # per-waypoint override

# Nudge inner depths near open boundaries toward an outer (coarser) model
# to preserve cross-sectional area at the nesting interface (step 4c-iii).
# nudge_boundaries:
#   enabled: true                    # set false to keep config but skip (default true)
#   outer_file: /path/to/outer_model_bathymetry.nc   # required when enabled
#   boundaries_file: northsea_1d15deg_bdy.csv  # *_bdy.csv from --write-boundaries
#   boundaries: [N, S, E, W]  # fallback sides; default all four if no file
#   width: 10                  # taper width in cells
#   shape: cosine              # cosine | linear | exponential
#   min_depth: 2.0             # floor for nudged depth (m)
#   max_scale: 3.0             # clamp scale to [1/max_scale, max_scale]
#   outer_depth_var: depth     # auto-detected if omitted

# Force geographic areas to land (step 4c, before isolation masking).
# Masking a fjord mouth causes its interior to be removed automatically
# by the isolation step (4d).
# Types: rectangle, polygon, point, ij_rectangle, ij_point.
# Grid-index convention: i = x/longitude column, j = y/latitude row (as in ncview).
# mask_regions:
#   - type: rectangle       # geographic bounding box
#     lon_min: 9.0  lon_max: 10.5  lat_min: 54.5  lat_max: 55.5
#     name: Kiel Bight
#   - type: polygon         # arbitrary geographic polygon
#     vertices: [[10,55],[11,55.5],[11.5,54.5],[10,54]]
#     name: Custom bay
#   - type: point           # nearest cell to given lon/lat
#     lon: 8.5  lat: 54.8
#     name: Isolated pool
#   - type: ij_rectangle    # 0-based grid indices (i=lon-column, j=lat-row)
#     i_min: 10  i_max: 20  j_min: 5  j_max: 15
#     name: Index-based region
#   - type: ij_point        # single cell by 0-based grid index
#     i: 15  j: 8
#     name: Single index cell
```

## Pipeline steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `grid` | Build target grid from parameters |
| 2 | `reader` | Read and clip fine-resolution source bathymetry |
| 3 | `interpolate` | xESMF conservative regrid; `bbox_depth_percentile` post-pass; result cached as `regrid_weights/{name}_raw_regrid.nc` |
| 4a | `analysis` | Apply user fixes from `fixes:` (set_depth, open_cell, close_cell) |
| 4b | `analysis` | Apply explicit `mask_regions:` (rectangle, polygon, point, ij_rectangle, ij_point) |
| 4c | `analysis` | Remove isolated ocean cells (flood-fill; keeps *nkeep* largest basins, or explicit `keep_basins` list) |
| 4c-ii | `analysis` | Detect LAND_BRIDGE cells (forced-land with wet_fraction > 0 between disconnected basins) |
| 4c-iii | `interpolate` | Boundary cross-section matching (optional; `nudge_boundaries:` in config) |
| 4d | `analysis` | Flag narrow / blocked interfaces (BLOCKED, SILL_DEFICIT, AREA_DEFICIT) |
| 4e | `thalweg` | Compare fine-vs-coarse thalweg depth profiles (optional; `--thalweg`) |
| 5 | `smooth` | rx0 slope smoothing via linear programming (optional) |
| 6 | — | Write output NetCDF + final plots for every depth variable |

**Why 4b before 4c?**  
Masking the mouth of a fjord or lagoon (4b) makes the interior cells
disconnected from the main ocean.  Running isolation masking afterwards (4c)
then removes those interior cells automatically without requiring them to be
listed individually.

## Coastline masking

GEBCO's own land mask is derived from its depth values: any cell with
non-negative elevation is considered land.  For very shallow near-coastal
cells this can be ambiguous.  Setting `regridding.coastline_mask` overlays a
**Natural Earth** land-polygon dataset on the raw bathymetry source *before*
regridding, forcing any source cell whose centre falls inside a land polygon
to land (`depth=NaN`, `land=True`).

```yaml
regridding:
  coastline_mask: "10m"   # "10m" | "50m" | "110m"
```

| Resolution | Scale | Typical feature size | Notes |
|------------|-------|---------------------|-------|
| `"10m"` | 1:10 000 000 | ~1 km | Recommended default |
| `"50m"` | 1:50 000 000 | ~5 km | Faster, less detail |
| `"110m"` | 1:110 000 000 | ~10 km | Coarse overview only |

The shapefiles are downloaded automatically by cartopy on first use and cached
in `~/.local/share/cartopy/`.  The step requires **rasterio** (`pip install
rasterio`).

Omit the key (or set it to `null`) to skip coastline masking and rely solely
on GEBCO's own land flag.

> **When to omit `coastline_mask`:** GEBCO and EMODnet already carry a native
> land flag at full source resolution.  For regional high-resolution domains
> (e.g. 250 m fjord grids) the NE 10m polygons are *coarser* than the source
> data and will incorrectly mask valid coastal ocean cells, leaving white gaps
> in the regridded result.  Leave the key commented out unless you have a
> specific reason to believe the raw source land flag is unreliable for your
> domain.

> **Thalweg analysis always uses raw GEBCO** (no coastline mask), even when
> `coastline_mask` is set.  NE land polygons clip narrow channel cells (e.g.
> Little Belt) as land, which disconnects the max-bottleneck MST used for path
> finding.  When `coastline_mask` is configured the thalweg source is reloaded
> from the original file without the mask.

## GEBCO auto-download

Set `source: gebco` (or `source: gebco2025`) to have the tool download
**GEBCO 2025** automatically from CEDA and cache it locally:

```yaml
source: gebco
```

On first use the ~10 GB ZIP is downloaded from:

```
https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/gebco_2025.zip
```

The extracted NetCDF is cached at `~/.cache/gebco/gebco_2025.nc`.
Subsequent runs use the cache directly and skip the download.  The ZIP is
deleted after extraction to free disk space.

To use a specific local GEBCO file (any year), supply the path directly:

```yaml
source: /server/data/GEBCO/GEBCO_2024.nc
```

## Grid coordinate convention (`interfaces`)

By default (`interfaces: false`), `lon_min`/`lat_min`/`lon_max`/`lat_max` are
**T-point (cell-centre) positions**.  The first T-point is at `lon_min`; the
last T-point is at `lon_max`; cell corners are offset outward by ±½ cell on
every side.  The number of cells is `round((lon_max − lon_min) / dlon) + 1`,
so both endpoints are included exactly.

When `interfaces: true` (or `--interfaces`), the four bounds are the
**outermost corner (interface) positions**.  T-points are then inset by
½ cell and the number of cells is `round((lon_max − lon_min) / dlon)`.

| Mode | `lon_min` / `lon_max` | T-point range | nx |
|------|----------------------|---------------|----|
| T-points (default) | first / last T-point | `lon_min` … `lon_max` | `Δlon / dlon + 1` |
| Interfaces (`--interfaces`) | west / east corner | `lon_min + dlon/2` … `lon_max − dlon/2` | `Δlon / dlon` |

## Wet-fraction thresholds

There are two separate wet-fraction parameters with different roles:

| Parameter | Applied at | Effect |
|-----------|-----------|--------|
| `regridding.min_wet_fraction` | post-regrid (step 3) | Cells below threshold are forced to land: `depth=NaN`, `mask=0`. They are eliminated entirely and never reach the analysis steps. |
| `analysis.wet_frac_threshold` | strait detection (step 4a) | Cells below threshold are flagged for inspection but kept wet. Default 0.3. |

**Which one to use for the Wadden Sea / tidal flat problem:**  
Tidal flat cells at the margins of a domain typically have very small wet
fractions (the fine-grid source is mostly land).  These should simply be land
on the coarse grid.  Raise `regridding.min_wet_fraction` (e.g. 0.05–0.4
depending on resolution) to eliminate them before any analysis is done.
Because weight files are cached, re-running from step 3 is fast — only the
regridded arrays need to be recomputed, not the ESMF weights.

## Output files

All files are prefixed with `name` and written to `report_dir`.

### NetCDF (`{name}.nc`)

| Variable | Description |
|----------|-------------|
| `depth` | Regridded bathymetry, positive-down (m) |
| `depth_rx0_0p20` | Smoothed bathymetry for rx0≤0.2 (one variable per rx0 target) |
| `wet_fraction` | Fraction of each coarse cell covered by ocean in the source grid |
| `mask` | Ocean mask: 1 = ocean, 0 = land |
| `basin_labels` | Connected-component basin IDs (from step 4d) |
| `depth_corrections` | Per-cell corrections applied by rx0 smoothing |

Multiple smoothed bathymetries can coexist in one file — each run with a
different `rx0` value adds a new `depth_rx0_*` variable.

### Report (`report_dir/`)

| File | Contents |
|------|----------|
| `{name}_report.md` | Markdown report linking all tables and figures |
| `{name}_02a_source_raw.png` | Source bathymetry overview map |
| `{name}_02b_source_coastline_masked.png` | Source after coastline masking (if used) |
| `{name}_03_regrid_result.png` | Regridded depth (cmocean *deep_r*) |
| `{name}_03_wet_fraction.png` | Wet fraction per coarse cell (cmocean *amp*) |
| `{name}_03b_source_comparison.png` | Side-by-side source comparison (if `source2:` given) |
| `{name}_03b_depth_diff.png` | Depth difference `source − source2` on common cells |
| `{name}_04b_mask_regions.png` | Depth after explicit masking (if used) |
| `{name}_04c_basins.png` | Connected basin map (kept=blue, removed=red) |
| `{name}_04c_land_bridges.csv` | Land-bridge cells (if any found) |
| `{name}_04d_straits.png/.html` | Strait analysis map (interactive) |
| `{name}_04d_straits.csv` | Flagged interface table |
| `{name}_04d_section_*.png` | Cross-section profiles with map inset |
| `fixes.yaml` | Suggested fixes grouped by cause; set `applied: true` and re-run with `--accept-fixes` |
| `{name}_04e_thalweg_NNN_*.png` | Thalweg comparison panels (one per thalweg; step 4e) |
| `{name}_05_smooth_*.png` | rx0 histogram + depth-correction map |
| `{name}_06_final_depth.png/.html` | Final unsmoothed bathymetry (interactive) |
| `{name}_06_final_depth_rx0_*.png/.html` | Final smoothed bathymetry (interactive) |

### Boundary coordinates (`{name}_bdy.csv`)

Written alongside the output NetCDF when `--write-boundaries` is given.
Contains all wet cells on the four outer edges in the order west (S→N),
north (W→E), east (S→N), south (W→E).  Format:

```
T-grid
lon,lat
6.00000,56.25000
6.00000,56.26000
…
```

The report Markdown includes a segment table showing the start/stop `(i,j)`
indices and cell count for each contiguous wet segment per side.

## Thalweg analysis

A *thalweg* is the line connecting the deepest points along a channel.
Comparing fine-resolution and coarse-resolution thalwegs reveals how much
depth information is lost after regridding — in particular how much the sill
depth (the shallowest point along the deepest route) is under-represented.

Thalweg analysis runs as step 4e after strait detection.  It is **off by
default** and is enabled with `--thalweg` on the command line.  If your
config file contains a `thalwegs:` section the step is enabled automatically.

```bash
bathymetry-regrid --config my_run.yaml --skip-regrid --thalweg
bathymetry-regrid --config my_run.yaml --skip-regrid --no-thalweg   # force off
```

### Three detection modes

All three modes run in the same pass and their results are combined.

#### Mode A — Strait-based

For every strait interface flagged in step 4d, the fine-resolution thalweg is
extracted through the strait window (along-channel depth profile, deepest wet
cell per cross-section) and the coarse-resolution depth is sampled at the same
geographic positions via a KD-tree nearest-neighbour lookup.  Up to
`analysis.max_section_profiles` straits are processed (default 10).

#### Mode B — Boundary auto-detection

The deepest wet cell on each domain edge provides a natural start point.  If
an island intersects a boundary, the edge is split into several disconnected
wet segments — **each segment produces its own independent start cell**.  The
max-bottleneck Dijkstra algorithm then finds the route between every cross-edge
pair of starts that keeps the minimum depth along the path as large as
possible.  This is the true deepest route through the domain, not simply the
shortest path.

Paths whose fine-resolution sill is shallower than 5 m are discarded.

#### Mode C — User waypoints

Specify one or more named start→end pairs under the `thalweg.waypoints` key:

```yaml
thalweg:
  boundary_thalwegs: false   # set true to enable Mode B auto-detection
  # bbox_depth_percentile: 75  # global fallback — see below
  waypoints:
    - name: "Great Belt"
      begin: [10.2, 55.3]
      end:   [11.0, 55.9]
      bbox:  [10.0, 12.0, 54.5, 56.0]   # bounding box: restricts MST + fix suggestions
    - name: "Little Belt"
      begin: [9.5, 55.0]
      via:
        - [9.8, 55.5]   # force path through the narrow strait
      end: [10.5, 56.5]
      bbox: [9.3, 10.2, 55.0, 55.7]
      bbox_depth_percentile: 75   # per-waypoint; see below
    - name: "Öresund"
      begin: [12.6, 55.4]
      end:   [12.9, 56.1]
```

Each endpoint is snapped to the nearest wet fine-grid cell inside the `bbox`.
The max-bottleneck MST algorithm then finds the deepest route between the
points.  Optional `via` points force the path through a specific location,
useful when the deepest detour would bypass a narrow strait entirely.
The presence of any `thalweg:` section in the config enables step 4e automatically.

#### Mode D — Automatic chokepoint detection (`auto_corridors`)

Narrow straits that you haven't explicitly listed can be found automatically
by searching for *articulation points* — wet coarse cells whose removal would
split the ocean domain into two or more disconnected components.  Clusters of
nearby articulation points form a *corridor*; begin/end waypoints are chosen as
the deepest wet cell within `auto_corridor_margin` cells of each disconnected
component on either side.

```yaml
thalweg:
  auto_corridors: true
  auto_corridor_min_basin_cells: 20   # ignore components smaller than N cells
  auto_corridor_min_length: 1         # min articulation-point cells per corridor
  auto_corridor_margin: 3             # bbox padding + begin/end search radius (cells)
  auto_corridor_merge_dist: 2         # merge corridors within this many cells
```

Auto-detected corridors are appended to any user-specified waypoints and run
through the same thalweg pipeline.  Any corridor whose bbox overlaps an
existing manual waypoint bbox (or is within a proximity threshold of the
waypoint's centroid) is skipped to avoid duplication.

#### `bbox_depth_percentile` — correcting coarse cell depths

Conservative area-averaging can both under- and over-represent a channel:

- **Too shallow**: when a coarse cell straddles land and water, the land area
  dilutes the average.
- **Too deep**: when an isolated deep hole (scour pit) dominates a coarse cell,
  the average is pulled unrealistically deep.

Setting `bbox_depth_percentile` replaces the area-average in coarse cells
inside the waypoint bbox with the Nth percentile of fine-grid depths within
each cell.  It works **bidirectionally** — it can deepen or shallow.  50th
percentile (median) is most robust against outliers; 75th retains the deeper
parts of a channel while ignoring isolated pits.

```yaml
waypoints:
  - name: "Little Belt"
    bbox: [9.3, 10.2, 55.0, 55.7]
    bbox_depth_percentile: 75   # per-waypoint
```

Or set a global fallback for all waypoints at the `thalweg:` level:

```yaml
thalweg:
  bbox_depth_percentile: 75   # applies to every waypoint that has a bbox
```

**The result is baked into the raw-regrid cache.**  If you change this value,
delete `regrid_weights/{name}_raw_regrid.nc` and rerun without `--skip-regrid`.

The same percentile also controls the suggested fix *value* in `fixes.yaml` for
that waypoint — preventing an isolated deep hole from inflating a fix to an
unrealistic depth.  When a percentile below 100 is used the comment in
`fixes.yaml` shows both the percentile value and the actual cell maximum:

```yaml
"001":
  value: 28.5
  comment: "thalweg: Little Belt; deficit=8.5 m (29.8%) p75; fine max=64.0 m"
```

#### Coastline resolution for all depth plots

`output.coastline_scale` controls the coastline dataset used in **all** depth,
basin, strait, thalweg, diff, and source-comparison plots (default `"10m"`):

```yaml
output:
  coastline_scale: "10m"       # NaturalEarth: "10m" | "50m" | "110m"
  # coastline_scale: "gshhg-h" # GSHHG high-res — recommended for fjord / high-res grids
  #                             # GSHHG scales: "gshhg-f" | "gshhg-h" | "gshhg-i" | "gshhg-l" | "gshhg-c"
  # coastline_scale: "none"    # no coastline or land fill on any plot
  final_coastline: true        # set false to suppress coastline on step-6 final plots only
```

GSHHG (Global Self-consistent Hierarchical High-resolution Geography) is auto-downloaded
by Cartopy and is significantly finer than NaturalEarth — use `"gshhg-h"` or `"gshhg-f"`
for grids where NE 10m coastlines are too coarse (e.g. 250 m fjord grids).

Set `coastline_scale: "none"` to show only the grid cells with no land/coastline overlay
on any plot.  `final_coastline: false` applies the same suppression to the step-6 final
depth plots only, while keeping coastlines on all diagnostic plots.

### Output

Each thalweg produces a two-panel figure (`04e_thalweg_NNN_<name>.png`):

- **Left panel — map**: fine-resolution depth as a background pcolormesh,
  the thalweg path overlaid in blue, coarse-resolution depths as a coloured
  scatter, and a triangle marker at the sill position.
- **Right panel — depth profile**: four series plotted against along-path
  distance (km) with an inverted y-axis:

  | Series | Style | When shown |
  |--------|-------|-----------|
  | Fine | Solid blue line | Always |
  | Pre-fix raw coarse | Light coral dots | Only when fixes were applied in this run |
  | Fixed raw coarse (or Raw coarse) | Orange dots | Always; labelled "Fixed" when fixes were applied |
  | `depth_rx0_*` coarse | Dashed green line | When rx0 smoothing is enabled |

  Dashed horizontal lines mark the fine and coarse sill depths.

The thalweg name in the figure title and report section encodes the detection
mode:

| Example name | Origin |
|---|---|
| `west→east` | Mode B, simple edge pair |
| `south[1]→north[0]` | Mode B, island on boundary (segment indices shown) |
| `Little Belt` | Mode C, user-specified waypoint |
| Strait category + lat/lon | Mode A, derived from strait record |

## Strait categories and land bridges

The pipeline identifies four categories of connectivity problems and writes
every suggested fix to `fixes.yaml` in the report directory.

| Category | Detected at step | Meaning | Suggested fix |
|----------|-----------------|---------|--------------|
| `BLOCKED` | 4d — strait detection | No wet fine-resolution path between two wet coarse cells | `open_cell` on the blocking point |
| `SILL_DEFICIT` | 4d — strait detection | Coarse sill shallower than fine-grid sill — dense overflow blocked | `set_depth` to fine-grid sill depth |
| `AREA_DEFICIT` | 4d — strait detection | Cross-sectional area under-represented — transport too weak | Widen or deepen at the interface |
| `LAND_BRIDGE` | 4c-ii — bridge detection | Land cell (wet_fraction > 0, forced below `min_wet_fraction`) sits between two disconnected wet basins | `open_cell` on the bridge cell |

`LAND_BRIDGE` cells appear only when `regridding.min_wet_fraction > 0`.  They
are coarse-grid cells that contain some ocean in the fine-resolution source but
were eliminated because the ocean fraction was below the threshold.  Raising the
threshold to eliminate tidal flats may inadvertently block a narrow strait like
the Little Belt.  Lowering `min_wet_fraction` on just those cells is not
possible — instead, use `open_cell` fixes to reopen individual bridge cells
after the run.

## Fixing workflow

The regridding step (step 3) is the most expensive part of the pipeline.
After the first run the raw post-regrid result is cached as
`regrid_weights/{name}_raw_regrid.nc`.  The typical iteration loop is:

```
First run (full)
  ↓  inspect  fixes.yaml  +  04d_straits.csv  +  04c_land_bridges.csv
  ↓  choose which fixes to apply
Re-run with --skip-regrid (fast — no regrid)
  ↓  inspect result
Re-run again until satisfied
```

### Iterating on `bbox_depth_percentile` and thalweg fixes

`bbox_depth_percentile` is baked into the regrid cache at step 3, so changing
it requires rebuilding the cache.  The typical workflow:

```bash
# 1. Edit the config — add or change bbox_depth_percentile on a waypoint:
#      bbox_depth_percentile: 75
#    (or 50 for median, 90 to only clip the very deepest outliers)

# 2. Delete only the raw-regrid cache (weight files are reused):
rm regrid_weights/northsea_1d15deg_raw_regrid.nc

# 3. Full rerun — rebuilds cache with new percentile, regenerates fixes.yaml
#    (do NOT use --skip-regrid here)
bathymetry-regrid --config config/northsea_1d15deg.yaml

# 4. Inspect the new fixes.yaml and depth profiles.
#    If the fix values look right, accept them:
bathymetry-regrid --config config/northsea_1d15deg.yaml \
    --skip-regrid --accept-fixes
```

When you change `bbox_depth_percentile`, also **delete `fixes.yaml`** (or at
least the affected `tw_` group) before the full rerun — otherwise the old fix
values with `applied: true` are re-applied on top of the new percentile-adjusted
cache before `fixes.yaml` is rewritten with fresh suggestions.

If only the thalweg waypoint geometry changes (begin/end/via/bbox) but not the
percentile, `--skip-regrid` is sufficient — the cache is still valid.

### `fixes.yaml` format

After the first run `fixes.yaml` is written to the report directory.
It uses a **dict** format with one entry per fix, each with an `applied` flag:

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

Flat entries (`b`, `s`, `a`, `lb` prefixes) each have their own `applied` flag.
Thalweg fixes are grouped by waypoint name under a `tw_<slug>:` key with a
single `applied` flag controlling the whole group.

Entries with `applied: true` survive re-runs as an audit trail even when the
underlying deficit is no longer detected.  Thalweg groups are preserved verbatim
across runs so manual edits to `applied` flags are never overwritten.

### Option A — apply all suggestions at once

```bash
bathymetry-regrid --config my_run.yaml --skip-regrid --apply-all-fixes
```

Applies every entry in `fixes.yaml` regardless of its `applied` flag and marks
all entries `applied: true` in the file as an audit trail.

### Option B — selective: mark entries in fixes.yaml

Open `fixes.yaml` and set `applied: true` on the entries or thalweg groups you
want.  Setting the group flag on a `tw_` entry enables all fixes in that group:

```yaml
tw_great_belt:
  applied: true   # apply all Great Belt thalweg fixes
  ...
b001:
  applied: true   # apply this specific blocked-strait fix
  ...
```

Then re-run with `--accept-fixes`:

```bash
bathymetry-regrid --config my_run.yaml --skip-regrid --accept-fixes
```

Only the entries / groups marked `applied: true` are applied.

### Option C — reference chosen keys in your config

Every flat entry in `fixes.yaml` carries a short key (`b001`, `s001`, `lb001`, …).
Paste just the keys you want into the `fixes:` section of your config:

```yaml
fixes:
  - key: lb001   # LAND_BRIDGE — wet_fraction=0.18, bridges 2 basin(s)
  - key: s001    # SILL_DEFICIT — sill_ratio=0.52
```

The full fix (lon, lat, action, depth) is resolved automatically from
`fixes.yaml` at run time — no need to copy coordinates.

Then re-run:

```bash
bathymetry-regrid --config my_run.yaml --skip-regrid
```

### Option D — explicit file

Point to any YAML file that has a top-level `fixes:` dict or list:

```bash
bathymetry-regrid --config my_run.yaml --skip-regrid --fixes-file my_fixes.yaml
```

### Fix actions

| Action | Effect |
|--------|--------|
| `open_cell` | Force the cell ocean at the given depth (m); use for BLOCKED / LAND_BRIDGE |
| `set_depth` | Force the cell ocean at the given depth (m); use for SILL_DEFICIT / thalweg |
| `close_cell` | Force the cell to land (depth=NaN, mask=0); removes a spurious wet cell |

Both `open_cell` and `set_depth` accept `depth:` or `value:` as the depth key.

**Haney smoothing and fixed cells:** when fixes are applied, every
`set_depth`/`open_cell` cell is passed to `smooth_rx0()` as a `pin_mask`.
The LP lower-bounds those corrections to ≥ 0, so the solver deepens
neighbours rather than shallowing fixed cells.  The smoothed field therefore
satisfies rx0 ≤ target at every interface, verified by a logged post-smooth
check.

## Grid types

| Type | YAML `type:` | Key parameters |
|------|-------------|---------------|
| `SphericalGrid` | `spherical` | `lon_min/max`, `lat_min/max`, `dlon`, `dlat`, `rotation`, `interfaces` |
| `CartesianGrid` | `cartesian` | `x_min/max`, `y_min/max`, `dx`, `dy`, `crs`, `rotation` — or `center_lon`, `center_lat`, `x_size` (km), `y_size` (km) |
| `RotatedPoleGrid` | `rotated_pole` | `pole_lon/lat`, `rlon/rlat_min/max`, `drot`, `axis_rotation` |
| `SuperGrid` | `supergrid` | `file` path to MOM6/pyGETM ocean_hgrid.nc |

A non-zero `rotation` rotates the grid axes CCW around the grid centre using a
local tangent-plane approximation.  The corner-coordinate arrays become 2-D so
ESMF and plotly both handle them correctly.

`SphericalGrid` supports the `interfaces` flag (see above).  `CartesianGrid`
and `RotatedPoleGrid` always treat their bounds as corner positions.

## rx0 smoothing

The smoothing step finds the minimum-L1-norm depth corrections that reduce the
Haney (1991) slope factor

    rx0 = |H₁ − H₂| / (H₁ + H₂)

below the target value at every wet interface, by solving a linear programme
(scipy.optimize.linprog with sparse constraints).  This is a direct adaptation
of `pygetm.domain.Domain.smooth()` from the GETM ocean model.

The smoothed field is saved as a separate NetCDF variable
(`depth_rx0_0p20` for rx0=0.2) alongside the unsmoothed `depth`, allowing
multiple bathymetry variants in a single file.

## Mask region workflow

To exclude a water body (fjord, lagoon, estuary) from the model domain:

1. Add a `mask_regions:` entry that covers the **mouth** of the feature.
   Use geographic types (`rectangle`, `polygon`, `point`) when you know the
   coordinates, or index types (`ij_rectangle`, `ij_point`) when you have
   identified specific cells in the output NetCDF (0-based: `i` = lon column,
   `j` = lat row, as shown in ncview).
2. Rerun with `--skip-regrid`.  The masking step (4b) closes the mouth; the
   isolation step (4c) then automatically removes the now-disconnected interior.

This avoids having to enumerate every interior cell individually.

## Colormaps

All depth figures use [cmocean](https://matplotlib.org/cmocean/) colormaps
with matplotlib fallbacks:

| Field | cmocean | Convention |
|-------|---------|-----------|
| Depth | `deep_r` | Shallow = light, deep = dark |
| Wet fraction | `amp` | 0 = white → 1 = dark orange |
| Depth corrections | `balance` | Diverging around zero |

## Open boundaries

Pass `--write-boundaries` (or add nothing to the YAML — it is a CLI-only flag)
to write an open-boundary T-grid coordinate CSV alongside the output NetCDF:

```bash
bathymetry-regrid --config my_run.yaml --write-boundaries
```

The file `{name}_bdy.csv` is written in the same directory as the output
NetCDF.  It contains the T-point lon/lat of all wet cells on the four outer
edges, traversed in the order:

| Side | Direction | Corner ownership |
|------|-----------|-----------------|
| west | S → N (j increasing) | includes corners |
| north | W → E (i increasing) | i = 1 … nx−2 |
| east | S → N (j increasing) | includes corners |
| south | W → E (i increasing) | i = 1 … nx−2 |

Cells that are land on an edge are simply skipped.  A single wet segment per
side is the common case; if the edge has land interruptions, each contiguous
run is written in sequence.

The Markdown report gains an **Open boundaries** section with a table showing
the start/stop `(i, j)` indices and cell count for every contiguous segment:

```
west seg 1 │ i=0..0, j=5..62, n=58
north seg 1 │ i=1..348, j=99..99, n=348
east seg 1 │ i=349..349, j=8..99, n=92
```

## Boundary cross-section matching

When your inner model receives open-boundary conditions from a coarser outer
model, the bathymetry at the boundary faces can be inconsistent: the outer
model may have a deeper (or shallower) channel than the inner model because of
different source data or different regridding.  This can produce spurious
pressure gradients and incorrect volume transport at the nesting interface.

`nudge_boundaries:` in the YAML config applies a cross-section-area-preserving
nudge at step 4c-iii (after basin isolation, before strait detection):

```yaml
nudge_boundaries:
  enabled: true          # set false to keep config but skip the step (default true)
  outer_file: /path/to/outer_model_bathymetry.nc   # required when enabled
  # Specify open boundary cells with a file, side names, or both:
  boundaries_file: northsea_1d15deg_bdy.csv  # *_bdy.csv from --write-boundaries
  boundaries: [N, S, E, W]   # fallback: nudge all cells on these grid edges
  width: 10                  # taper zone width in cells (default 10)
  shape: cosine              # taper: cosine (default) | linear | exponential
  min_depth: 2.0             # floor for nudged depth (m)
  max_scale: 3.0             # clamp scale factor to [1/max_scale, max_scale]
  outer_depth_var: depth     # depth variable name in outer file; auto-detected if omitted
```

### Identifying open boundary cells

Two methods are supported and can be combined:

**`boundaries_file`** (recommended for nested models): point to the
`*_bdy.csv` file written by `--write-boundaries`.  The file format is a
two-row header ("T-grid" / "lon,lat") followed by one coordinate pair per
wet boundary cell.  Each coordinate is snapped to the nearest inner-grid
cell via KDTree, so only the actual open boundary cells are nudged —
important when only some edges carry an open boundary condition.

**`boundaries`**: list of side names (`N`, `S`, `E`, `W`).  Every wet cell
on those grid edges is treated as an open boundary cell.  This is the
default when no `boundaries_file` is given.

Both can appear together: the file-based cells are collected first, then
the side-based cells are appended and de-duplicated.

### How it works

For each boundary face the outer model cell that is nearest to each inner
boundary cell is found via KDTree nearest-neighbour lookup.  Inner cells that
share the same outer cell are grouped, and a scale factor is computed:

```
scale = A_outer / A_inner
      = (h_outer × width_outer) / Σᵢ (hᵢ × widthᵢ)
```

where width is the approximate along-face cell size in degrees × cos(lat).
The scale is clamped to `[1/max_scale, max_scale]` to prevent extreme
corrections from bad outer-model values.

A BFS flood propagates each scale factor inland up to `width` cells.  The
taper weight at distance `d` from the boundary is:

| `shape` | `w(d)` |
|---------|--------|
| `cosine` | `0.5 × (1 + cos(π d / width))` |
| `linear` | `1 − d / width` |
| `exponential` | `exp(−3 d / width)` |

The nudged depth is `depth_in × (w × scale + (1 − w))`, which equals
`depth_in × scale` at the boundary face and `depth_in` at `width` cells inland.

A minimum depth (`min_depth`) is enforced everywhere.  Cells that are
**deepened** by the nudge are soft-pinned for Haney smoothing: the LP can
deepen their neighbours but cannot shallow the nudged cells themselves.

### Outer file format

Any NetCDF file is accepted.  The depth variable is auto-detected from common
names (`depth`, `bathy`, `bathymetry`, `h`, `deptho`); override with
`outer_depth_var` if needed.  Coordinates are searched under names `lon`,
`longitude`, `nav_lon`, `x_T`, `xt_ocean` (and the lat equivalents).  Both 1-D
and 2-D lon/lat arrays are supported.  Depth values must be positive = wet (the
sign convention is auto-detected from the median of finite values).

## References

- Haney, R.L. (1991). On the pressure gradient force over steep topography in
  sigma coordinate ocean models. *J. Phys. Oceanogr.*, 21, 610–619.
- Beckmann, A. & Haidvogel, D.B. (1993). Numerical simulation of flow around a
  tall isolated seamount. *J. Phys. Oceanogr.*, 23, 1736–1753.
- EMODnet Bathymetry Consortium (2022). EMODnet Digital Bathymetry (DTM).
  https://doi.org/10.12770/ff3aff8a-cff1-44a3-a2c8-1910bf109f85
- GEBCO Compilation Group (2023). GEBCO 2023 Grid.
  https://doi.org/10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b
