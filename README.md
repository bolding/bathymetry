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
├── example_northsea.yaml          ← template configuration file
├── lib/                           ← flat library modules (on sys.path after install)
│   ├── grid.py        SphericalGrid, CartesianGrid, CurvilinearGrid (stub)
│   ├── reader.py      GEBCO and EMODnet readers
│   ├── interpolate.py xESMF conservative regridding with weight-file caching
│   ├── analysis.py    strait detection, mask regions, isolated-cell masking
│   ├── smooth.py      rx0 slope-factor smoothing (LP, ported from GETM)
│   └── report.py      Markdown report, ASCII tables, cartopy + plotly plots
└── cli/
    └── regrid.py      CLI entry point (bathymetry-regrid)
```

## Quick start

### From a YAML config (recommended)

```bash
bathymetry-regrid --config example_northsea.yaml
```

Override any setting on the command line:

```bash
bathymetry-regrid --config example_northsea.yaml --smooth-rx0 0.15 --name northsea_v2
```

### Spherical grid — North Sea, command-line only

```bash
bathymetry-regrid \
    --source /server/data/GEBCO/GEBCO_2023.nc \
    --name northsea_0p05deg \
    --grid spherical \
    --lon-min 0 --lon-max 15 --lat-min 50 --lat-max 60 \
    --dlon 0.05 --dlat 0.05 \
    --min-depth 2 \
    --smooth-rx0 0.2 \
    --output northsea.nc \
    --report-dir ./report/northsea/
```

### Equidistant spherical grid

By default `dlon` and `dlat` are both in degrees, so cells become narrower
towards the poles.  Use `--equidistant` to compute `dlon` automatically from
`dlat` and the central latitude, giving approximately square cells in physical
distance:

```bash
bathymetry-regrid \
    --source /server/data/GEBCO/GEBCO_2023.nc \
    --name northsea_equidist \
    --grid spherical \
    --lon-min 0 --lon-max 15 --lat-min 50 --lat-max 60 \
    --dlat 0.05 --equidistant \
    --min-depth 2 --output northsea.nc
```

At a central latitude of 55 °N, `cos(55°) ≈ 0.574`, so `dlon ≈ 0.05 / 0.574 ≈ 0.0872°`.
The grid will have fewer longitude points than latitude points (≈ 172 × 200 instead of
300 × 200 for `dlon = dlat = 0.05°`).  The computed values are printed at startup:

```
[equidistant] lat_center=55.00°  dlon=0.087126°  dlat=0.05°  → grid 172 × 200 (lon × lat)
```

In YAML:

```yaml
grid:
  type: spherical
  lon_min: 0.0
  lon_max: 15.0
  lat_min: 50.0
  lat_max: 60.0
  dlat: 0.05          # dlon is computed automatically
  equidistant: true
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
A fully annotated template is provided in `example_northsea.yaml`.

```yaml
# Short identifier — used in all output file and directory names.
# E.g. name: northsea_0p05deg  →  northsea_0p05deg.nc,
#                                   report/northsea_0p05deg/northsea_0p05deg_*.png
name: northsea_0p05deg

source: /server/data/GEBCO/GEBCO_2023.nc   # or "emodnet"
pad_deg: 1.0                                # buffer added when reading source

grid:
  type: spherical        # spherical | cartesian
  lon_min: 0.0
  lon_max: 15.0
  lat_min: 50.0
  lat_max: 60.0
  dlon: 0.05
  dlat: 0.05
  rotation: 0.0          # degrees CCW (non-zero → rotated curvilinear grid)

regridding:
  cache_dir: ./regrid_weights   # xESMF weight files cached here
  min_depth: 2.0                # shallow-clamp after regridding (m)
  min_wet_fraction: 0.05
  coastline_mask: "10m"         # overlay NE land polygons on source; omit to skip

analysis:
  nkeep_basins: 1
  wet_frac_threshold: 0.3
  sill_ratio_threshold: 0.7
  area_ratio_threshold: 0.5

smooth:
  rx0: 0.2               # Haney slope-factor target; omit / null to skip

output:
  file: northsea_0p05deg.nc
  report_dir: ./report/northsea_0p05deg

# After the first run, re-run with --accept-fixes to apply all suggested fixes,
# or --fixes-file <path> for an explicit fixes file.
# To hand-pick fixes, paste selected entries here.
# Actions: set_depth, open_cell, close_cell
# fixes:
#   - lon: 5.3
#     lat: 55.7
#     action: set_depth
#     value: 22.0

# Force geographic areas to land (step 4c, before isolation masking).
# Masking a fjord mouth causes its interior to be removed automatically
# by the isolation step (4d).  Types: rectangle, polygon, point.
# mask_regions:
#   - type: rectangle
#     lon_min: 9.0  lon_max: 10.5  lat_min: 54.5  lat_max: 55.5
#     name: Kiel Bight
#   - type: polygon
#     vertices: [[10,55],[11,55.5],[11.5,54.5],[10,54]]
#     name: Custom bay
#   - type: point
#     lon: 8.5  lat: 54.8
#     name: Isolated pool
```

## Pipeline steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `grid` | Build target grid from parameters |
| 2 | `reader` | Read and clip fine-resolution source bathymetry |
| 3 | `interpolate` | xESMF conservative regrid; weight file cached for reuse |
| 4a | `analysis` | Flag narrow / blocked interfaces (SILL_DEFICIT, AREA_DEFICIT, BLOCKED) |
| 4b | `analysis` | Apply user fixes from `fixes:` (set_depth, open_cell, close_cell) |
| 4c | `analysis` | Apply explicit `mask_regions:` (rectangle, polygon, point) |
| 4d | `analysis` | Remove isolated ocean cells (flood-fill; keeps *nkeep* largest basins) |
| 5 | `smooth` | rx0 slope smoothing via linear programming (optional) |
| 6 | — | Write output NetCDF + final plots for every depth variable |

**Why 4c before 4d?**  
Masking the mouth of a fjord or lagoon (4c) makes the interior cells
disconnected from the main ocean.  Running isolation masking afterwards (4d)
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
| `{name}_02_source_depth.png` | Source bathymetry overview map |
| `{name}_03_regrid_result.png` | Regridded depth (cmocean *deep_r*) |
| `{name}_03_wet_fraction.png` | Wet fraction per coarse cell (cmocean *amp*) |
| `{name}_04a_straits.png/.html` | Strait analysis map (interactive) |
| `{name}_04a_straits.csv` | Flagged interface table |
| `{name}_04a_section_*.png` | Cross-section profiles with map inset |
| `{name}_fixes_suggested.yaml` | Suggested strait fixes, grouped by cause (BLOCKED / SILL_DEFICIT / AREA_DEFICIT) |
| `{name}_04c_mask_regions.png` | Depth after explicit masking (if used) |
| `{name}_04d_basins.png` | Connected basin map (kept=blue, removed=red) |
| `{name}_05_smooth_*.png` | rx0 histogram + depth-correction map |
| `{name}_06_final_depth.png/.html` | Final unsmoothed bathymetry (interactive) |
| `{name}_06_final_depth_rx0_*.png/.html` | Final smoothed bathymetry (interactive) |

## Strait categories

| Category | Meaning | Suggested fix |
|----------|---------|--------------|
| `SILL_DEFICIT` | Coarse sill shallower than fine-grid sill — dense overflow blocked | `set_depth` to fine-grid sill depth |
| `AREA_DEFICIT` | Cross-sectional area under-represented — transport too weak | Widen or deepen at the interface |
| `BLOCKED` | No wet fine-resolution path between two wet coarse cells | `open_cell` on the blocking point |

Suggested fixes are written to `fixes_suggested.yaml`, grouped by cause.
On the next run, load them automatically with `--accept-fixes` (reads from the
report directory) or `--fixes-file <path>` for an explicit path.  You can also
paste selected entries into the `fixes:` section of your YAML config.

## Grid types

| Type | Status | Key parameters |
|------|--------|---------------|
| `SphericalGrid` | Implemented | `lon_min/max`, `lat_min/max`, `dlon`, `dlat`, `rotation_deg` |
| `CartesianGrid` | Implemented | `x_min/max`, `y_min/max`, `dx`, `dy`, `crs`, `rotation_deg` |
| `CurvilinearGrid` | Planned (stub) | Pre-computed 2-D corner lon/lat arrays |

A non-zero `rotation_deg` rotates the grid axes CCW around the grid centre
using a local tangent-plane approximation.  The corner-coordinate arrays
become 2-D so ESMF and plotly both handle them correctly.

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

## Incremental fixing workflow

The regridding step (step 3) is the most expensive part of the pipeline.
After the first run, the raw post-regrid result is cached as
`{cache_dir}/{name}_raw_regrid.nc`.  Subsequent runs can load this cache and
skip straight to the analysis and fixing steps:

```bash
# First run — regrids and caches the result
bathymetry-regrid --config my_run.yaml

# Inspect fixes_suggested.yaml, add selected entries to my_run.yaml, then:
bathymetry-regrid --config my_run.yaml --skip-regrid

# Apply another round of fixes without re-regridding
bathymetry-regrid --config my_run.yaml --skip-regrid
```

`--skip-regrid` also pairs with `--accept-fixes`:

```bash
bathymetry-regrid --config my_run.yaml --skip-regrid --accept-fixes
```

The output NetCDF (`{name}.nc`) is always regenerated at the end of each run
(with the current fixes and smoothing applied), but the raw-regrid cache is
never overwritten by `--skip-regrid`, so the base data is always recoverable.

## Mask region workflow

To exclude a water body (fjord, lagoon, estuary) from the model domain:

1. Add a `mask_regions:` entry (rectangle, polygon, or point) that covers the
   **mouth** of the feature.
2. Rerun.  The masking step (4c) closes the mouth; the isolation step (4d)
   then automatically removes the now-disconnected interior.

This avoids having to enumerate every interior cell individually.

## Colormaps

All depth figures use [cmocean](https://matplotlib.org/cmocean/) colormaps
with matplotlib fallbacks:

| Field | cmocean | Convention |
|-------|---------|-----------|
| Depth | `deep_r` | Shallow = light, deep = dark |
| Wet fraction | `amp` | 0 = white → 1 = dark orange |
| Depth corrections | `balance` | Diverging around zero |

## References

- Haney, R.L. (1991). On the pressure gradient force over steep topography in
  sigma coordinate ocean models. *J. Phys. Oceanogr.*, 21, 610–619.
- Beckmann, A. & Haidvogel, D.B. (1993). Numerical simulation of flow around a
  tall isolated seamount. *J. Phys. Oceanogr.*, 23, 1736–1753.
- EMODnet Bathymetry Consortium (2022). EMODnet Digital Bathymetry (DTM).
  https://doi.org/10.12770/ff3aff8a-cff1-44a3-a2c8-1910bf109f85
- GEBCO Compilation Group (2023). GEBCO 2023 Grid.
  https://doi.org/10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b
