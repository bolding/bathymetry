# Ocean model changes required for a rotated-pole grid

When moving from a regular lat/lon grid (e.g. AMM7) to a rotated-pole grid
(e.g. AMM15), most of the dynamical core is unchanged.  The changes cluster
around coordinate metadata, vector field handling, and output.

---

## What changes

### 1. Coriolis parameter

`f = 2Ω · sin(φ)` where `φ` is **geographic** latitude, not rotated latitude.
On a rotated grid the j-index no longer maps to geographic latitude, so f
cannot be computed from grid indices.  The model must use the stored 2-D
geographic latitude array at every cell centre (NEMO: `nav_lat`).

Production codes typically handle this correctly because they carry the full
geographic coordinate arrays, but it is an easy mistake in custom or
lightweight models that assume regular grids.

### 2. Vector rotation of atmospheric forcing  ← most critical

This is the most important and most error-prone change.

Wind stress `(τx, τy)` from an atmospheric model is delivered in
**geographic** coordinates (East, North).  The ocean model's velocity
components `(u, v)` are aligned with the **model grid axes**, which are
rotated by an angle α relative to geographic East/North.  Every vector field
must be rotated after interpolation onto the ocean grid:

```
u_model =  u_geo · cos(α) + v_geo · sin(α)
v_model = −u_geo · sin(α) + v_geo · cos(α)
```

The local rotation angle α varies spatially across the domain.  NEMO
computes it in `geo2oce.F90` and stores it as `gcost`/`gsint` arrays.

**Workflow when meteo forcing is interpolated at runtime:**

```
atmospheric grid (u_geo, v_geo)
        ↓  scalar interpolation  (bilinear / conservative)
model grid (u_geo_interp, v_geo_interp)    ← still in geographic frame
        ↓  vector rotation  (multiply by gcost/gsint)
model grid (u_model, v_model)              ← aligned with grid axes
```

The interpolation step and the rotation step must be kept separate — the
rotation cannot be folded into the interpolation weights.

**What goes wrong if this is missed:** the model runs without crashing, but
the wind-driven circulation is incorrect.  The error is largest near the
domain edges where α deviates most from zero, and manifests as a spurious
along-boundary current or wrong transport through straits.

### 3. Grid metrics (e1, e2)

Cell widths in metres (`e1`, `e2`) are computed from geographic distances
between cell corners.  No algorithmic change is needed — the same formulas
apply.  The values become nearly uniform across the domain (that is the
point of the rotated pole), reducing discretisation errors in the momentum
and tracer equations.

### 4. Open boundary velocities

Normal and tangential velocity components at open boundaries are provided by
a coarser model in geographic coordinates.  These must be rotated into model
grid coordinates (same formula as for wind stress) before being applied as
boundary conditions.  Tidal elevation (a scalar) is unaffected.

### 5. Diagnostic and product output

Model `(u, v)` components must be rotated back to geographic
`(U_east, V_north)` before delivery to users.  For CMEMS products the
rotated-grid output is also re-interpolated onto a regular geographic grid
(the yellow dashed box in the Tonani et al. 2019 Figure 1 is that regular
product grid laid on top of the tilted AMM15 domain).

---

## What does NOT change

| Component | Reason |
|-----------|--------|
| Governing equations | Written in tensor form; coordinate-independent |
| Pressure gradient | Operates in grid coordinates via scale factors |
| Advection and diffusion | Same; use e1, e2 |
| Vertical coordinate | Completely independent of horizontal grid type |
| Time-stepping scheme | Unchanged |
| Scalar forcing fields | SST relaxation, precipitation, heat flux magnitudes — interpolate and apply directly, no rotation needed |
| Tidal elevation at boundaries | Scalar; no rotation |

---

## Rotation angle α from RotatedPoleGrid

The local rotation angle between model grid axes and geographic East/North
can be computed analytically from the corner coordinates produced by
`RotatedPoleGrid._build()`:

```python
# At each cell centre (i, j), using geographic corners:
# right edge midpoint minus left edge midpoint
dlon = corner_lon[j, i+1] - corner_lon[j, i]
dlat = corner_lat[j, i+1] - corner_lat[j, i]
alpha = np.arctan2(dlat, dlon * np.cos(np.radians(center_lat[j, i])))
# alpha[j, i] is the angle of the model x-axis relative to geographic East
```

This array can be written to the bathymetry output NetCDF so the ocean model
can read it directly without recomputing.  Adding it as a planned output of
`RotatedPoleGrid` is noted for the implementation.

---

## Summary checklist for moving AMM7 → AMM15-style grid

- [ ] Store 2-D geographic lat/lon arrays (`nav_lat`, `nav_lon`) in domain file
- [ ] Compute Coriolis from `nav_lat`, not grid j-index
- [ ] Compute and store local rotation angle α (`gcost`, `gsint`)
- [ ] Apply vector rotation to all atmospheric vector forcing (wind stress u, v)
- [ ] Apply vector rotation to open boundary velocity components
- [ ] Apply inverse rotation to u, v before writing user-facing output
- [ ] Recompute e1, e2 from geographic corner distances (they will now be nearly uniform)
- [ ] Verify bathymetry is regridded onto the rotated corner coordinates (this tool handles that)

---

## Reference

Tonani, M. et al. (2019).  The impact of a new high-resolution ocean model on
the Met Office North-West European Shelf forecasting system.  *Ocean Sci.*,
15, 1133–1158.  <https://doi.org/10.5194/os-15-1133-2019>

See also: `docs/amm7_amm15_grids.md` for the AMM7/AMM15 grid type comparison.
