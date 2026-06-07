#!/usr/bin/env python3
"""Demo: rotated-pole grid vs regular lat/lon over the North Sea.

Shows three panels:
  1. Regular lat/lon  — cell width varies with latitude
  2. Rotated pole     — cells equidistant, axes still roughly N-S/E-W
  3. Rotated pole + 30° axis rotation  — cells equidistant AND axes tilted
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# Rotated-pole inverse transform  (CF convention)
# (rlon, rlat) in rotated system  →  geographic (lon, lat)
# pole_lon/pole_lat = geographic location of the rotated North Pole
# ---------------------------------------------------------------------------

def rot2geo(rlon, rlat, pole_lon, pole_lat):
    rlon = np.asarray(rlon, float)
    rlat = np.asarray(rlat, float)
    rr = np.radians(rlon)
    lr = np.radians(rlat)
    pp = np.radians(pole_lat)

    sin_lat = np.sin(pp) * np.sin(lr) + np.cos(pp) * np.cos(lr) * np.cos(rr)
    lat = np.degrees(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))

    cos_lat = np.cos(np.radians(lat))
    safe = cos_lat > 1e-10
    sn = np.where(safe, np.cos(lr) * np.sin(rr) / cos_lat, 0.0)
    cs = np.where(safe,
                  (np.cos(pp) * np.sin(lr)
                   - np.sin(pp) * np.cos(lr) * np.cos(rr)) / cos_lat,
                  np.sign(np.cos(pp)))
    lon = (pole_lon + np.degrees(np.arctan2(sn, cs)) + 180) % 360 - 180
    return lon, lat


def rot2geo_axrot(rlon, rlat, pole_lon, pole_lat, axis_deg):
    """Apply an axis rotation before the pole transform.

    Rotating the (rlon, rlat) axes by axis_deg CCW tilts all grid lines
    by that angle on the geographic map.  The domain centre (0, 0) is
    unchanged.
    """
    th = np.radians(axis_deg)
    c, s = np.cos(th), np.sin(th)
    rlon_r = c * np.asarray(rlon, float) - s * np.asarray(rlat, float)
    rlat_r = s * np.asarray(rlon, float) + c * np.asarray(rlat, float)
    return rot2geo(rlon_r, rlat_r, pole_lon, pole_lat)


# ---------------------------------------------------------------------------
# Domain parameters
# ---------------------------------------------------------------------------
lon0, lat0 = 2.0, 55.0          # North Sea centre (geographic)

# Pole placement: puts (lon0, lat0) at the rotated origin (rlon=0, rlat=0)
#   sin(lat0) = cos(pole_lat)  →  pole_lat = 90 - lat0
#   lon0 = pole_lon + 180      →  pole_lon = lon0 - 180
pole_lat = 90.0 - lat0          # 35 °N
pole_lon = (lon0 - 180.0 + 180) % 360 - 180   # −178 °E

drot = 1.0    # rotated-degree cell size (coarse for legibility)
rext = 9.0    # domain half-width in rotated degrees
axis_rot = 30.0

N = 300       # points per iso-line for smooth curves
rlin = np.linspace(-rext, rext, N)
rvals = np.arange(-rext, rext + drot, drot)

# Earth radius for physical-size annotation
R = 6371.0
dx_rot = R * np.radians(drot)   # km per rotated degree ≈ same in both axes

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                          subplot_kw=dict(projection=ccrs.PlateCarree()))
geo_ext = [-10, 15, 44, 68]

LAND  = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                      facecolor='#f5f0e8', edgecolor='none')
COAST = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                      edgecolor='#555', facecolor='none', lw=0.6)

def _base(ax):
    ax.set_extent(geo_ext)
    ax.add_feature(LAND, zorder=1)
    ax.add_feature(COAST, zorder=2)
    gl = ax.gridlines(draw_labels=True, lw=0.4, color='grey',
                      alpha=0.6, linestyle='--', zorder=0)
    gl.top_labels = gl.right_labels = False
    return gl

# ---------------------------------------------------------------------------
# Panel 1 — regular lat/lon
# ---------------------------------------------------------------------------
ax = axes[0]
_base(ax)
ax.set_title('1  Regular lat/lon\n(cell width varies with latitude)', fontsize=11)

dlon_eq = drot / np.cos(np.radians(lat0))   # equidistant at centre
lons = np.arange(lon0 - rext * dlon_eq, lon0 + rext * dlon_eq + dlon_eq, dlon_eq)
lats = np.arange(lat0 - rext * drot,    lat0 + rext * drot + drot,    drot)

for lo in lons:
    ax.plot([lo, lo], [lats[0], lats[-1]], color='steelblue', lw=0.9,
            transform=ccrs.PlateCarree(), zorder=3)
for la in lats:
    ax.plot([lons[0], lons[-1]], [la, la], color='steelblue', lw=0.9,
            transform=ccrs.PlateCarree(), zorder=3)
ax.plot(lon0, lat0, 'b+', ms=11, mew=2, transform=ccrs.PlateCarree(), zorder=5)

dx_bot = R * np.cos(np.radians(lats[ 0])) * np.radians(dlon_eq)
dx_top = R * np.cos(np.radians(lats[-1])) * np.radians(dlon_eq)
dy     = R * np.radians(drot)
ax.text(0.02, 0.04,
        f'Δlon={dlon_eq:.2f}°  Δlat={drot:.1f}°\n'
        f'Cell height = {dy:.0f} km (constant)\n'
        f'Cell width at {lats[0]:.0f}°N = {dx_bot:.0f} km\n'
        f'Cell width at {lats[-1]:.0f}°N = {dx_top:.0f} km',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.9))

# ---------------------------------------------------------------------------
# Panel 2 — rotated pole, no axis rotation
# ---------------------------------------------------------------------------
ax = axes[1]
_base(ax)
ax.set_title(f'2  Rotated pole (no axis rotation)\n'
             f'pole at {pole_lat:.0f}°N, {pole_lon:.0f}°E — cells equidistant',
             fontsize=11)

for rv in rvals:
    lo, la = rot2geo(np.full(N, rv), rlin, pole_lon, pole_lat)
    ax.plot(lo, la, color='firebrick', lw=0.9, transform=ccrs.PlateCarree(), zorder=3)
    lo, la = rot2geo(rlin, np.full(N, rv), pole_lon, pole_lat)
    ax.plot(lo, la, color='firebrick', lw=0.9, transform=ccrs.PlateCarree(), zorder=3)

ax.plot(pole_lon, pole_lat, 'k*', ms=13, transform=ccrs.PlateCarree(), zorder=5,
        label=f'Pole ({pole_lat:.0f}°N, {pole_lon:.0f}°E)')
ax.plot(lon0, lat0, 'r+', ms=11, mew=2, transform=ccrs.PlateCarree(), zorder=5,
        label=f'Centre = rlon=0, rlat=0')
ax.legend(fontsize=8.5, loc='lower right')

ax.text(0.02, 0.04,
        f'Δrot={drot:.1f}° (both axes)\n'
        f'Cell size ≈ {dx_rot:.0f} km everywhere\n'
        f'Lines curve slightly but stay\nroughly N–S / E–W',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.9))

# ---------------------------------------------------------------------------
# Panel 3 — rotated pole + 30° axis rotation
# ---------------------------------------------------------------------------
ax = axes[2]
_base(ax)
ax.set_title(f'3  Rotated pole + {axis_rot:.0f}° axis rotation\n'
             f'cells equidistant AND axes tilted',
             fontsize=11)

for rv in rvals:
    lo, la = rot2geo_axrot(np.full(N, rv), rlin, pole_lon, pole_lat, axis_rot)
    ax.plot(lo, la, color='seagreen', lw=0.9, transform=ccrs.PlateCarree(), zorder=3)
    lo, la = rot2geo_axrot(rlin, np.full(N, rv), pole_lon, pole_lat, axis_rot)
    ax.plot(lo, la, color='seagreen', lw=0.9, transform=ccrs.PlateCarree(), zorder=3)

ax.plot(pole_lon, pole_lat, 'k*', ms=13, transform=ccrs.PlateCarree(), zorder=5,
        label=f'Pole ({pole_lat:.0f}°N, {pole_lon:.0f}°E)')
lo_c, la_c = rot2geo_axrot(0, 0, pole_lon, pole_lat, axis_rot)
ax.plot(lo_c, la_c, 'g+', ms=11, mew=2, transform=ccrs.PlateCarree(), zorder=5,
        label=f'Centre ({la_c:.1f}°N, {lo_c:.1f}°E)')
ax.legend(fontsize=8.5, loc='lower right')

ax.text(0.02, 0.04,
        f'Δrot={drot:.1f}° (both axes)\n'
        f'Cell size ≈ {dx_rot:.0f} km everywhere\n'
        f'Axis rotation = {axis_rot:.0f}° CCW',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.9))

# ---------------------------------------------------------------------------
plt.suptitle('Rotated-pole grid variants — North Sea domain', fontsize=13, y=1.01)
plt.tight_layout()
out = 'rotated_pole_demo.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
