#!/usr/bin/env python3
"""
How greenland_20260724_inputs.json was made. It is only needed to make the
inputs for another day or place; katabatic_idealised_input.py reads the JSON
file and needs no RACMO data.

The source is a nested, RACMO-driven MicroHH set-up of the ice-sheet margin,
as written by racmo_to_openbc.py: an outer domain (400 m, folder dom0/) and an
inner domain (100 m, folder dom1/). The script takes the terrain along x
through a weather station and RACMO's profiles at the ice-sheet (east)
boundary, and writes them to one small JSON file.

    python3 extract_inputs.py --case /path/to/the/nested/setup \
            --out greenland_20260724_inputs.json [--station 67.09545,-50.0567]

What it reads (all written by racmo_to_openbc.py):
  dom0/racmo_openbc_diagnostics.nc   400 m DEM (dem_amsl), ib_ice, lat, lon
  dom1/racmo_openbc_diagnostics.nc   100 m DEM of the inner domain (optional)
  dom0/racmo_openbc.ini              [thermo] pbot, the domain floor pressure
  dom0/lbc_{thl,qt,u,v}_east.<t>     RACMO at the east (ice-sheet) boundary,
                                     every 3 h; averaged over the rows within
                                     --band m of the station row and over the
                                     sponge columns

The transect is the row of the outer domain nearest the station (and the
inner-domain row at the same latitude), x as in the diagnostics file
(centred on the domain, east positive).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import netCDF4 as nc

STATION = (67.095450, -50.056700)      # the weather station of 24 Jul 2026


def row_through(diag, latlon):
    lat, lon = np.array(diag['lat'][:]), np.array(diag['lon'][:])
    d2 = (lat - latlon[0])**2 + ((lon - latlon[1]) * np.cos(np.deg2rad(latlon[0])))**2
    return np.unravel_index(np.argmin(d2), d2.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--band', type=float, default=4000., help='rows within this distance of the station [m]')
    ap.add_argument('--station', default=None, help='lat,lon of the station (default: %.5f,%.5f)' % STATION)
    a = ap.parse_args()
    site = tuple(float(v) for v in a.station.split(',')) if a.station else STATION
    d0 = os.path.join(a.case, 'dom0')
    g0 = nc.Dataset(os.path.join(d0, 'racmo_openbc_diagnostics.nc'))
    j0, i0 = row_through(g0, site)
    x0 = np.array(g0['x'][:])
    dema0 = np.array(g0['dem_amsl'][:])
    floor_amsl = float(np.median(dema0 - np.array(g0['dem'][:])))
    out = dict(source=os.path.abspath(a.case), site='automatic weather station on the ice slope',
               site_latlon=site, floor_amsl=floor_amsl,
               outer=dict(dx=float(x0[1] - x0[0]), x=x0.tolist(), z_amsl=dema0[j0].tolist(),
                         ice=np.array(g0['ib_ice'][:])[j0].astype(int).tolist(), j=int(j0), i_site=int(i0),
                         x_site=float(x0[i0])))
    d1 = os.path.join(a.case, 'dom1', 'racmo_openbc_diagnostics.nc')
    if os.path.exists(d1):
        g1 = nc.Dataset(d1)
        j1, i1 = row_through(g1, site)
        x1 = np.array(g1['x'][:])
        # the slope of the ice in the inner domain: the mean surface gradient over
        # the ice cells (the large-scale slope) and the median local slope
        z1 = np.array(g1['dem_amsl'][:])
        ice1 = np.array(g1['ib_ice'][:]) > 0.5
        gy, gx = np.gradient(z1, x1[1] - x1[0])
        loc = np.degrees(np.arctan(np.hypot(gx, gy)))
        out['inner_ice_slope'] = dict(
            mean_gradient_deg=float(np.degrees(np.arctan(np.hypot(gx[ice1].mean(), gy[ice1].mean())))),
            mean_dzdx=float(gx[ice1].mean()), mean_dzdy=float(gy[ice1].mean()),
            median_local_deg=float(np.median(loc[ice1])), mean_local_deg=float(loc[ice1].mean()),
            z_ice_min=float(z1[ice1].min()), z_ice_max=float(z1[ice1].max()),
            extent_m=float(x1[-1] - x1[0] + (x1[1] - x1[0])))
        out['inner'] = dict(dx=float(x1[1] - x1[0]), x=x1.tolist(),
                           z_amsl=np.array(g1['dem_amsl'][:])[j1].tolist(),
                           ice=np.array(g1['ib_ice'][:])[j1].astype(int).tolist(), j=int(j1), i_site=int(i1),
                           x_site=float(x1[i1]))
    ini = open(os.path.join(d0, 'racmo_openbc.ini')).read()
    out['pbot'] = float(re.search(r'^pbot=(.*)$', ini, re.M).group(1))
    m = re.search(r'^# datetime_utc = (.*)$', ini, re.M) or re.search(r'^datetime_utc=(.*)$', ini, re.M)
    out['datetime_utc_t0'] = m.group(1).strip() if m else None
    # RACMO at the east boundary
    z = np.array(g0['z'][:])
    kt = len(z)
    y0 = np.array(g0["y"][:])
    jt = len(y0)
    rows = np.where(np.abs(y0 - y0[j0]) <= a.band)[0]
    z_east = float(dema0[rows, -1].mean())
    prof = {}
    for f in sorted(glob.glob(os.path.join(d0, 'lbc_thl_east.*'))):
        t = int(f.rsplit('.', 1)[1])
        rec = {}
        for v in ('thl', 'qt', 'u', 'v'):
            arr = np.fromfile(os.path.join(d0, f'lbc_{v}_east.{t:07d}'))
            arr = arr.reshape(kt, jt + 6, -1)          # (k, ghost + j + ghost, columns)
            # rows: the halo has 3 ghost rows below row 0
            rec[v] = arr[:, rows + 3, :].mean(axis=(1, 2)).tolist()
        prof[str(t)] = rec
    out['racmo_east'] = dict(z_amsl=(z + floor_amsl).tolist(), z_surface_amsl=z_east, times=prof,
                             note='RACMO at the east (ice-sheet) boundary of the outer domain, mean over '
                                  f'the sponge columns and the rows within {a.band:g} m of the station; values '
                                  'below the terrain are fill')
    json.dump(out, open(a.out, 'w'), indent=0)
    print(f'wrote {a.out}: outer row j={j0} (station at x = {x0[i0]/1e3:.1f} km, {dema0[j0, i0]:.0f} m), '
          f'{"inner row j=%d, " % out["inner"]["j"] if "inner" in out else ""}'
          f'{len(prof)} RACMO times, east surface {z_east:.0f} m, pbot {out["pbot"]:.1f} Pa at {floor_amsl:.1f} m')


if __name__ == '__main__':
    main()
