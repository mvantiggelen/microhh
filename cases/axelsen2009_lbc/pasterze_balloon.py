#!/usr/bin/env python3
"""
IMAU tethered-balloon (ADAS) soundings at Pasterze site A1, PASTEX 1994
(files ADAS_<MDD>-<HH>h<MM>-schoon, "cleaned"): read, heights from the
hypsometric equation, potential temperature, along-slope wind.

    python3 pasterze_balloon.py DIR... [--out pasterze_balloon_1994.json] [--downslope-from 300]

File format (as found: CR line ends, 7 tab-separated columns per record of
about 10 s): elapsed time MMSS (min*100 + s), pressure (hPa), temperature
(deg C), relative humidity (%), wind speed (m/s), wind direction (deg, from),
and a 7th column that is not used here. Records with pressure < 600 hPa, or
with speed = direction = col7 = 0 (end-of-file markers), are dropped.

Height above the launch point:
    z = int R_d T_v / g d ln p (from the surface, along the ascent),  T_v = T (1 + 0.61 q),
with q from RH (Magnus formula, e_s = 6.1094 exp(17.625 T/(T + 243.04)) hPa). The reference pressure
is the first record; if the last record is back at the surface (within
1 hPa) a linear drift between the two is removed (the profile is up and down).
Potential temperature relative to the launch-point pressure p_s, as Axelsen
(2010) plots it: theta = (T + 273.15) (p_s/p)^(R_d/c_p) - 273.15.
Along-slope wind: u_s = speed cos(direction - D), D = --downslope-from, the
direction a downslope wind comes from (fitted below to Axelsen Fig. 3.3a if
observations_P1.json is next to this script).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

RD, CP, G = 287.05, 1004.6, 9.81


def read(path):
    txt = open(path, 'rb').read().decode('latin1').replace('\r', '\n')
    rows = [l.split() for l in txt.split('\n') if l.strip()]
    a = np.array([[float(x) for x in r] for r in rows if len(r) == 7])
    keep = (a[:, 1] > 600.) & ~((a[:, 4] == 0.) & (a[:, 5] == 0.) & (a[:, 6] == 0.))
    a = a[keep]
    mmss = a[:, 0]
    t_s = (mmss // 100) * 60. + mmss % 100
    p, T, rh, ff, dd = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    es = 6.1094 * np.exp(17.625 * T / (T + 243.04))
    e = np.clip(rh, 0., 100.) / 100. * es
    q = 0.622 * e / (p - 0.378 * e)
    Tv = (T + 273.15) * (1. + 0.61 * q)
    p_ref = np.full_like(p, p[0])
    drift = abs(p[-1] - p[0]) < 1.
    if drift and t_s[-1] > t_s[0]:
        p_ref = p[0] + (p[-1] - p[0]) * (t_s - t_s[0]) / (t_s[-1] - t_s[0])
    # Height: z = int R_d T_v / g d ln p from the surface. x = ln(p_ref/p) of
    # every record; the hypsometric integral z(x) is built on the ascent (in x
    # order) and every record, ascent and descent, is mapped through it.
    x = np.log(p_ref / p)
    top = int(np.argmax(x))
    xa, tva = x[:top + 1], Tv[:top + 1]
    o = np.argsort(xa)
    xa, tva = np.concatenate(([0.], xa[o])), np.concatenate(([Tv[0]], tva[o]))
    za = np.concatenate(([0.], np.cumsum(RD / G * 0.5 * (tva[1:] + tva[:-1]) * np.diff(xa))))
    zi = np.interp(x, xa, za)
    zi[x < 0.] = RD / G * Tv[x < 0.] * x[x < 0.]
    theta = (T + 273.15) * (p_ref / p)**(RD / CP) - 273.15
    up = np.zeros(len(p), bool)
    up[:top + 1] = True
    m = re.search(r'ADAS_(\d)(\d\d)-(\d\d)h(\d\d)', os.path.basename(path))
    month, day, hh, mm = (int(x) for x in m.groups())
    return dict(file=os.path.basename(path), date=f'1994-{month:02d}-{day:02d}', launch=f'{hh:02d}:{mm:02d}',
                doy=int((np.datetime64(f'1994-{month:02d}-{day:02d}') - np.datetime64('1994-01-01'))
                        / np.timedelta64(1, 'D')) + 1,
                t_s=t_s.tolist(), p_hPa=p.tolist(), T_C=T.tolist(), rh=rh.tolist(), ff=ff.tolist(), dd=dd.tolist(),
                z_m=zi.tolist(), theta_C=theta.tolist(), ascent=up.tolist(), p_surface_drift_removed=bool(drift))


def u_slope(s, D):
    return (np.array(s['ff']) * np.cos(np.deg2rad(np.array(s['dd']) - D))).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', default='pasterze_balloon_1994.json')
    ap.add_argument('--downslope-from', type=float, default=None,
                    help='direction (deg) a downslope wind comes from (default: fitted to Axelsen Fig. 3.3a)')
    a = ap.parse_args()
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, 'ADAS_*')))
    S = [read(f) for f in files]

    D, how = a.downslope_from, '--downslope-from'
    obs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'observations_P1.json')
    ref = next((s for s in S if s['doy'] == 212 and s['launch'] == '10:25'), None)
    if D is None and os.path.exists(obs) and ref is not None:
        dig = np.array(json.load(open(obs))['axelsen']['fig3_3a']['z_m_u_ms'])     # (z, u)
        up = np.array(ref['ascent'])           # Axelsen plots the ascent only
        z = np.array(ref['z_m'])[up]
        best = None
        for d in np.arange(250., 350.1, 0.5):
            u = np.array(u_slope(ref, d))[up]
            o = np.argsort(z)
            ui = np.interp(dig[:, 0], z[o], u[o])
            err = np.sqrt(np.mean((ui - dig[:, 1])**2))
            if best is None or err < best[0]:
                best = (err, d)
        D = best[1]
        how = f'fitted to Axelsen (2010) Fig. 3.3a (JD 212 10:25 sounding), rms {best[0]:.2f} m/s'
    if D is None:
        D, how = 300., 'default 300 deg'
    for s in S:
        s['u_slope'] = u_slope(s, D)
    out = dict(_source=('IMAU tethered-balloon soundings (ADAS), PASTEX 1994, Pasterze site A1 (~2100 m asl); '
                        'files from the user (ADAS-cleaned). Heights hypsometric, theta relative to the launch '
                        'pressure (as Axelsen 2010), u_slope = speed cos(direction - downslope_from).'),
               downslope_from_deg=D, downslope_from_how=how, soundings=S)
    json.dump(out, open(a.out, 'w'))
    print(f'downslope_from = {D:g} deg ({how})')
    for s in S:
        z = np.array(s['z_m'])
        print(f"  {s['date']} (DOY {s['doy']}) {s['launch']}: {len(z)} records, z_max {z.max():5.0f} m, "
              f"theta_s {s['theta_C'][0]:5.2f} C, u_s(<20 m) max {np.max(np.where(z < 20, s['u_slope'], -99)):4.1f} m/s, "
              f"drift removed {s['p_surface_drift_removed']}")
    print('wrote', a.out)


if __name__ == '__main__':
    main()
