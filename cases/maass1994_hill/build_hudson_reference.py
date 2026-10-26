#!/usr/bin/env python3
"""
Build reference_hudson1996.json from the LDV measurements of Hudson,
Dykhno & Hanratty (case 076 of the ERCOFTAC Classic Collection, data set
wb3 of Hudson 1993), the experiment Maass & Schumann (1994) simulated.

    python3 build_hudson_reference.py            # reads hudson_case076_raw.json

Source: http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case076
(files readme.txt, xlam00.dat ... xlam09.dat, pressure.dat; downloaded 6 Oct
2026). hudson_case076_raw.json holds the first six columns of xlam0n.dat
(y [cm], U, V, u', v' [cm/s], u'v' [(cm/s)^2], 5 significant digits);
the near-wall pressures below are copied from pressure.dat.

Conventions of the data (readme.txt and the case page):
  - y is the height above the LOCAL wavy surface, x along the channel,
    "v" the vertical (wall-normal at the mean surface) component.
  - U_b = 12.0 cm/s is the bulk velocity of the LOWER channel half;
    Re_h = U_b h / nu = 3380 with h = H/2 = 2.54 cm (Re_H = 6760).
  - friction velocity at the wavy wall u_tau = 1.29 cm/s; nu = 0.0088 cm2/s.
  - u'v' is listed positive in the lower half: it is -<u'v'> (the
    downward momentum flux); this script stores <u'w'> = -u'v'.
  - x/lambda = 0 of the data is ~1.2 mm DOWNSTREAM of the wave crest; the
    surface is (y - y_o)/h = 0.1 cos(2 pi x/lambda + 0.144). This script
    stores x_true = x/lambda + 0.144/(2 pi), the position relative to the
    crest, which is what the MicroHH case uses (crest at x = 0).
  - pressure.dat: P+ = p/(rho u_tau^2), vertical profiles at the 10
    stations; the mean pressure gradient (~4 +- 2 P+ per wave, pressure
    decreasing downstream) is NOT removed; the value nearest the wall at
    x/lambda = 0.3 is the third one (the two lowest are missing).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H_CM = 5.08            # channel height
UB = 12.0              # bulk velocity of the lower half [cm/s]
UTAU = 1.29            # friction velocity at the wavy wall [cm/s]
NU = 0.0088            # [cm2/s]
LAMBDA_CM = 5.08
PHASE = 0.144          # rad: surface = 0.1 h cos(2 pi x/lambda + 0.144)
DX_TRUE = PHASE / (2. * np.pi)     # 0.0229 lambda = 1.16 mm

# pressure.dat: the value nearest the wall per station: ((y - y_o)/h, P+)
P_WALL = {0.0: (0.103, -18.0), 0.1: (0.081, -4.13), 0.2: (0.035, -5.10),
          0.3: (-0.023, 2.19), 0.4: (-0.078, 6.47), 0.5: (-0.094, 10.4),
          0.6: (-0.058, 16.0), 0.7: (0.004, 8.68), 0.8: (0.060, 5.12),
          0.9: (0.093, -6.84)}
MEAN_DPDX_PLUS = 4.0   # P+ drop per wavelength, not removed in pressure.dat (+- 2)


def main():
    raw = json.load(open(os.path.join(HERE, 'hudson_case076_raw.json')))
    stations = {}
    for n in range(10):
        d = np.array(raw['xlam%02d.dat' % n])
        y, U, V, up, vp, uv = d.T
        xs = 0.1 * n
        stations['%.1f' % xs] = {
            'x_data': xs,
            'x_true': round(xs + DX_TRUE, 4),
            'y_over_H': (y / H_CM).round(6).tolist(),
            'U': (U / UB).round(5).tolist(),
            'W': (V / UB).round(5).tolist(),
            'uu': ((up / UB) ** 2).round(6).tolist(),
            'ww': ((vp / UB) ** 2).round(6).tolist(),
            'uw': (-uv / UB ** 2).round(6).tolist(),
        }
    xt = np.array([0.1 * n + DX_TRUE for n in range(10)])
    pplus = np.array([P_WALL[round(0.1 * n, 1)][1] for n in range(10)])
    corr = pplus + MEAN_DPDX_PLUS * (xt - xt[0])     # mean gradient added back
    fac = 2. * UTAU ** 2 / UB ** 2
    ref = {
        'meta': {
            'source': 'ERCOFTAC Classic Collection case 076, http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case076 (data set wb3 of Hudson 1993), downloaded 6 Oct 2026',
            'references': [
                'Hudson, J. D., Dykhno, L. and Hanratty, T. J. (1996). Turbulence production in flow over a wavy wall. Exp. Fluids 20, 257-265.',
                'Hudson, J. D. (1993). The effect of a wavy boundary on turbulent flow. PhD thesis, University of Illinois, Urbana.'],
            'geometry': 'H = 50.8 mm, lambda = 50.8 mm, delta = 2.54 mm (2 delta/lambda = 0.1), water channel 50.8 x 610 mm, 36 waves, LDV above the 31st',
            'flow': 'U_b = 12.0 cm/s (lower-half bulk velocity), Re_h = U_b h/nu = 3380, Re_H = 6760; u_tau at the wavy wall 1.29 cm/s (u_tau/U_b = %.4f, tau/(rho U_b^2) = %.5f)' % (UTAU / UB, (UTAU / UB) ** 2),
            'normalisation': 'velocities / U_b, second moments / U_b^2, heights / H; y = height above the local surface',
            'x': 'x_true = x_data + 0.144/(2 pi) = position relative to the crest (the data x = 0 is ~1.2 mm downstream of it)',
            'sign': 'uw = -u\'v\' of the files: <u\'w\'> in the usual convention (negative = downward momentum flux)',
            'note_re': 'Maass & Schumann (1994) used Re = 4780 with Um = 86.4 mm/s; the database gives U_b = 120 mm/s and Re_H = 6760 for the same data set.',
        },
        'utau_wavy_over_Ub': UTAU / UB,
        'tau_wavy_over_Ub2': (UTAU / UB) ** 2,
        'Re_H': 6760,
        'stations': stations,
        'surface_pressure': {
            'note': 'value nearest the wall per station from pressure.dat; Cp = 2 (P - P(x_true[0])) / (rho U_b^2) = 2 u_tau^2/U_b^2 (P+ - P+_0); "Cp_corrected" adds back the mean gradient of %g P+ per wave (+-2, pressure.dat header)' % MEAN_DPDX_PLUS,
            'x_true': xt.round(4).tolist(),
            'y_over_h': [P_WALL[round(0.1 * n, 1)][0] for n in range(10)],
            'P_plus': pplus.tolist(),
            'Cp': (fac * (pplus - pplus[0])).round(4).tolist(),
            'Cp_corrected': (fac * (corr - corr[0])).round(4).tolist(),
            'Cp_corrected_err': (fac * 2. * (xt - xt[0])).round(4).tolist(),
        },
    }
    with open(os.path.join(HERE, 'reference_hudson1996.json'), 'w') as f:
        json.dump(ref, f, indent=1)
    print('wrote reference_hudson1996.json: 10 stations, %d points; Cp max %.3f at x_true %.3f'
          % (sum(len(s['U']) for s in stations.values()), max(ref['surface_pressure']['Cp_corrected']),
             xt[np.argmax(ref['surface_pressure']['Cp_corrected'])]))


if __name__ == '__main__':
    main()
