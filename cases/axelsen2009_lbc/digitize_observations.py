#!/usr/bin/env python3
"""
Read the observed (and the published LES) profiles of case P1 out of the
VECTOR figures of

  Axelsen (2010), PhD thesis, Utrecht University, chapter 3 (= Axelsen and van
  Dop 2009, Part 1): Figs. 3.3a, 3.4, 3.5 and 3.7 (Pasterze site A1, Julian
  day 212, 10:30), and
  Smeets, Duynkerke and Vugts (1998), Bound.-Layer Meteorol. 87, 117-145:
  Fig. 3 (mean of 12 tethered-balloon soundings at A1) and Fig. 4d (mean
  mast profiles at A1), both for "no wind maximum below 13 m",

and write them to observations_P1.json:

    python3 digitize_observations.py --axelsen axelsen.pdf --smeets smeets1998.pdf [--out observations_P1.json]

The figures are vector graphics, so the markers and lines are read as drawn
(the axes are calibrated on the frames and tick marks; the tick marks are
checked against their labels). This is the plotted data, not the original
measurement files: mast values are 30-min means with the standard deviation
the paper draws as error bars, balloon values are single (instantaneous)
soundings.
"""
import argparse
import json

import numpy as np
import pdfplumber


def objects(page):
    out = []
    for o in page.curves + page.lines:
        p = np.array(o['pts'], float)
        # Marker position: the centre of its bounding box (a closed polygon
        # repeats its first point, so the mean of the points is biased).
        out.append(dict(p=p, n=len(p), cx=(p[:, 0].min() + p[:, 0].max()) / 2,
                        cy=(p[:, 1].min() + p[:, 1].max()) / 2,
                        w=np.ptp(p[:, 0]), h=np.ptp(p[:, 1]), fill=o.get('fill'),
                        gray=(o.get('non_stroking_color') or (0, 0, 0, 0))[-1]
                        if isinstance(o.get('non_stroking_color'), (tuple, list)) else None))
    return out


class Panel:
    """Frame in PDF points -> data. Log axes via log10 values."""
    def __init__(self, x0, x1, v0, v1, ytop, ybot, vtop, vbot, exclude=None, logy=False):
        self.x0, self.x1, self.v0, self.v1 = x0, x1, v0, v1
        self.yt, self.yb, self.vt, self.vb = ytop, ybot, vtop, vbot
        self.exclude, self.logy = exclude, logy

    def inside(self, x, y, pad=1.):
        if self.exclude and self.exclude[0] <= x <= self.exclude[1] and self.exclude[2] <= y <= self.exclude[3]:
            return False
        return self.x0 - pad <= x <= self.x1 + pad and self.yt - pad <= y <= self.yb + pad

    def X(self, px):
        return self.v0 + (np.asarray(px) - self.x0) / (self.x1 - self.x0) * (self.v1 - self.v0)

    def Y(self, py):
        v = self.vb + (np.asarray(py) - self.yb) / (self.yt - self.yb) * (self.vt - self.vb)
        return 10.**v if self.logy else v


def lines(objs, P, nmin=20):
    """Polylines with >= nmin points inside the panel, in data units, sorted by x."""
    out = []
    for o in objs:
        if o['n'] >= nmin and P.inside(o['cx'], o['cy']):
            out.append(np.c_[P.X(o['p'][:, 0]), P.Y(o['p'][:, 1])])
    return out


def axelsen(path):
    pdf = pdfplumber.open(path)
    res = {}

    # Fig. 3.7 (page 70 of the thesis PDF): z/H 0-1.5; tick marks verified.
    objs = objects(pdf.pages[69])
    pa = Panel(104.9, 240.8, 2., 18., 82.8, 210.5, 1.5, 0., exclude=(216.6, 240.1, 84.0, 114.0))
    pb = Panel(268.0, 403.1, -0.8, 0.6, 82.8, 209.2, 1.5, 0.)
    pc = Panel(104.9, 240.8, -80., 10., 260.7, 387.7, 1.5, 0.)
    pd = Panel(268.0, 405.3, -1.2, 0., 260.8, 388.3, 1.5, 0.)

    def mast(P, ebars=True):
        sq = sorted([(float(P.Y(o['cy'])), float(P.X(o['cx']))) for o in objs
                     if o['n'] == 7 and o['w'] < 3 and o['h'] < 4 and P.inside(o['cx'], o['cy'])])
        out = []
        for z, v in sq:
            sd = np.nan
            if ebars:   # the horizontal error bar at the same height, not an axis tick
                for o in objs:
                    if (o['n'] == 2 and o['h'] < 0.3 and 0.5 < o['w'] < 0.5 * (P.x1 - P.x0)
                            and abs(float(P.Y(o['cy'])) - z) < 0.012 and P.x0 + 2 < o['cx'] < P.x1 - 2):
                        lo, hi = sorted(P.X(o['p'][:, 0]))
                        if lo - 1e-6 <= v <= hi + 1e-6:
                            sd = 0.5 * (hi - lo)
            out.append(dict(z_over_H=z, value=v, sd=None if np.isnan(sd) else sd))
        return out

    def balloon(P):
        return sorted([(float(P.Y(o['cy'])), float(P.X(o['cx']))) for o in objs
                       if o['n'] == 5 and o['fill'] and o['w'] < 3 and o['h'] < 3 and P.inside(o['cx'], o['cy'])])

    res['fig3_7'] = dict(
        caption='Axelsen (2010) Fig. 3.7 = Axelsen and van Dop (2009) P1: Pasterze A1, JD 212; '
                'mast 30-min means (+- sd), balloon instantaneous, LES (their run P1)',
        u_over_V=dict(mast=mast(pa), balloon=balloon(pa), les=lines(objs, pa)[0].tolist()),
        tau_over_V2=dict(mast=mast(pb, False), les=lines(objs, pb)[0].tolist()),
        b_over_B=dict(mast=mast(pc), balloon=balloon(pc), les=lines(objs, pc)[0].tolist()),
        F_over_VB=dict(mast=mast(pd, False), les=lines(objs, pd)[0].tolist()))

    # Fig. 3.3a (page 64): sounding JD 212 10:30, x: wind (m/s) and theta (deg C) on one axis.
    objs = objects(pdf.pages[63])
    p33 = Panel(108.2, 243.0, -5., 15., 82.8, 209.2, 600., 0., exclude=(108.9, 119.6, 84.0, 120.0))
    inset = (158.1, 215.6, 92.3, 149.1)
    u, th = [], []
    for o in objs:
        if (o['n'] == 5 and o['fill'] and o['w'] < 2 and p33.inside(o['cx'], o['cy'])
                and not (inset[0] <= o['cx'] <= inset[1] and inset[2] <= o['cy'] <= inset[3])):
            pt = (float(p33.Y(o['cy'])), float(p33.X(o['cx'])))
            (th if o['gray'] is not None and o['gray'] < 0.9 else u).append(pt)
    res['fig3_3a'] = dict(caption='Axelsen (2010) Fig. 3.3a: balloon sounding at Pasterze A1, JD 212 10:30',
                          z_m_u_ms=sorted(u), z_m_theta_C=sorted(th))

    # Fig. 3.4 (page 65): (a) UD/(VH) > 0 and BD/(BH) < 0, (b) u/V at z/H = 0.06, 0.15, 1.01, 2.01.
    objs = objects(pdf.pages[64])
    p4a = Panel(82.3, 218.1, 0., 3., 221.4, 348.2, 20., -20.)
    p4b = Panel(242.3, 381.0, 0., 3., 221.5, 347.3, 25., 0., exclude=(351.9, 381.0, 217.7, 261.))
    la = sorted(lines(objs, p4a, 50), key=lambda c: -np.mean(c[:, 1]))
    lb = sorted(lines(objs, p4b, 50), key=lambda c: -np.mean(c[len(c)//2:, 1]))
    res['fig3_4'] = dict(caption='Axelsen (2010) Fig. 3.4, LES P1 against t/tau_osc',
                         UD_over_VH=la[0].tolist(), BD_over_BH=la[1].tolist(),
                         u_over_V={'0.15': lb[0].tolist(), '0.06': lb[1].tolist(),
                                   '1.01': lb[2].tolist(), '2.01': lb[3].tolist()})

    # Fig. 3.5 (page 66): z_inv/H and b_s/B against t/tau_osc.
    objs = objects(pdf.pages[65])
    p5a = Panel(119.2, 237.9, 0., 3., 209.1, 333.6, 1., 0.)
    p5b = Panel(282.3, 400.9, 0., 3., 209.1, 333.4, -60., -80.)
    res['fig3_5'] = dict(caption='Axelsen (2010) Fig. 3.5, LES P1 against t/tau_osc',
                         zinv_over_H=lines(objs, p5a, 50)[0].tolist(),
                         bs_over_B=lines(objs, p5b, 50)[0].tolist())
    return res


def smeets(path):
    pdf = pdfplumber.open(path)
    objs = objects(pdf.pages[7])        # journal page 124: Fig. 3
    p3a = Panel(147.9, 260.9, 2., 8., 147.1, 260.1, 400., 0.)
    p3b = Panel(291.8, 404.8, 8., 14., 147.1, 260.1, 400., 0.)
    ua = lines(objs, p3a, 20)[0]
    tb = lines(objs, p3b, 20)[0]
    objs = objects(pdf.pages[9])        # journal page 126: Fig. 4d, log z
    # Log z: major ticks 10^0 at y = 377.5 pt and 10^1 at 315.0 pt (62.5 pt per decade).
    p4d = Panel(283.7, 400.8, 3., 10., 304.0, 421.1, (377.5 - 304.0) / 62.5, (377.5 - 421.1) / 62.5, logy=True)
    mk = [(float(p4d.Y(o['cy'])), float(p4d.X(o['cx']))) for o in objs
          if o['n'] == 5 and o['w'] < 4 and p4d.inside(o['cx'], o['cy'])]
    mk.sort()
    # Two columns of 8 circles: U (left) and T (right) at the same heights.
    U, T = [], []
    for k in range(0, len(mk), 2):
        a, b = sorted(mk[k:k+2], key=lambda t: t[1])
        U.append(a); T.append(b)
    return dict(
        fig3=dict(caption='Smeets et al. (1998) Fig. 3: mean of 12 tethered-balloon soundings at A1, '
                          'katabatic conditions with NO wind maximum below 13 m (large-scale forcing)',
                  z_m_speed_ms=[(float(z), float(v)) for v, z in ua], z_m_T_C=[(float(z), float(v)) for v, z in tb]),
        fig4d=dict(caption='Smeets et al. (1998) Fig. 4d: mean mast profiles at A1 (22 composite 1-h runs, '
                           'no wind maximum below 13 m)', z_m_U_ms=U, z_m_T_C=T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axelsen', required=True, help='Axelsen (2010) PhD thesis PDF')
    ap.add_argument('--smeets', required=True, help='Smeets et al. (1998) PDF')
    ap.add_argument('--out', default='observations_P1.json')
    a = ap.parse_args()
    out = dict(
        _source=('Digitised from the vector figures: Axelsen, S. L. (2010), Large-eddy simulation and '
                 'analytical modelling of katabatic winds, PhD thesis, Utrecht University, ch. 3 '
                 '(= Axelsen and van Dop 2009, Acta Geophys. 57, 803-836, doi:10.2478/s11600-009-0041-6); '
                 'Smeets, Duynkerke and Vugts (1998), Bound.-Layer Meteorol. 87, 117-145, '
                 'doi:10.1023/A:1000860406093. Scaled values use the paper\'s V, B, H for P1.'),
        axelsen=axelsen(a.axelsen), smeets=smeets(a.smeets))
    json.dump(out, open(a.out, 'w'), indent=1)
    f = out['axelsen']['fig3_7']
    print('Fig 3.7 mast u/V:', [(round(m['z_over_H'], 3), round(m['value'], 2), m['sd'] and round(m['sd'], 2))
                                for m in f['u_over_V']['mast']])
    print('Fig 3.7 mast b/B:', [(round(m['z_over_H'], 3), round(m['value'], 1), m['sd'] and round(m['sd'], 1))
                                for m in f['b_over_B']['mast']])
    print('Fig 3.7 mast tau/V2:', [(round(m['z_over_H'], 3), round(m['value'], 3)) for m in f['tau_over_V2']['mast']])
    print('Fig 3.7 mast F/VB:', [(round(m['z_over_H'], 3), round(m['value'], 3)) for m in f['F_over_VB']['mast']])
    print('balloon u/V', len(f['u_over_V']['balloon']), 'b/B', len(f['b_over_B']['balloon']))
    s = out['axelsen']['fig3_3a']
    print('Fig 3.3a: u', len(s['z_m_u_ms']), 'theta', len(s['z_m_theta_C']))
    m = out['smeets']['fig4d']
    print('Smeets 4d U:', [(round(z, 2), round(v, 2)) for z, v in m['z_m_U_ms']])
    print('Smeets 4d T:', [(round(z, 2), round(v, 2)) for z, v in m['z_m_T_C']])
    print('wrote', a.out)


if __name__ == '__main__':
    main()
