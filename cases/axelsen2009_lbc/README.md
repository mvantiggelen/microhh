# axelsen2009_lbc: Axelsen P1 with open lateral boundaries

The katabatic case P1 of Axelsen & van Dop (2009) on a finite
**plateau – 5° slope – valley** section along x, with open boundaries at the
west (inflow) and east (outflow) edges, and periodic in y. The slope is an
immersed boundary (IB). Because the slope has a top and a foot, the domain
edges sit where the state is known (air at rest), so the open boundaries can
be exact.

Two grids: `--res coarse` (default; dx = dy = 100 m, dz = 10 m, 160³, the
resolution of a regional LES over a large ice sheet) or `--res fine`
(dx = 5 m, dz = 1 m). The slope-averaged profiles are compared with the paper
and with the PASTEX 1994 observations on the Pasterze glacier (Smeets et al.
1998). The coarse grid shows what a 10-m grid does to a thin katabatic jet.
The same flow in MicroHH's own rotated slope frame is in
`../axelsen2009_rotated`.

| file | what |
|---|---|
| `axelsen2009_lbc_input.py` | writes a run directory (`.ini`, DEM, flux map, lateral boundary files, input `.nc`, `case_parameters.json`) |
| `axelsen2009_lbc_init.py` | after `microhh init`: initial SFS TKE (Beare et al. 2006 profile), optional jet |
| `job.sh` | init, TKE, run and quick look; on a laptop or a SLURM cluster |
| `axelsen2009_lbc_check.py` | quick look at one output time (`check_<t>.png`, printed diagnostics) |
| `axelsen2009_lbc_analysis.py` | slope-averaged profiles, time series and metrics against the paper and observations (`figs/`) |
| `axelsen2009_lbc_compare.py` | several runs in one figure |
| `axelsen2009_lbc_turbulence.py`, `vortex_criteria.py` | TKE, spectra, vortex criteria (Q, λ2) |
| `axelsen2009_lbc_drag_budget.py` | form and skin drag along the slope |
| `observations_P1.json`, `pasterze_balloon_1994.json` | digitised paper profiles and PASTEX soundings |
| `digitize_observations.py`, `pasterze_balloon.py` | how those two files were made (need the source PDFs and data) |

The cross-section converter and `reference_axelsen2009.json` are taken from
`../axelsen2009/`, so keep the two folders side by side.

## Run

    python3 axelsen2009_lbc_input.py --outdir runs/P1_lbc
    cp /path/to/microhh runs/P1_lbc/
    cd runs/P1_lbc
    sbatch ../../job.sh          # SLURM cluster such as Snellius, or:
    bash ../../job.sh            # laptop / workstation

or the steps by hand:

    ./microhh init axelsen2009_lbc
    python3 ../../axelsen2009_lbc_init.py --rundir .
    ./microhh run axelsen2009_lbc

Both grids use 128 MPI tasks by default (coarse 8 × 16, fine 4 × 32). On a laptop,
choose fewer, e.g. `--npx 2 --npy 4`; the coarse grid then takes a few hours.

Other options: `--res fine`, `--bc flux|temp`, `--melt physical|uniform`,
`--sgs tke2|smag2`, `--momentum staircase|vr` (see `../README.md`),
`--slope-length`, `--no-periodic-y`, `--output minimal|full`.

**Output.** `--output minimal` (default) writes only what the analysis
needs, every 300 s: two xz planes of `u, w, thl, evisc` (+ `eviscs` with
tke2), the surface maps `ustar_ib, thl_fluxbot_ib, thl_sbot_ib`, and the
terrain-following planes `u_tf, thl_tf` at 5/10/20/50 m (fine grid
1/2/5/10/20 m). That is 21 files per output time. `--output full` also
writes yz planes, towers and time-averaged planes, which
`axelsen2009_lbc_turbulence.py`, `axelsen2009_lbc_drag_budget.py` and
`vortex_criteria.py` (with `--dump-every`) need. On a parallel file system
many small files can make the run spend most of its time writing.

## Process the output

    python3 axelsen2009_lbc_check.py --rundir runs/P1_lbc
    python3 axelsen2009_lbc_analysis.py --rundir runs/P1_lbc --tavg 3600
    python3 axelsen2009_lbc_compare.py --rundir runs/P1_lbc runs/P1_lbc_fine --label "dz 10 m" "dz 1 m"
    python3 ../axelsen2009/write_cross_to_nc.py --rundir runs/P1_lbc --case axelsen2009_lbc --window 600

## References

- Axelsen, S. L. and van Dop, H. (2009), Large-eddy simulation of katabatic winds. Part 1, *Acta Geophysica* 57, 803–836. https://doi.org/10.2478/s11600-009-0041-6
- Beare, R. J. et al. (2006), An intercomparison of large-eddy simulations of the stable boundary layer, *Boundary-Layer Meteorol.* 118, 247–272. https://doi.org/10.1007/s10546-004-2820-6
- Davies, H. C. (1976), A lateral boundary formulation for multi-level prediction models, *Q. J. R. Meteorol. Soc.* 102, 405–418. https://doi.org/10.1002/qj.49710243210
- Fedorovich, E. and Shapiro, A. (2009), Structure of numerically simulated katabatic and anabatic flows along steep slopes, *Acta Geophysica* 57, 981–1010. https://doi.org/10.2478/s11600-009-0027-4
- Smeets, C. J. P. P., Duynkerke, P. G. and Vugts, H. F. (1998), Turbulence characteristics of the stable boundary layer over a mid-latitude glacier. Part I, *Boundary-Layer Meteorol.* 87, 117–145. https://doi.org/10.1023/A:1000860406093
