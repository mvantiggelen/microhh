# axelsen2009: katabatic and anabatic slope flows (Axelsen & van Dop 2009)

The idealised slope-flow cases A1, P1, P2, V1 and V2 of Axelsen & van Dop
(2009, Table 1). The paper uses a rotated, periodic slope frame. Here the
slope is an **immersed boundary (IB)** on an ordinary gravity-aligned grid,
and the lateral boundaries are made "terrain-following periodic": every edge
is fed from the opposite edge at the same height above the surface, so the
slope behaves as if it were infinite. The surface is either a prescribed
buoyancy flux (`--bc flux`, as the paper) or a prescribed surface
temperature (`--bc temp`). The jet speed, jet height and buoyancy deficit are
compared with the paper.

Two grids: the paper's (`--res paper`, default) or a coarse grid of dx = 100 m
and dz = 10 m (`--res coarse`), the resolution of a regional LES over a large
ice sheet. The paper's thin katabatic jet (a few metres deep) is below the
first level of the coarse grid, so `--res coarse` shows what that resolution
does to a jet; it does not reproduce the paper.

| file | what |
|---|---|
| `axelsen2009_input.py` | writes a run directory from `axelsen2009.ini.base` (`.ini`, input `.nc`, DEM, surface maps, `case_parameters.json`, cross-section scripts) |
| `axelsen2009.ini.base` | the .ini template, with the recommended `[IB]` settings |
| `axelsen2009_init.py` | after `microhh init`: initial SFS TKE, `--init-jet`, or a warm start |
| `axelsen2009_analysis.py` | slope-normal profiles and metrics against the paper (`figs/summary.md`) |
| `job.sh` | init, `axelsen2009_init.py`, run and analysis; on a laptop or a SLURM cluster |
| `reference_axelsen2009.json` | the paper's Table 1 and digitised profiles |
| `cross_to_nc.py`, `write_cross_to_nc.py`, `microhh_tools.py` | conversion of the cross-sections to netCDF |

## Run

    python3 axelsen2009_input.py --case P1 --bc flux --outdir runs/P1_flux
    cp /path/to/microhh runs/P1_flux/
    cd runs/P1_flux
    sbatch ../../job.sh          # SLURM cluster such as Snellius, or:
    bash ../../job.sh            # laptop / workstation

or the steps by hand:

    ./microhh init axelsen2009 && python3 ../../axelsen2009_init.py --rundir . && ./microhh run axelsen2009

The generator prints the expected cost. The P1 paper grid needs about 8 h on
128 cores. For a quick look on a laptop, use `--res coarse` or a shorter run
(`--n-tau 0.5`). `--all --outdir runs` writes all ten runs.

`--momentum staircase` (default) or `vr` sets how the IB applies the wall
stress, see `../README.md`.

`--bc temp` cannot start from rest: with no wind there is no MOST flux. Start
it from a developed `--bc flux` run (`WARM_FROM=<run dir> WARM_TIME=<s>` in
`job.sh`).

## Process the output

    python3 axelsen2009_analysis.py --rundir runs/P1_flux --outdir figs
    python3 write_cross_to_nc.py --rundir runs/P1_flux --case axelsen2009 --window 600

## References

- Axelsen, S. L. and van Dop, H. (2009), Large-eddy simulation of katabatic winds. Part 1, *Acta Geophysica* 57, 803–836. https://doi.org/10.2478/s11600-009-0041-6
- Fedorovich, E. and Shapiro, A. (2009), Structure of numerically simulated katabatic and anabatic flows along steep slopes, *Acta Geophysica* 57, 981–1010. https://doi.org/10.2478/s11600-009-0027-4
- Beare, R. J. et al. (2006), An intercomparison of large-eddy simulations of the stable boundary layer, *Boundary-Layer Meteorol.* 118, 247–272. https://doi.org/10.1007/s10546-004-2820-6
