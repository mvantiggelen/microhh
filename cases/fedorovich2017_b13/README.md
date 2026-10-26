# fedorovich2017_b13: nocturnal low-level jet over a gentle slope (Fedorovich et al. 2017, B13)

Case B13 of Fedorovich, Gibbs & Shapiro (2017): a 0.18° slope under a
10 m/s geostrophic wind, with a daytime surface buoyancy that switches to a
night-time value at sunset, so that a low-level jet forms from the inertial
oscillation and the slope. The paper uses a rotated (slope-following) frame;
here the slope is an **immersed boundary (IB) on a gravity-aligned grid**.
The lateral boundaries are terrain-following periodic: every edge is fed from
the opposite edge at the same height above the surface. `--alpha 0` gives
the flat twin, case B11.

| file | what |
|---|---|
| `fedorovich2017_b13_input.py` | writes a run directory from `fedorovich2017_b13.ini.base` (`.ini`, DEM, day/night surface temperature maps, input `.nc`) |
| `fedorovich2017_b13.ini.base` | the .ini template (IB with the staircase momentum method) |
| `fedorovich2017_b13_analysis.py` | jet speed and height, hodographs and profiles against the paper |
| `reference_b13.json` | the paper's numbers |
| `cross_to_nc.py`, `write_cross_to_nc.py`, `microhh_tools.py` | conversion of the cross-sections to netCDF |

Needs MicroHH from the develop_openbc branch. The `[IB]` block uses the
staircase momentum method only (`sw_momentum_flux`, `sw_strain_most`,
optionally `--kinematic`): the surface stress of the MOST wall model is
applied as a momentum source in the air cells next to the terrain.

## Run

The default grid (dx = 8 m, dz = 4 m, 4 × 1 km) has 26 M cells, which is a
job for a cluster. A small version for a laptop:

    python3 fedorovich2017_b13_input.py --dx 16 --lx 1024 --ly 512 --endtime 3600 --outdir run_small
    cd run_small && cp /path/to/microhh . && ./microhh init fedorovich2017_b13 && ./microhh run fedorovich2017_b13

The full case, with MPI (e.g. 16 × 8 = 128 tasks):

    python3 fedorovich2017_b13_input.py --npx 16 --npy 8 --outdir run_b13
    python3 fedorovich2017_b13_input.py --alpha 0 --npx 16 --npy 8 --outdir run_b11   # flat twin
    cd run_b13 && mpirun -np 128 ./microhh init fedorovich2017_b13 && mpirun -np 128 ./microhh run fedorovich2017_b13

On a SLURM cluster such as Snellius, put the last two commands in a job
script with `#SBATCH --ntasks=128` and `srun` instead of `mpirun -np 128`.
`microhh_tools.py` is taken from `$MICROHH_ROOT/python` if `MICROHH_ROOT` is
set, else from this folder.

Options: `--wall-slope`, `--kinematic`, `--sgs smag2|aniso|tke2`, `--advec 2i5|2i62`,
`--lbc tfperiodic|nudge --precursor <stats.nc>`, `--z0m/--z0h`.

## The slope is sub-grid

The terrain drops only 13 m over the 4-km default domain, about 3 dz, so the
IB staircase has only 3 steps: the floors in between are flat, the air near
the surface cannot follow the terrain down, and the slope terms
N² u sin α and b sin α are largely missing. The generator warns when there are
fewer than 3 steps. Remedies:

- `--kinematic`: w on the floor faces follows the sub-grid slope (w = u·∇η);
- a longer domain (`--lx`) or a finer dz, so that more steps are resolved.

The paper's grid (`--dx 4 --lx 1024`) has no step at all.

## Process the output

    python3 ../fedorovich2017_b13_analysis.py --rundir . --outdir figs
    python3 ../write_cross_to_nc.py --rundir . --case fedorovich2017_b13 --window 600

## References

- Fedorovich, E., Gibbs, J. A. and Shapiro, A. (2017), Numerical study of nocturnal low-level jets over gently sloping terrain, *J. Atmos. Sci.* 74, 2813–2834. https://doi.org/10.1175/JAS-D-17-0013.1
- van Heerwaarden, C. C. et al. (2017), MicroHH 1.0, *Geosci. Model Dev.* 10, 3145–3165. https://doi.org/10.5194/gmd-10-3145-2017
