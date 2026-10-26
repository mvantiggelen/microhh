# andren1994_ib: neutral Ekman layer (Andrén et al. 1994), also on a flat immersed boundary

The classic LES intercomparison of a neutral boundary layer driven by a
geostrophic wind of 10 m/s, with f = 1e-4 s⁻¹, z0 = 0.1 m and a stress-free
lid. The grid is 40 × 40 × 40 and the run lasts 10/f, with the statistics
averaged over 7/f–10/f. The case tests the Smagorinsky/Mason closure, the
surface model, the Coriolis forcing and a passive scalar flux.

With `--surface ib` the same case is run over a **flat immersed boundary
(IB)**: the domain floor is buried under a flat DEM, and the surface is
handled by the IB wall model instead of MicroHH's own surface model. Comparing
the two runs shows whether the IB wall model reproduces the normal surface.

| file | what |
|---|---|
| `andren1994_setup.py` | writes a run directory (`.ini`, input `.nc`, DEM, `case_parameters.json`, `job.sh`, copies of the scripts); `--perturb` adds the initial perturbations after `microhh init` |
| `andren1994_plot.py` | profiles and bulk numbers against the paper (PNG + table) |
| `reference_andren1994.json` | the paper's tables and digitised profiles |
| `cross_to_nc.py`, `write_cross_to_nc.py`, `microhh_tools.py` | conversion of the cross-sections to netCDF |

MicroHH's original set-up of this case is in `../andren1994`; it runs on
MicroHH's own surface only. This folder is separate from it:
`andren1994_setup.py` writes a complete run directory of its own (the MicroHH
case name is still `andren1994`, so the files are `andren1994.ini` and so on).

## Run

On a laptop (small grid, about a minute):

    python3 andren1994_setup.py --res small --endtime 7200 --npx 1 --npy 1 --outdir runs/small
    cd runs/small
    cp /path/to/microhh .
    ./microhh init andren1994
    python3 andren1994_setup.py --perturb --rundir .
    ./microhh run andren1994

The generated `job.sh` does the same steps (init, perturbations, run, plots).
It runs on a laptop with `bash job.sh` and on a SLURM cluster such as
Snellius with `sbatch job.sh` (check its `#SBATCH` lines first):

    python3 andren1994_setup.py --res paper --outdir runs/paper
    cp /path/to/microhh runs/paper/ && cd runs/paper && sbatch job.sh

Options:
- `--res paper|small|fine`: the grid (the domain is always the paper's);
- `--surface flat|ib`: MicroHH's surface model, or a flat immersed boundary;
- `--momentum staircase|vr` (with `--surface ib`): how the IB applies the
  surface stress, see below;
- `--ib-set key=value`: change any `[IB]` setting, e.g. `--ib-set sw_wall_sample=false`;
- `--npx`, `--npy`: the MPI decomposition (default: chosen automatically).

## The flat-IB variant

With `--surface ib` the floor is buried 4 levels deep under a flat DEM, so
that the air column above the surface still has the paper's 1500 m. The `[IB]`
block uses the recommended IB settings, with the paper's z0 and scalar flux.
The IB needs `[grid] utrans = 0` (it works in the model frame), so the flat
twin to compare with is

    python3 andren1994_setup.py --advec 2i5 --utrans 0 --outdir runs/flat_2i5

## How the IB applies the surface stress (`--momentum`)

- `staircase` (default): the IB terrain is a staircase of grid cells. The
  surface stress of the Monin–Obukhov (MOST) wall model is applied as a
  momentum source in the air cells next to the terrain, the eddy viscosity
  there is rescaled to match it, w on the floor faces follows the sub-grid
  slope, and momentum is exchanged with the terrain only through MOST and
  pressure. This works with the develop_openbc branch.
- `vr`: velocity reconstruction. The velocity in the air cells next to the
  terrain is set from the MOST profile along the surface normal. This needs a
  MicroHH build with velocity reconstruction, which is not in develop_openbc.

## Process the output

    python3 andren1994_plot.py --rundir runs/small --outdir runs/small/figs
    python3 andren1994_plot.py --rundir runs/flat_2i5 --compare runs/ib --labels "flat" "flat IB" --outdir figs

u*/U_G, the surface stress angle and the profiles of wind, stress, variances
and the scalar against the paper (u*/U_G 0.040–0.045, angle 12.5–17°).

## References

- Andrén, A. et al. (1994), Large-eddy simulation of a neutrally stratified boundary layer: a comparison of four computer codes, *Q. J. R. Meteorol. Soc.* 120, 1457–1484. https://doi.org/10.1002/qj.49712052003
- Mason, P. J. and Thomson, D. J. (1992), Stochastic backscatter in large-eddy simulations of boundary layers, *J. Fluid Mech.* 242, 51–78. https://doi.org/10.1017/S0022112092002271
- van Heerwaarden, C. C. et al. (2017), MicroHH 1.0, *Geosci. Model Dev.* 10, 3145–3165. https://doi.org/10.5194/gmd-10-3145-2017
