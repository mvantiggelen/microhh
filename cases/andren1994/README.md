# andren1994: neutral Ekman layer (Andrén et al. 1994)

The LES intercomparison of a neutral, shear-driven boundary layer under a
stress-free lid: geostrophic wind 10 m/s, f = 1e-4 s⁻¹, z0 = 0.1 m, 40 × 40 × 40
grid, run to 10/f with the 7/f–10/f average. It tests the smag2/Mason set-up,
the MOST surface, the Coriolis forcing and the scalar flux at a coarse grid.
With `--surface ib` the surface is a flat IB DEM with the S5 production [IB]
block, so the IB wall model can be checked against MicroHH's own surface.

| file | what |
|---|---|
| `andren1994_setup.py` | writes a run directory (`.ini`, input `.nc`, DEM, `case_parameters.json`, `job_atos.sh`, copies of the scripts); `--perturb` adds the initial perturbations after `microhh init` |
| `andren1994_plot.py` | profiles and bulk numbers against the paper (PNG + table) |
| `reference_andren1994.json` | the paper's tables and digitised profiles |
| `cross_to_nc.py`, `write_cross_to_nc.py`, `microhh_tools.py` | cross-section conversion to NetCDF |

## Run

    python3 andren1994_setup.py --res small --endtime 7200 --npx 1 --npy 1 --outdir runs/smoke
    cd runs/smoke
    /path/to/microhh init andren1994
    python3 andren1994_setup.py --perturb --rundir .
    /path/to/microhh run andren1994

Paper grid on Atos: `python3 andren1994_setup.py --res paper --outdir $SCRATCH/andren1994/paper`,
copy the binary into it and `sbatch job_atos.sh` (it does init, perturb, run and the plots).

Options: `--res paper|small|fine`, `--surface flat|ib`, `--momentum staircase|vr`
(IB momentum, default staircase = case026, see `../README.md`), `--ib-set key=value`.

## Process the output

    python3 andren1994_plot.py --rundir runs/smoke --outdir runs/smoke/figs

u*/U_G, the surface stress angle and the profiles of wind, stress, variances
and the scalar against the paper (u*/U_G 0.040–0.045, angle 12.5–17°).

## References

- Andrén, A. et al. (1994), Large-eddy simulation of a neutrally stratified boundary layer: a comparison of four computer codes, *Q. J. R. Meteorol. Soc.* 120, 1457–1484. https://doi.org/10.1002/qj.49712052003
- Mason, P. J. and Thomson, D. J. (1992), Stochastic backscatter in large-eddy simulations of boundary layers, *J. Fluid Mech.* 242, 51–78. https://doi.org/10.1017/S0022112092002271
- van Heerwaarden, C. C. et al. (2017), MicroHH 1.0, *Geosci. Model Dev.* 10, 3145–3165. https://doi.org/10.5194/gmd-10-3145-2017
