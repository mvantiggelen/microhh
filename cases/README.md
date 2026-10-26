# Test cases for MicroHH with open boundaries and an immersed boundary (IB)

Test cases for the immersed-boundary (IB) terrain and the open lateral
boundaries of the **develop_openbc** branch of MicroHH
(https://github.com/mvantiggelen/microhh). Each case is a known flow from the
literature, set up so that the IB results can be compared with the paper,
with measurements or with MicroHH's own surface model.

One folder per case, with only what is needed to set it up, run it and
process the output: the Python scripts, the `.ini.base` templates, a job
script, a README and the reference data. Run directories (with the large
MicroHH output) go wherever you run the case.

| case | what |
|---|---|
| `andren1994` | neutral Ekman layer (Andrén et al. 1994), on MicroHH's surface or on a flat IB |
| `axelsen2009` | Axelsen & van Dop (2009) katabatic and anabatic slope flows, on an IB slope with terrain-following periodic boundaries |
| `axelsen2009_lbc` | Axelsen case P1 with open lateral boundaries: plateau – 5° slope – valley |
| `axelsen2009_rotated` | Axelsen case P1 in MicroHH's rotated slope frame: MicroHH's surface against a flat IB |
| `katabatic_idealised` | a Greenland katabatic flow (24 Jul 2026): IB slope with open boundaries against the rotated slope frame |
| `fedorovich2017_b13` | Fedorovich et al. (2017) case B13: a nocturnal low-level jet over a gentle IB slope |
| `maass1994_hill` | Maass & Schumann (1994): channel flow over a wavy wall |

## Running

Every case runs on a laptop (with a small grid, see the case README) and on a
cluster such as Snellius. The job scripts (`job.sh`, or the one a setup
script writes into the run directory) work both ways:

    bash job.sh        # laptop / workstation: uses mpirun
    sbatch job.sh      # SLURM cluster: uses srun; check the #SBATCH lines first

Optional environment variables: `MICROHH_ENV` (a script to source first, e.g.
the module loads of your build), `MPIRUN` (the MPI launcher) and, where
used, `MICROHH_BIN` (the binary). Python needs numpy, netCDF4 and
matplotlib (scipy for a few analysis scripts).

## How the IB applies the surface stress (`--momentum`)

Several cases have a `--momentum` option (`--variant` in maass1994_hill):

- `staircase` (default): the IB terrain is a staircase of grid cells. The
  surface stress of the Monin–Obukhov (MOST) wall model is applied as a
  momentum source in the air cells next to the terrain, the eddy viscosity
  there is rescaled to match it, w on the floor faces follows the sub-grid
  slope, and momentum is exchanged with the terrain only through MOST and
  pressure. This works with the develop_openbc branch.
- `vr`: velocity reconstruction. The velocity in the air cells next to the
  terrain is set from the MOST profile along the surface normal. This needs a
  MicroHH build with velocity reconstruction, which is not in develop_openbc.

## Shared files

`axelsen2009_lbc` uses `../axelsen2009/` (the cross-section converter and
`reference_axelsen2009.json`), and `axelsen2009_rotated` uses
`../axelsen2009_lbc/observations_P1.json`. Keep the folders side by side.
