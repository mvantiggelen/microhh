# katabatic_idealised: a katabatic flow down a Greenland ice slope

An idealised test of the immersed boundary (IB) in a realistic katabatic
regime. The slope, the air above it and the katabatic layer are taken from a
regional MicroHH simulation of the west Greenland ice-sheet margin on
24 July 2026, driven by the regional climate model RACMO. The data are in
`greenland_20260724_inputs.json`, so no RACMO data is needed. The case
complements `../axelsen2009_lbc`, which uses the paper's thin Alpine-glacier
jet.

The case has two runs:

- `rotated/`: the reference, MicroHH's own rotated, periodic slope frame
  (`swthermo=buoy`, as in `cases/prandtlslope` of MicroHH): an infinite slope
  with MicroHH's surface model;
- `ib/`: the test, a finite slope (plateau – slope – valley) as an IB on an
  ordinary gravity-aligned grid, with open boundaries in x and periodic in y.

Everything physical is the same in both runs by default, so what differs is
the terrain (finite vs infinite slope) and the surface method (the IB vs
MicroHH's own surface).

> **The two runs are made and run ONE AFTER THE OTHER** (the default
> `--inflow rotated`). The inflow of the IB run, and the katabatic layer it
> starts with, are the **equilibrium profiles of the finished rotated run**.
> So:
> 1. stage 1 writes `rotated/` only;
> 2. run `rotated/` **to its end**;
> 3. stage 2 writes `ib/` from that output (`job.sh` does this itself);
> 4. run `ib/`.
>
> The `ib/` directory does not exist until stage 2, so it cannot be started
> too early. Why: when the IB run was given the RACMO profile as its inflow,
> the air crossed the 16-km slope in ~35 min, a sixth of the oscillation
> period, so the IB jet stayed at the inflow's (8 m/s at 40 m) instead of
> finding its own balance (6.2 m/s at 15 m in the rotated run). `--inflow
> racmo` keeps that set-up (both directories at once).

## The case

- **Terrain** (`--terrain constant`, the default): a uniform slope at
  α = 1.93°, the mean surface gradient of the ice in the regional simulation
  (a drop from 691 to 178 m over 16 km). It ends at the height of the ice
  margin (233 m amsl), with the breaks smoothed over 500 m (`--break-width`).
  The flow runs towards +x. A flat 2.5 km plateau sits in front (the inflow)
  and a flat 5 km valley behind (the outflow). The terrain is uniform in y.
  - Default: 24 km long, plateau 770 m amsl, a 538 m drop, slope from x = 2.5
    to 19 km; the weather station of the regional simulation (536 m) is at
    x = 9.7 km.
  - `--alpha`, `--slope-length` change the angle and the length.
  - `--terrain transect`: the real terrain along x through the weather station
    (from the 400-m and 100-m DEMs of the regional simulation, smoothed over
    300 m), 43.2 km from 1004 m down to the ice margin, mean slope 1.21°.
- **Surface:** a prescribed (Dirichlet) ice surface temperature, z0m = 2 cm.
- **Kept the same as the rotated reference** (the defaults):

  | option | default (= rotated run) | realistic alternative, IB run only |
  |---|---|---|
  | `--surface` | `deficit`: θ_s = θ_a(z_s) − 10.7 K everywhere, the melting deficit at the slope's mid-height | `melt`: T_s = 273.15 K everywhere |
  | `--humidity` | `off`: q_t = 0 (the rotated run's buoyancy is dry) | `racmo`: RACMO q_t, saturated surface (needed with `--radiation`) |
  | `--z0h-model` | `fixed`: z0h = `--z0h` = 5×10⁻⁴ m in both | `yang2008`: the ice model of Yang et al. (2008) |
  | `--basestate` | `boussinesq`: θ_v0 = θ_r (288.1 K), ρ = 1, as `thermo buoy` | `anelastic` |
  | noise | 0.1 m/s and 0.1 K, ×(1 − n/100 m)², in the lowest 100 m above the surface | |

  Also the same: the slope angle, N = (g/θ_r · Γ)^½, z0m, the katabatic layer,
  the closure (anisotropic Smagorinsky), dx, dz, the y width, viscosity,
  cflmax, dtmax, the top buffer, no Coriolis, no large-scale pressure gradient.
- **State:** RACMO's air at the ice-sheet (east) boundary of the regional
  simulation at 15 UTC. The ambient θ is a linear fit 200–1000 m above the
  surface, extrapolated downwards (Γ = 6.6 K/km, N = 0.0150 s⁻¹). The rotated
  run starts from, and `--inflow racmo` uses, a fit to the RACMO katabatic
  layer: u(n) = U_j (n/n_j) exp(1 − n/n_j) with U_j = 7.5 m/s and n_j = 28 m,
  and θ'(n) = Δθ₀ exp(−n/h_θ) with Δθ₀ = −5.6 K and h_θ = 66 m (n the height
  above the local surface).
- **Boundaries of `ib/`:** west: constant inflow (ambient air plus the
  katabatic layer above the plateau); east: the same profiles above the
  valley floor (`--outflow jet`) or the ambient air at rest (`--outflow
  rest`); y periodic. The katabatic layer is also laid over the whole terrain
  at t = 0 (`katabatic_idealised_init.py`).
- **Model:** the recommended IB settings (staircase momentum method, MOST wall
  model, wall sampling, Mason wall damping, anisotropic Smagorinsky) on
  dx = 100 m, dz = 10 m, 240 × 32 × 160 cells, 16 × 8 = 128 MPI tasks. The
  rotated run has 64 × 32 × 100 cells on 4 × 8 tasks.
- **Radiation:** ready but off (`--radiation off`); `--radiation lw` or
  `lwsw` (sun fixed at the RACMO time) switches RRTMGP on, and `job.sh` then
  links the coefficient files (`RRTMGP_DATA`).
- **Length:** 3 periods of the katabatic oscillation 2π/(N sin α):
  τ = 12 505 s, so 39 600 s (11 h).

**Output** (every 600 s, few files): two xz planes of `u, w, thl`; the
surface maps `hfss_ib, hfls_ib, thl_fluxbot_ib, ustar_ib, obuk_ib,
thl_sbot_ib`; `u_tf, thl_tf` at 5–100 m above the surface; 4 columns (top,
mid-slope, the weather station, foot); statistics.

| file | what |
|---|---|
| `katabatic_idealised_input.py` | stage 1: `<outdir>/rotated` and `generator_args.json`; stage 2: `<outdir>/ib` and `inputs.png` (terrain, inflow against RACMO). `--inflow racmo`: both at once |
| `katabatic_idealised_init.py` | after `microhh init` in `ib/`: the initial katabatic layer over the whole terrain |
| `katabatic_idealised_compare.py` | IB against rotated: profiles at the stations, the jet and the heat flux along the slope, time series (`compare/`) |
| `greenland_20260724_inputs.json` | the terrain and the RACMO profiles of 24 Jul 2026 |
| `extract_inputs.py` | how the JSON file was made, from a nested RACMO-driven MicroHH set-up (only needed for another day or place) |
| `job.sh` | for either directory; at the end of `rotated/` it runs stage 2 (`KAT_STAGE2=1`, default); on SLURM, `KAT_CHAIN=1` also submits `ib/` |

## Run (one after the other)

On a SLURM cluster such as Snellius (check the `#SBATCH` lines of `job.sh`
first):

    # stage 1: rotated/ (and generator_args.json); ib/ is not written yet
    python3 katabatic_idealised_input.py --outdir runs/test [options]
    cd runs/test/rotated && ln -s /path/to/microhh . && sbatch ../../../job.sh
    #   when it ends, the job runs stage 2 itself: ../ib is written and the
    #   binary linked there. To have it submit ib/ too:
    #   sbatch --export=ALL,KAT_CHAIN=1 ../../../job.sh
    # stage 2 by hand (if KAT_STAGE2=0, or to change options of the IB run only):
    python3 katabatic_idealised_input.py --outdir runs/test [IB-only options]
    cd ../ib && sbatch ../../../job.sh

On a laptop, a coarse version (a few minutes each run); `bash job.sh` works
as well:

    python3 katabatic_idealised_input.py --outdir runs/small --dx 400 --dz 20 --ly 3200 --npx 1 --npy 1 \
            --n-sponge 4 --rot-npx 1 --rot-npy 1 --endtime 3600
    cd runs/small/rotated && ln -s /path/to/microhh . && ./microhh init katabatic_rotated \
            && ./microhh run katabatic_rotated
    cd ../../.. && python3 katabatic_idealised_input.py --outdir runs/small          # stage 2
    cd runs/small/ib && ln -s /path/to/microhh . && ./microhh init katabatic_idealised \
            && python3 ../../../katabatic_idealised_init.py && ./microhh run katabatic_idealised

Stage 2 reads the stage-1 options from `generator_args.json`; options given
again override them. It refuses any option that would make the IB run differ
from the rotated run it takes its inflow from (slope angle, θ_r, surface
deficit, dx, dz, z0m, z0h): make both again in a new `--outdir` for those.
Options of the IB run only are fine at stage 2: `--lx`, `--ly`, `--zsize`,
`--radiation`, `--humidity`, `--surface`, `--z0h-model`, `--basestate`,
`--momentum`, `--outflow`, `--npx`, `--npy`, `--out-every`. The rotated run
must have reached its end time (else `--rot-window` averages what is there,
with a warning).

Options:
- **inflow and stages:** `--inflow rotated|racmo`, `--stage auto|rotated|ib|all`,
  `--rot-window` (s; default the last τ of the rotated run);
- **slope:** `--terrain constant|transect`, `--alpha` (deg), `--slope-length`
  (m), `--break-width`;
- **domain and resolution:** `--dx`, `--dz` (both runs); IB `--lx`, `--ly`
  (default 3200 m), `--zsize`, `--min-free-air` (500 m), or `--plateau`,
  `--valley`, `--free-air`, `--jtot`; rotated `--rot-lx` (6400 m),
  `--rot-ly` (3200 m), `--rot-lz` (1000 m); MPI tasks `--npx`, `--npy`,
  `--rot-npx`, `--rot-npy`;
- **checks** (the generator stops and prints the size it needs): the slope
  must end at least the outflow sponge + 1 km before the east edge (`--lx`);
  the terrain top must leave `--min-free-air` below the buffer (`--zsize`);
  every length must be a multiple of dx or dz and every cell count must divide
  over the tasks; 0 < α ≤ 30° (warning above 15°: the IB wall model is
  untested there); the slope longer than twice the break width; the plateau
  longer than the sponge;
- **radiation:** `--radiation off|lw|lwsw` (needs `--humidity racmo`), `--albedo`;
- **physics:** `--surface`, `--humidity`, `--z0h-model`, `--z0h`, `--basestate`,
  `--rnd-depth` (see the table), `--momentum staircase|vr` (see `../README.md`),
  `--outflow jet|rest`, `--n-tau`, `--endtime`;
- **RACMO:** `--racmo-time` (s after 03 UTC, default 43200 = 15 UTC);
  `--jet-speed`, `--jet-height`, `--dtheta0`, `--theta-depth` (the fit, used
  by the rotated start and `--inflow racmo`); `--alpha-ref` (rotated).

Examples:

    --alpha 5 --slope-length 8000                # steeper and shorter (the domain follows)
    --dx 50 --dz 5                               # double resolution, both runs (same sizes in m)
    --ly 6400 --zsize 2080                       # wider and deeper IB domain
    --radiation lwsw --humidity racmo            # RRTMGP in the IB run (stage 2 is enough)

## Process the output

    python3 katabatic_idealised_compare.py --rundir runs/test      # default window: the last tau_osc

`job.sh` runs it at the end of `ib/`. The dotted inflow line in the figure is
the profile the IB run was given (the rotated equilibrium, or the RACMO fit
with `--inflow racmo`).

## First result

With `--inflow racmo` and the other defaults (39 600 s; IB run 59 min on 128
cores, rotated run 4 min): the rotated run oscillates with τ and settles at
6.2 m/s at 15 m, with a heat flux H of −77 W/m². The IB run is held by its
inflow: 8 m/s at 40 m, H −54 W/m² (70 %), with an exchange per unit wind as
in MOST. The gap is the weaker wind in the lowest 20 m: the form drag of the
staircase steps is 1.2–2 times the skin stress. Hence the default
`--inflow rotated`.

## References

- McNider, R. T. (1982), A note on velocity fluctuations in drainage flows, *J. Atmos. Sci.* 39, 1658–1660. https://doi.org/10.1175/1520-0469(1982)039<1658:ANOVFI>2.0.CO;2 (the period 2π/(N sin α))
- Fedorovich, E. and Shapiro, A. (2009), Structure of numerically simulated katabatic and anabatic flows along steep slopes, *Acta Geophys.* 57, 981–1010. https://doi.org/10.2478/s11600-009-0027-4
- Axelsen, S. L. and van Dop, H. (2009), Large-eddy simulation of katabatic winds. Part 1, *Acta Geophys.* 57, 803–836. https://doi.org/10.2478/s11600-009-0041-6
- Axelsen, S. L. and van Dop, H. (2009), Part 2: Sensitivity study and comparison with analytical models, *Acta Geophys.* 57, 837–856. https://doi.org/10.2478/s11600-009-0042-5
- Grisogono, B. and Oerlemans, J. (2001), Katabatic flow: analytic solution for gradually varying eddy diffusivities, *J. Atmos. Sci.* 58, 3349–3354. https://doi.org/10.1175/1520-0469(2001)058<3349:KFASFG>2.0.CO;2
- Van den Broeke, M. R., Duynkerke, P. G. and Henneken, E. A. C. (1994), Heat, momentum and moisture budgets of the katabatic layer over the melting zone of the west Greenland ice sheet in summer, *Boundary-Layer Meteorol.* 71, 393–413. https://doi.org/10.1007/BF00712177
- Yang, K. et al. (2008), Turbulent flux transfer over bare-soil surfaces: characteristics and parameterization, *J. Appl. Meteor. Climatol.* 47, 276–290. https://doi.org/10.1175/2007JAMC1547.1
- Wing, A. A. et al. (2018), Radiative–convective equilibrium model intercomparison project, *Geosci. Model Dev.* 11, 793–813. https://doi.org/10.5194/gmd-11-793-2018 (ozone profile)
