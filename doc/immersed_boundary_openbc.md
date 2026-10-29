# Immersed-boundary terrain and open lateral boundaries

This page lists the input keys that the immersed-boundary (IB) terrain and
open lateral boundary (LBC) work on `develop_openbc` adds, with their default
and the value recommended for an IB run over terrain with open boundaries.
Every key is off (or without effect) by default, so existing cases run as
before.

The recommended values are the staircase IB set-up that the test cases
`andren1994_ib`, `axelsen2009_lbc` and `katabatic_idealised` use. A value in
brackets depends on the case; `-` means keep the default.

## A minimal IB run over terrain with open boundaries

```ini
[advec]
swadvec=2i5
fluxlimit_list=thl,qt

[boundary_lateral]
sw_openbc=true
sw_sponge=true
n_sponge=10
tau_sponge=60
# sw_periodic_y=true      for a 2-D slope section, periodic in y

[IB]
sw_immersed_boundary=dem
n_idw_points=5
sw_dem_halo_replicate=true
sw_wall_model=most
z0m=0.01
z0h=0.001
sbcbot=dirichlet          # or flux; with sbot[thl], sbot[qt] or 2-D sbot_spatial maps
sbot[thl]=273.15
sbot[qt]=0.
sw_wall_slope=true
sw_wall_kinematic=true
sw_momentum_air_only=true
sw_wall_sample=true
sw_mason_wall=true
evisc_solid=cap_last
sw_blank_solid=true
sw_blank_solid_tend=true
sw_advec_wall=true
sw_flux_accum=true
```

`sw_wall_kinematic` needs `sw_wall_slope`, `sw_momentum_air_only` needs
`swadvec=2i5`, and `sw_strain_most` (on with the wall model) needs
`[diff] swdiff=smag2` or `tke2`.

## All new keys


### Wall model

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_wall_model` | 0 | most | Wall model on the IB faces: 0 (off, upstream behaviour), neutral (log law) or most (Monin-Obukhov). |
| `[IB]` | `z0m` | required with a wall model | (site) | Momentum roughness length of the IB surface [m]. |
| `[IB]` | `z0h` | required with a wall model | (site) | Heat roughness length of the IB surface [m] (z0h_model=fixed). |
| `[IB]` | `sw_scalar_flux` | true with a wall model | true | Replace the SGS scalar flux through the wall faces by the wall-model flux. |
| `[IB]` | `sw_scalar_flux_vertical` | true | false | Also on riser (vertical) faces. Forced to false by sw_wall_slope. |
| `[IB]` | `sw_wall_stability_vertical` | false | false | Stability correction on riser faces as well (else neutral there). |
| `[IB]` | `sw_momentum_flux` | true with a wall model | true | MOST wall stress applied as a momentum source in the air cells next to the terrain. |
| `[IB]` | `sw_momentum_flux_vertical` | true | true | Also on riser faces. Forced to false by sw_wall_slope. |
| `[IB]` | `sw_strain_most` | true with a wall model | true (false with tke2 and `sw_tke2_wall`) | Rescale the eddy viscosity next to the floor so that K dU/dz = u*^2 (smag2 or tke2 only). With tke2 it rescales K only after the SGS TKE has been computed, so K then no longer equals c_m l √e. |
| `[IB]` | `sw_strain_most_vertical` | false | false | The same rescaling next to riser faces. |
| `[IB]` | `strain_most_min` | 0.05 | 0.05 | Lower bound of that rescaling factor. |
| `[IB]` | `sw_strain_mlen` | true | - | Divide out the Mason damping that Diff_smag2 applied with the height above the domain floor. |
| `[IB]` | `sw_wall_slope` | false | true | MOST on the sub-grid DEM surface of each floor face: wall-normal distance, tangential wind, area factor. |
| `[IB]` | `wall_dn_min` | 0.1 | 0.1 | Lower bound of the wall-normal distance, in dz. |
| `[IB]` | `sw_wall_sample` | false | true | Evaluate the wall model at an image point wall_sample_dz*dz along the surface normal. |
| `[IB]` | `wall_sample_dz` | 1.5 | 1.5 | Distance of that image point, in dz. |
| `[IB]` | `flux_ds_max` | 30 | - | With sbcbot=flux: cap on |phi_s - phi_1| of the diagnosed surface value where u* f_h collapses. |

### Roughness

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_z0m_map` | false | (true with a land-cover map) | Read a per-column z0m map (z0m.0000000) instead of the constant z0m. |
| `[IB]` | `z0h_model` | fixed | fixed or yang2008 | z0h model on ice: fixed (z0h), ratio (z0m/z0h_ratio), yang2008 (Yang et al. 2008) or andreas1987 (Andreas 1987); vantiggelen2023 is reserved and stops the run. |
| `[IB]` | `z0h_model_land` | z0h_model | - | The same for the non-glaciated (land) columns. |
| `[IB]` | `z0h_ratio` | 10 | 10 | z0m/z0h for z0h_model=ratio. |
| `[IB]` | `z0_nu` | 1.5e-5 | 1.5e-5 | Kinematic viscosity in the roughness Reynolds number [m2 s-1]. |
| `[IB]` | `z0_yang_beta` | 7.2 | 7.2 | Coefficient of the Yang et al. (2008) z0h model. |
| `[IB]` | `z0h_min` | 1e-10 | - | Lower bound of the modelled z0h [m]. |
| `[IB]` | `z0h_max` | 0.1 | - | Upper bound of the modelled z0h [m]. |

### Terrain coupling

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_wall_kinematic` | false | true | w on the floor faces = u.grad(eta): the flow follows the sub-grid slope; the terrain leaves the pressure solve. Needs sw_wall_slope. |
| `[IB]` | `sw_momentum_air_only` | false | true | Momentum is exchanged with the terrain only through the MOST stress and the pressure (needs 2i5 advection). |
| `[IB]` | `sw_impermeable` | false | false | Zero the velocity normal to the wall faces after the pressure solve (not needed with sw_wall_kinematic). |
| `[IB]` | `sw_blank_solid` | false | true | Reset the cells inside the terrain (not the ghost cells) every substep. |
| `[IB]` | `sw_blank_solid_tend` | false | true | And zero their tendencies after the pressure solve. |
| `[IB]` | `sw_advec_wall` | false | true | Scalar advection next to the wall uses air cells only. |
| `[IB]` | `sw_advec_wall_conserve` | true | true | Make that correction exactly conservative across the wall faces. |
| `[IB]` | `sw_dem_halo_replicate` | false | true with open LBCs | Fill the DEM halo at the outer edges by replication instead of cyclically. |

### SGS near the wall

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_mason_wall` | false | true | Mason wall damping of the mixing length with the height above the local surface. Applies to smag2, swanisotropic and (since 29 Oct 2026) tke2, where it is done inside `Diff_tke2` for both the eddy viscosity and the dissipation. |
| `[IB]` | `sw_tke2_wall` | false | true with tke2 | tke2 only: the first air cell above each floor face takes its vertical shear (in S²) and N² from the wall model's u* and L, as tke2 does at the first level above the domain floor. Needs a wall model. |
| `[IB]` | `evisc_solid` | off | cap_last | Eddy viscosity inside the terrain: off, zero, cap (at the air neighbours, before wall damping) or cap_last (after it). |
| `[IB]` | `sw_evisc_solid_zero` | false | - | Older spelling of evisc_solid=zero. |
| `[IB]` | `dn_probe_every` | 0 | 20000 | Repeat the diffusion-number probe every N calls (0 = only once). |

### Surface energy balance

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_seb` | false | (true for land/water columns) | Surface energy balance on the non-glaciated columns: linearised Ts, soil diffusion on land, mixed layer on water. |
| `[IB]` | `seb_emis_land` | 0.98 | - | Surface emissivity, land. |
| `[IB]` | `seb_emis_water` | 0.99 | - | Surface emissivity, water. |
| `[IB]` | `seb_lambda_veg` | 10 | - | Skin conductivity, vegetation [W m-2 K-1]. |
| `[IB]` | `seb_lambda_bare` | 15 | - | Skin conductivity, bare ground [W m-2 K-1]. |
| `[IB]` | `seb_cs_land` | 0 | - | Skin heat capacity, land [J m-2 K-1]. |
| `[IB]` | `seb_frs_veg` | 0.05 | - | Shortwave share reaching the soil under full cover. |
| `[IB]` | `seb_lake_mld` | 2 | - | Mixed-layer depth of water columns [m]. |
| `[IB]` | `seb_albedo_land` | 0.15 | - | Albedo of land, without an albedo_ib map. |
| `[IB]` | `seb_albedo_water` | 0.07 | - | Albedo of water, without an albedo_ib map. |
| `[IB]` | `seb_water_min_frac` | 0.5 | - | ib_water fraction above which a column is water. |
| `[IB]` | `sw_vegetation` | false | (true for vegetated land) | Canopy and soil resistance for the moisture flux of the land columns. |
| `[IB]` | `veg_soil_ktot` | 4 | 4 | Soil levels per column for the vegetation and SEB. |

### Prescribed surface

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_timedep_sbot` | false | (true with time-varying sbot maps) | Advance the 2-D sbot_spatial fields in time from <scalar>_sbot.<time> files. |
| `[IB]` | `sbot_loadtime` | required with sw_timedep_sbot | (file interval) | Interval of those files [s]. |

### Output

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[IB]` | `sw_flux_accum` | false | true | Write the IB fluxes and drag as means over the output interval (output only). |
| `[IB]` | `sw_flux_accum_column` | true | false | false: the column time series of the fluxes stay instantaneous; the maps keep the means. |
| `[IB]` | `mean_crosslist` | empty | (state maps) | IB maps of state variables (e.g. `T_2m_ib`, `wspd_10m_res_ib`, `cloud_cover_ib`) also written as interval means. |
| `[IB]` | `cross_sampletime` | 0 (every cross output) | 600 | The IB maps, and their means, only every this many seconds (a multiple of `[cross] sampletime`), so vertical planes can be written more often [s]. |
| `[IB]` | `cloud_cover_qc_min` | 0 | 0 | Condensate (ql + qi) above which a cell is cloud for `cloud_cover_ib` [kg kg-1]. |
| `[IB]` | `sw_tf_cross` | false | true | Terrain-following xy cross-sections <var>_tf<h>. |
| `[IB]` | `tf_cross_heights` | empty | 5,10,20,50 | Heights above the local surface of those planes [m]. |
| `[IB]` | `tavg_interval` | 0 | (output interval) | Time statistics on terrain-following planes over this interval [s] (0 = off; needs swcross). |
| `[IB]` | `tavg_heights` | tf_cross_heights | - | Heights of those statistics [m]. |
| `[IB]` | `tavg_cm` | 0.12 | - | c_m of the SGS TKE estimate in those statistics. |
| `[IB]` | `columnlist` | empty | (as needed) | IB variables written as column output. |
| `[IB]` | `diag_z_mom` | 10 | 10 | Height of the terrain-following wind diagnostics [m]. |
| `[IB]` | `diag_z_scalar` | 2 | 2 | Height of the terrain-following scalar diagnostics [m]. |

### Surface diagnostics added on 28 Oct 2026

Names for `[cross] crosslist` (maps) and `[IB] columnlist` (time series).
They are output only and do not change the simulation.

| Name | Unit | What it is |
|---|---|---|
| `u_10m_res_ib`, `v_10m_res_ib`, `wspd_10m_res_ib`, `wdir_10m_res_ib` | m s-1, degrees | The resolved wind at `diag_z_mom` above the local surface, interpolated linearly between the two levels around it (never below the first air level), at the cell centre. `u_10m_ib` is instead the Monin-Obukhov extrapolation from the first air cell. |
| `sw_up_ib`, `lw_up_ib` | W m-2 | Upward short- and longwave radiation at the IB surface (RRTMGP, `[radiation] sw_ib_columns`). |
| `ps_ib` | Pa | Base-state (hydrostatic) pressure at the surface height, log-linear between half levels; follows `swupdatebasestate`. |
| `cloud_cover_ib` | - | 1 where an air cell of the column holds more than `cloud_cover_qc_min` of condensate; as an interval mean (`mean_crosslist`) the fraction of time with cloud. |
| `sgs_tke` (in `[IB] columnlist`) | m2 s-2 | A profile: the prognostic SGS TKE with `tke2`, or (K_m / (c_m Δ))² from the eddy viscosity with `smag2`, as `calc_sgs_tke` (Deardorff 1980). |

### tke2 over the terrain (29 Oct 2026)

MicroHH's tke2 (Deardorff 1980) treats only the first level above the domain floor as a
surface cell: there the vertical shear in S² and N² come from Monin-Obukhov theory
(`calc_strain2<Surface_model::Enabled>`, `bgradbot`), and Mason's length uses the height
above that floor. Above IB terrain the first air cell is an interior cell, so its shear came
from the no-slip ghost below (∂u/∂z ≈ 2u₁/dz), its N² from the resolved field across the
wall, and its Mason length from the height above the buried domain floor. `sw_strain_most`
rescaled K afterwards, but the SGS TKE equation had already used the uncorrected K and S², and
K is rebuilt from e at every step.

* `[IB] sw_mason_wall=true` now also applies to tke2: `Diff_tke2` computes K_m, K_h and the
  dissipation of every air cell over terrain with the distance d to the IB surface (the
  wall-normal distance for the first air cell, max(z − dem, dz/2) above it) and the IB z0m.
* `[IB] sw_tke2_wall=true`: in the first air cell above each floor face the vertical-shear
  terms of S² are replaced by ½(∂u/∂z)² + ½(∂v/∂z)² with ∂u/∂z = u* φm(ζ) (u/U) / (κ dn), and
  N² by ∂b/∂z = u*² φh(ζ) ζ / (κ dn)², exactly as at the domain floor. u* and L are those of the
  previous wall-model call (the same lag as `sw_strain_most`).
* Code: `Diff::set_ib_wall_mason` and `Diff::set_ib_wall_gradients` (no-ops except in
  `Diff_tke2`), called by `Immersed_boundary::exec_tke2_wall` just before
  `Diff::exec_viscosity`. The tke2 kernels themselves are unchanged; the corrections are applied
  at the listed cells between them.
* With both keys off, and for every other closure, results are bit-identical (tested with
  `-ffp-contract=off`: the 7 IB cases, drycblles, gabls1, bomex, ib_sine, and drycblles with
  tke2).

Tests, flat IB 4 levels above a buried floor against MicroHH's own surface, both with tke2:

| | e in the first air cells (× reference) | K_m first cell | u* | wind at the first 4 levels (× reference) |
|---|---|---|---|---|
| Andrén (1994) Ekman layer, dz 47 m, last hour of 6 h, before | 181, 10, 5.8, 2.7 | 11.4 | +7 % | 1.12, 0.89, 0.90, 0.94 |
| same, `sw_mason_wall` + `sw_tke2_wall` | 1.7, 1.2, 1.6, 1.4 | 0.24 | +1 % | 0.78, 0.85, 0.92, 0.94 |
| same, and `sw_strain_most=false` | 1.6, 1.1, 1.5, 1.3 | 1.29 | +1 % | 0.78, 0.84, 0.91, 0.93 |
| for comparison: smag2 on the flat IB | – | – | −3 % | 0.83, 0.90, 0.92, 0.93 |
| `axelsen2009_rotated` katabatic jet, dz 1 m, last 15 min of 1 h, before | 20, 1.4, 1.6 | 7.2 | | jet 6.6 vs 8.1 m/s |
| same, both keys, `sw_strain_most=false` | 1.2, 0.7, 1.1 | 1.04 | | jet 6.2 vs 8.1 m/s |

The near-surface wind deficit that remains is the same as smag2 shows on the IB at this grid,
so it comes from the IB momentum treatment, not from the closure. Recommended with tke2:
`sw_mason_wall=true`, `sw_tke2_wall=true`, `sw_strain_most=false`.

tke2 keeps its isotropic filter width Δ = (Δx Δy Δz)^(1/3); an anisotropic version (separate
horizontal and vertical lengths in K and in the dissipation, as WRF's 1.5-order TKE with
`mix_isotropic=0`, Skamarock et al. 2019, and the formulations compared by Simon and Chow 2021;
see also Scotti et al. 1993 for the length scale on anisotropic grids) is not implemented.

### Lateral boundaries

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[boundary_lateral]` | `sw_periodic_y` | false | (true for a 2-D slope section) | Open boundaries at west and east only; y stays periodic (DFT in y in the pressure solve). |
| `[boundary_lateral]` | `sw_lbc_tf_periodic` | false | (true for an idealised infinite slope) | Terrain-following periodic LBCs: each edge is fed from the opposite edge at the same height above the surface. Needs sw_sponge and n_sponge >= 1. |
| `[boundary_lateral]` | `lbc_tf_dem` | dem.0000000 | - | DEM file for the terrain-following LBCs. |
| `[boundary_lateral]` | `lbc_tf_gradient[<scalar>]` | 0 | (ambient dphi/dz) | Environmental vertical gradient per scalar, added for the height difference between the edges. |
| `[boundary_lateral]` | `lbc_tf_rock_return` | uniform | - | Where the edge-pair balance returns the terrain mass flux: uniform or surface (needs sw_wall_kinematic). |
| `[boundary_lateral]` | `lbc_tf_rock_return_depth` | 10 | - | Depth of the near-surface band for surface [m]. |
| `[boundary_lateral]` | `lbc_tf_zero_pgrad` | false | - | Zero mean pressure gradient in the upper air, by an integral controller on the edge faces. |
| `[boundary_lateral]` | `lbc_tf_pgrad_zmin` | 20 | - | Measure that gradient above the highest terrain plus this [m]. |
| `[boundary_lateral]` | `lbc_tf_pgrad_gain` | 0.5 | - | Gain of the controller, (0, 1]. |
| `[boundary_lateral]` | `lbc_tf_budget_every` | 0 | - | Write the edge-pair mass budget every N refreshes (0 = off). |

### Radiation

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[radiation]` | `sw_ib_columns` | false | (true with rrtmgp over terrain) | RRTMGP columns start at the IB surface instead of the domain floor. |

### Nesting

| Section | Key | Default | Recommended | What it does |
|---|---|---|---|---|
| `[subdomain]` | `zstart` | 0 | - | Bottom of the vertical window of a nested sub-domain (parent height) [m]. |
| `[subdomain]` | `zend` | -1 | - | Top of that window; negative means the parent top [m]. |
| `[subdomain]` | `interpolation` | nearest | linear | Parent-to-child interpolation: nearest (upstream) or linear (conservative limited slopes, no staircase). |

## References

* Deardorff, J. W. (1980). Stratocumulus-capped mixed layers derived from a three-dimensional model. *Boundary-Layer Meteorol.*, 18, 495-527. https://doi.org/10.1007/BF00119502
* Andreas, E. L. (1987). A theory for the scalar roughness and the scalar transfer coefficients over snow and sea ice. *Boundary-Layer Meteorol.*, 38, 159-184. https://doi.org/10.1007/BF00121562
* Andren, A., Brown, A. R., Graf, J., Mason, P. J., Moeng, C.-H., Nieuwstadt, F. T. M. and Schumann, U. (1994). Large-eddy simulation of a neutrally stratified boundary layer: a comparison of four computer codes. *Q. J. R. Meteorol. Soc.*, 120, 1457-1484. https://doi.org/10.1002/qj.49712052003
* Axelsen, S. L. and van Dop, H. (2009). Large-eddy simulation of katabatic winds. Part 1: Comparison with observations. *Acta Geophysica*, 57, 803-836. (DOI not verified: 10.2478/s11600-009-0041-6)
* Businger, J. A., Wyngaard, J. C., Izumi, Y. and Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *J. Atmos. Sci.*, 28, 181-189. (DOI not verified: 10.1175/1520-0469(1971)028<0181:FPRITA>2.0.CO;2)
* Davies, H. C. (1976). A lateral boundary formulation for multi-level prediction models. *Q. J. R. Meteorol. Soc.*, 102, 405-418. (DOI not verified: 10.1002/qj.49710243210)
* Fedorovich, E., Gibbs, J. A. and Shapiro, A. (2017). Numerical study of nocturnal low-level jets over gently sloping terrain. *J. Atmos. Sci.*, 74, 2813-2834. https://doi.org/10.1175/JAS-D-17-0013.1
* Maass, C. and Schumann, U. (1994). Numerical simulation of turbulent flow over a wavy boundary. In: *Direct and Large-Eddy Simulation I*, Kluwer, 287-297. (DOI not verified: 10.1007/978-94-011-1000-6_25)
* Mason, P. J. and Thomson, D. J. (1992). Stochastic backscatter in large-eddy simulations of boundary layers. *J. Fluid Mech.*, 242, 51-78. https://doi.org/10.1017/S0022112092002271
* Scotti, A., Meneveau, C. and Lilly, D. K. (1993). Generalized Smagorinsky model for anisotropic grids. *Phys. Fluids A*, 5, 2306-2308. (DOI not verified)
* Simon, J. S. and Chow, F. K. (2021). Alternative anisotropic formulations for eddy-viscosity models in the Weather Research and Forecasting model. *Boundary-Layer Meteorol.*, 181, 11-37. https://doi.org/10.1007/s10546-021-00642-0
* Skamarock, W. C. et al. (2019). A description of the Advanced Research WRF model version 4. NCAR Tech. Note NCAR/TN-556+STR. https://doi.org/10.5065/1dfh-6p97
* Tseng, Y.-H. and Ferziger, J. H. (2003). A ghost-cell immersed boundary method for flow in complex geometry. *J. Comput. Phys.*, 192, 593-623. https://doi.org/10.1016/j.jcp.2003.07.024
* van Heerwaarden, C. C. et al. (2017). MicroHH 1.0: a computational fluid dynamics code for direct numerical simulation and large-eddy simulation of atmospheric boundary layer flows. *Geosci. Model Dev.*, 10, 3145-3165. https://doi.org/10.5194/gmd-10-3145-2017
* Yang, K., Koike, T., Ishikawa, H., Kim, J., Li, X., Liu, H., Liu, S., Ma, Y. and Wang, J. (2008). Turbulent flux transfer over bare-soil surfaces: characteristics and parameterization. *J. Appl. Meteorol. Climatol.*, 47, 276-290. (DOI not verified: 10.1175/2007JAMC1547.1)
