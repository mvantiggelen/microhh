/*
 * MicroHH
 * Copyright (c) 2011-2020 Chiel van Heerwaarden
 * Copyright (c) 2011-2020 Thijs Heus
 * Copyright (c) 2014-2020 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "boundary_lateral.h"
#include "netcdf_interface.h"
#include "grid.h"
#include "fields.h"
#include "input.h"
#include "master.h"
#include "timeloop.h"
#include "stats.h"
#include "constants.h"

namespace
{
    #ifdef USEMPI
    template<typename TF> MPI_Datatype mpi_fp_type();
    template<> MPI_Datatype mpi_fp_type<double>() { return MPI_DOUBLE; }
    template<> MPI_Datatype mpi_fp_type<float>() { return MPI_FLOAT; }
    #endif


    template<typename TF>
    bool in_list(const TF value, const std::vector<TF>& list)
    {
        if(std::find(list.begin(), list.end(), value) != list.end())
            return true;
        else
            return false;
    }

    bool any_true(std::map<Lbc_location, bool>& map_in)
    {
        return std::any_of(map_in.begin(), map_in.end(), [](const auto& p) { return p.second; });
    }

    // This kernel enforces a Neumann BC of 0 on w.
    template<typename TF, Lbc_location location>
    void set_ghost_cell_kernel_w(
            TF* const restrict a,
            const int istart, const int iend, const int igc,
            const int jstart, const int jend, const int jgc,
            const int kstart, const int kend,
            const int icells, const int jcells, const int kcells,
            const int ijcells)
    {
        int ijk;
        int ijk_gc;
        int ijk_d;

        // Set the ghost cells using extrapolation.
        if (location == Lbc_location::West || location == Lbc_location::East)
        {
            for (int k=kstart; k<kend; ++k)
                for (int j=jstart; j<jend; ++j)
                    for (int i=0; i<igc; ++i)
                    {
                        if (location == Lbc_location::West)
                        {
                            ijk_d  = (istart    ) + j*icells + k*ijcells;
                            ijk_gc = (istart-1-i) + j*icells + k*ijcells;
                        }
                        else if (location == Lbc_location::East)
                        {
                            ijk_d  = (iend-1  ) + j*icells + k*ijcells;
                            ijk_gc = (iend+i  ) + j*icells + k*ijcells;
                        }

                        a[ijk_gc] = a[ijk_d];
                    }
        }
        else if (location == Lbc_location::North || location == Lbc_location::South)
        {
            for (int k=kstart; k<kend; ++k)
                for (int i=istart; i<iend; ++i)
                    for (int j=0; j<jgc; ++j)
                    {
                        if (location == Lbc_location::South)
                        {
                            ijk_d  = i + (jstart    )*icells + k*ijcells;
                            ijk_gc = i + (jstart-1-j)*icells + k*ijcells;
                        }
                        else if (location == Lbc_location::North)
                        {
                            ijk_d  = i + (jend-1  )*icells + k*ijcells;
                            ijk_gc = i + (jend+j  )*icells + k*ijcells;
                        }

                        a[ijk_gc] = a[ijk_d];
                    }
        }
    }


    template<typename TF>
    TF diffusion_3x3x3(
        const TF* const restrict fld,
        const TF lbc_val,
        const int ijk,
        const int icells,
        const int ijcells)
    {
        auto index = [&](
                const int i3, const int j3, const int k3)
        {
            return ijk + i3-1 + (j3-1)*icells + (k3-1)*ijcells;
        };

        //const TF fld_diff =
        //        - TF(1) * fld[index(0,0,0)] + TF(2) * fld[index(0,1,0)] - TF(1) * fld[index(0,2,0)]
        //        + TF(2) * fld[index(1,0,0)] - TF(4) * fld[index(1,1,0)] + TF(2) * fld[index(1,2,0)]
        //        - TF(1) * fld[index(2,0,0)] + TF(2) * fld[index(2,1,0)] - TF(1) * fld[index(2,2,0)]
        //        + TF(2) * fld[index(0,0,1)] - TF(4) * fld[index(0,1,1)] + TF(2) * fld[index(0,2,1)]
        //        - TF(4) * fld[index(1,0,1)] + TF(8) * fld[index(1,1,1)] - TF(4) * fld[index(1,2,1)]
        //        + TF(2) * fld[index(2,0,1)] - TF(4) * fld[index(2,1,1)] + TF(2) * fld[index(2,2,1)]
        //        - TF(1) * fld[index(0,0,2)] + TF(2) * fld[index(0,1,2)] - TF(1) * fld[index(0,2,2)]
        //        + TF(2) * fld[index(1,0,2)] - TF(4) * fld[index(1,1,2)] + TF(2) * fld[index(1,2,2)]
        //        - TF(1) * fld[index(2,0,2)] + TF(2) * fld[index(2,1,2)] - TF(1) * fld[index(2,2,2)];

        const TF vc = lbc_val - fld[ijk];
        const TF v1 = lbc_val - fld[ijk-1];
        const TF v2 = lbc_val - fld[ijk+1];
        const TF v3 = lbc_val - fld[ijk-icells];
        const TF v4 = lbc_val - fld[ijk+icells];

        const TF fld_diff = v1 + v2 + v3 + v4 - TF(4) * vc;

        return fld_diff;
    }


    // Shape of the lateral sponge weight across the relaxation band.
    //
    // `x` is the LINEAR weight the kernels used to apply directly: 1 at the
    // outermost sponge cell, 1/nsponge (u,v) or 0.5/nsponge (scalars) at the
    // innermost one, and 0 in the first interior cell. That profile stops at
    // a non-zero value and then steps to zero in one cell, which puts a
    // discontinuity in the momentum tendency - and therefore in the
    // divergence, and therefore in w - at the inner edge of the sponge.
    //
    // S(x) = x^2*(3-2x) keeps both ends and has dS/dx = 0 at both, so the
    // nudging dies away smoothly into the interior. See apply_sponge_taper.py.
    template<typename TF>
    inline TF sponge_taper(const TF x)
    {
        return x*x*(TF(3) - TF(2)*x);
    }


    template<typename TF, Lbc_location location>
    void lateral_sponge_kernel_u(
            TF* const restrict ut,
            const TF* const restrict u,
            const TF* const restrict lbc_u,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const int npy,
            const int mpiidy,
            const int igc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        const int igc_pad = (location==Lbc_location::West) ? igc+1 : igc;
        const int nstart = (location==Lbc_location::West) ? 2 : 1;
        const int jstride_lbc = igc_pad+nsponge;

        const TF w_dt = TF(1) / tau_sponge;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                for (int n=nstart; n<=nsponge; ++n)
                {
                    const int ilbc = (location==Lbc_location::West) ? igc+n-1 : nsponge-n;
                    const int ijk_lbc = ilbc + j*jstride_lbc + k*jstride_lbc*jcells;

                    const int i = (location==Lbc_location::West) ? istart+(n-1) : iend-n;
                    const int ijk = i + j*icells + k*ijcells;

                    const TF u_diff = diffusion_3x3x3(
                            u, lbc_u[ijk_lbc], ijk, icells, ijcells);

                    // Nudge coefficient, smoothly tapered to zero at the
                    // inner edge of the band (see sponge_taper).
                    const TF f_sponge = sponge_taper<TF>((TF(1)+nsponge-n) / nsponge);
                    const TF w1n = w_dt * f_sponge;
                    const TF w2n = w_diff * f_sponge;

                    ut[ijk] += w1n * (lbc_u[ijk_lbc]-u[ijk]);
                    ut[ijk] -= w2n * u_diff;
                }
    }

    template<typename TF, Lbc_location location>
    void lateral_sponge_kernel_v(
            TF* const restrict vt,
            const TF* const restrict v,
            const TF* const restrict lbc_v,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const int npx,
            const int mpiidx,
            const int jgc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        const int jgc_pad = (location==Lbc_location::South) ? jgc+1 : jgc;

        const TF w_dt = TF(1) / tau_sponge;

        // South is the staggered side: v[jstart] is the boundary FACE, set
        // by set_lbc_gcs, so the band starts at n = 2. North is not: v[jend]
        // is the face, curtain plane r is face jend-nsponge+r, and the band is
        // jend-1 .. jend-nsponge with n = 1 .. nsponge - exactly what
        // lateral_sponge_kernel_u does at East. See apply_lbc_sponge_edges.py.
        const int nstart = (location==Lbc_location::South) ? 2 : 1;

        for (int k=kstart; k<kend; ++k)
            for (int i=istart; i<iend; ++i)
                for (int n=nstart; n<=nsponge; ++n)
                {
                    const int jlbc = (location==Lbc_location::South) ? jgc+n-1 : nsponge-n;
                    const int ijk_lbc = i + jlbc*icells + k*icells*(jgc_pad+nsponge);

                    const int j = (location==Lbc_location::South) ? jstart+(n-1) : jend-n;
                    const int ijk = i + j*icells + k*ijcells;

                    const TF v_diff = diffusion_3x3x3(
                            v, lbc_v[ijk_lbc], ijk, icells, ijcells);

                    // Nudge coefficient, smoothly tapered to zero at the
                    // inner edge of the band (see sponge_taper).
                    const TF f_sponge = sponge_taper<TF>((TF(1)+nsponge-n) / nsponge);
                    const TF w1n = w_dt * f_sponge;
                    const TF w2n = w_diff * f_sponge;

                    vt[ijk] += w1n * (lbc_v[ijk_lbc]-v[ijk]);
                    vt[ijk] -= w2n * v_diff;
                }
    }

    template<typename TF, Lbc_location location, bool sw_recycle>
    void lateral_sponge_kernel_s(
            TF* const restrict at,
            const TF* const restrict a,
            const TF* const restrict lbc,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const TF tau_recycle,
            const TF recycle_offset,
            const int npx, const int npy,
            const int mpiidx, const int mpiidy,
            const int igc, const int jgc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        const TF w_dt = TF(1) / tau_sponge;
        const TF r_dt = TF(1) / tau_recycle;

        if (location == Lbc_location::West || location == Lbc_location::East)
        {
            for (int k=kstart; k<kend; ++k)
                for (int n=1; n<=nsponge; ++n)
                {
                    // Offset in y-direction for domain corners.
                    const int jstart_loc = mpiidy == 0     ? jstart + (n-1) : jstart;
                    const int jend_loc   = mpiidy == npy-1 ? jend   - (n-1) : jend;

                    for (int j=jstart_loc; j<jend_loc; ++j)
                    {
                        // Index in LBC:
                        const int jstride_lbc = igc+nsponge;
                        const int ilbc = (location==Lbc_location::West) ? igc+n-1 : nsponge-n;
                        const int ijk_lbc = ilbc + j*jstride_lbc + k*jstride_lbc*jcells;

                        const int i = (location==Lbc_location::West) ? istart+n-1 : iend-n;
                        const int ijk = i + j*icells + k*ijcells;

                        const TF a_diff = diffusion_3x3x3(
                                a, lbc[ijk_lbc], ijk, icells, ijcells);

                        // Nudge coefficient, smoothly tapered to zero at the
                        // inner edge of the band (see sponge_taper).
                        const TF f_sponge = sponge_taper<TF>((TF(1)+nsponge-(n+TF(0.5))) / nsponge);
                        const TF w1n = w_dt * f_sponge;
                        const TF w2n = w_diff * f_sponge;

                        // Apply nudge and sponge tendencies.
                        at[ijk] += w1n * (lbc[ijk_lbc]-a[ijk]);
                        at[ijk] -= w2n * a_diff;

                        // Turbulence recycling.
                        if (sw_recycle)
                        {
                            // Recycle strength; 0 at boundary, 1 at edge nudging zone.
                            const TF f_recycle = TF(1) - f_sponge;

                            // Source of recycling.
                            const int offset = (location == Lbc_location::West) ? recycle_offset : -recycle_offset;
                            const int ijko = (i+offset) + j*icells + k*ijcells;

                            TF a_mean = 0;
                            for (int jc=-3; jc<4; ++jc)
                                for (int ic=-3; ic<4; ++ic)
                                    a_mean += a[ijko + ic + jc*icells];
                            a_mean /= TF(49.);

                            at[ijk] += f_recycle * r_dt * ((a[ijko] - a_mean) - (a[ijk] - lbc[ijk_lbc]));
                        }
                    }
                }
        }
        if (location == Lbc_location::South || location == Lbc_location::North)
        {
            for (int k=kstart; k<kend; ++k)
                for (int n=1; n<=nsponge; ++n)
                {
                    const int istart_loc = (mpiidx == 0)     ? istart + n : istart;
                    const int iend_loc   = (mpiidx == npx-1) ? iend   - n : iend;

                    for (int i=istart_loc; i<iend_loc; ++i)
                    {
                        const int kstride_lbc = jgc + nsponge;
                        const int jlbc = (location==Lbc_location::South) ? jgc+n-1 : nsponge-n;
                        const int ijk_lbc = i + jlbc*icells + k*icells*kstride_lbc;

                        const int j = (location==Lbc_location::South) ? jstart+n-1 : jend-n;
                        const int ijk = i + j*icells + k*ijcells;

                        const TF a_diff = diffusion_3x3x3(
                                a, lbc[ijk_lbc], ijk, icells, ijcells);

                        // Nudge coefficient, smoothly tapered to zero at the
                        // inner edge of the band (see sponge_taper).
                        const TF f_sponge = sponge_taper<TF>((TF(1)+nsponge-(n+TF(0.5))) / nsponge);
                        const TF w1n = w_dt * f_sponge;
                        const TF w2n = w_diff * f_sponge;

                        at[ijk] += w1n*(lbc[ijk_lbc]-a[ijk]);
                        at[ijk] -= w2n*a_diff;

                        if (sw_recycle)
                        {
                            // Recycle strength; 0 at boundary, 1 at edge nudging zone.
                            const TF f_recycle = TF(1) - f_sponge;

                            // Source of recycling.
                            const int offset = (location == Lbc_location::South) ? recycle_offset : -recycle_offset;
                            const int ijko = i + (j+offset)*icells + k*ijcells;

                            TF a_mean = 0;
                            for (int jc=-3; jc<4; ++jc)
                                for (int ic=-3; ic<4; ++ic)
                                    a_mean += a[ijko + ic + jc*icells];
                            a_mean /= TF(49.);

                            at[ijk] += f_recycle * r_dt * ((a[ijko] - a_mean) - (a[ijk] - lbc[ijk_lbc]));
                        }
                    }
                }
        }
    }


    template<typename TF>
    void set_corner_ghost_cell_kernel(
            TF* const restrict fld,
            const int mpiidx, const int mpiidy,
            const int npx, const int npy,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int kcells, const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        const int igc = istart;
        const int jgc = jstart;

        if (mpiidx == 0 && mpiidy == 0)
        {
            for (int k=kstart; k<kend; ++k)
            {
                const int ijk0 = istart + jstart*jj + k*kk;
                const TF dfdi = fld[ijk0] - fld[ijk0-ii];
                const TF dfdj = fld[ijk0] - fld[ijk0-jj];

                for (int dj=1; dj<jgc+1; ++dj)
                    for (int di=1; di<igc+1; ++di)
                    {
                        const int i = istart-di;
                        const int j = jstart-dj;
                        const int ijk = i + j*jj + k*kk;

                        fld[ijk] = fld[ijk0] - di*dfdi - dj*dfdj;
                    }
            }

            for (int k=0; k<kstart; ++k)
                for (int j=jstart-1; j>=0; --j)
                    for (int i=istart-1; i>=0; --i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijks = i+j*jj + kstart*kk;
                        fld[ijk] = fld[ijks];
                    }

            for (int k=kend; k<kcells; ++k)
                for (int j=jstart-1; j>=0; --j)
                    for (int i=istart-1; i>=0; --i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijke = i+j*jj + (kend-1)*kk;
                        fld[ijk] = fld[ijke];
                    }
        }

        if (mpiidx == npx-1 && mpiidy == 0)
        {
            // South-east corner
            for (int k=kstart; k<kend; ++k)
            {
                const int ijk0 = (iend-1) + jstart*jj + k*kk;
                const TF dfdi = fld[ijk0+ii] - fld[ijk0];
                const TF dfdj = fld[ijk0] - fld[ijk0-jj];

                for (int dj=1; dj<jgc+1; ++dj)
                    for (int di=1; di<igc+1; ++di)
                    {
                        const int i = (iend-1)+di;
                        const int j = jstart-dj;
                        const int ijk = i + j*jj + k*kk;

                        fld[ijk] = fld[ijk0] + di*dfdi - dj*dfdj;
                    }
            }

            for (int k=0; k<kstart; ++k)
                for (int j=jstart-1; j>=0; --j)
                    for (int i=iend; i<icells; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijks = i+j*jj + kstart*kk;
                        fld[ijk] = fld[ijks];
                    }

            for (int k=kend; k<kcells; ++k)
                for (int j=jstart-1; j>=0; --j)
                    for (int i=iend; i<icells; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijke = i+j*jj + (kend-1)*kk;
                        fld[ijk] = fld[ijke];
                    }
        }

        if (mpiidx == 0 && mpiidy == npy-1)
        {
            // North-west corner
            for (int k=kstart; k<kend; ++k)
            {
                const int ijk0 = istart + (jend-1)*jj + k*kk;
                const TF dfdi = fld[ijk0] - fld[ijk0-ii];
                const TF dfdj = fld[ijk0+jj] - fld[ijk0];

                for (int dj=1; dj<jgc+1; ++dj)
                    for (int di=1; di<igc+1; ++di)
                    {
                        const int i = istart-di;
                        const int j = (jend-1)+dj;
                        const int ijk = i + j*jj + k*kk;

                        fld[ijk] = fld[ijk0] - di*dfdi + dj*dfdj;
                    }
            }

            for (int k=0; k<kstart; ++k)
                for (int j=jend; j<jcells; ++j)
                    for (int i=istart-1; i>=0; --i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijks = i+j*jj + kstart*kk;
                        fld[ijk] = fld[ijks];
                    }

            for (int k=kend; k<kcells; ++k)
                for (int j=jend; j<jcells; ++j)
                    for (int i=istart-1; i>=0; --i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijke = i+j*jj + (kend-1)*kk;
                        fld[ijk] = fld[ijke];
                    }
        }

        if (mpiidx == npx-1 && mpiidy == npy-1)
        {
            // North-east corner
            for (int k=kstart; k<kend; ++k)
            {
                const int ijk0 = (iend-1) + (jend-1)*jj + k*kk;
                const TF dfdi = fld[ijk0+ii] - fld[ijk0];
                const TF dfdj = fld[ijk0+jj] - fld[ijk0];

                for (int dj=1; dj<jgc+1; ++dj)
                    for (int di=1; di<igc+1; ++di)
                    {
                        const int i = (iend-1)+di;
                        const int j = (jend-1)+dj;
                        const int ijk = i + j*jj + k*kk;

                        fld[ijk] = fld[ijk0] + di*dfdi + dj*dfdj;
                    }
            }

            for (int k=0; k<kstart; ++k)
                for (int j=jend; j<jcells; ++j)
                    for (int i=iend; i<icells; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijks = i+j*jj + kstart*kk;
                        fld[ijk] = fld[ijks];
                    }

            for (int k=kend; k<kcells; ++k)
                for (int j=jend; j<jcells; ++j)
                    for (int i=iend; i<icells; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        const int ijke = i+j*jj + (kend-1)*kk;
                        fld[ijk] = fld[ijke];
                    }
        }
    }


    template<typename TF>
    void interpolate_lbc_kernel(
            TF* const restrict fld,
            const TF* const restrict fld_prev,
            const TF* const restrict fld_next,
            const int size,
            const TF f0)
    {
        const TF f1 = TF(1) - f0;
        for (int n=0; n<size; ++n)
            fld[n] = f0 * fld_prev[n] + f1 * fld_next[n];
    }


    template<typename TF>
    void set_lbc_gcs(
            TF* const restrict fld,
            const TF* const restrict lbc,
            const int ngc, const int nsponge,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int kcells,
            Lbc_location location)
    {
        const int jstride_out = icells;
        const int kstride_out = icells * jcells;

        if (location == Lbc_location::West)
        {
            const int jstride_w = ngc + nsponge;
            const int kstride_w = jstride_w * jcells;

            for (int k=kstart; k<kend; k++)
                for (int j=0; j<jcells; j++)
                    for (int i=0; i<ngc; i++)
                    {
                        const int ijk_in = i + j*jstride_w + k*kstride_w;
                        const int ijk_out = i + j*jstride_out + k*kstride_out;
                        fld[ijk_out] = lbc[ijk_in];
                    }

            // Set one ghost cell below surface. What BC to use?
            for (int j=0; j<jcells; j++)
                for (int i=0; i<ngc; i++)
                {
                    const int ijk = i + j*jstride_out + kstart*kstride_out;
                    const int ijk_gc = i + j*jstride_out + (kstart-1)*kstride_out;
                    fld[ijk_gc] = fld[ijk];
                }
        }

        if (location == Lbc_location::East)
        {
            const int jstride_e = ngc + nsponge;
            const int kstride_e = jstride_e * jcells;

            for (int k=kstart; k<kend; k++)
                for (int j=0; j<jcells; j++)
                    for (int i=0; i<ngc; i++)
                    {
                        const int ijk_in = (i+nsponge) + j*jstride_e + k*kstride_e;
                        const int ijk_out = (i+iend) + j*jstride_out + k*kstride_out;
                        fld[ijk_out] = lbc[ijk_in];
                    }

            // Set one ghost cell below surface. What BC to use?
            for (int j=0; j<jcells; j++)
                for (int i=0; i<ngc; i++)
                {
                    const int ijk = (i+iend) + j*jstride_out + kstart*kstride_out;
                    const int ijk_gc = (i+iend) + j*jstride_out + (kstart-1)*kstride_out;
                    fld[ijk_gc] = fld[ijk];
                }
        }

        if (location == Lbc_location::South)
        {
            const int jstride_s = icells;
            const int kstride_s = jstride_s * (ngc + nsponge);

            for (int k=kstart; k<kend; k++)
                for (int j=0; j<ngc; j++)
                    for (int i=0; i<icells; i++)
                    {
                        const int ijk_in = i + j*jstride_s + k*kstride_s;
                        const int ijk_out = i + j*jstride_out + k*kstride_out;
                        fld[ijk_out] = lbc[ijk_in];
                    }

            // Set one ghost cell below surface. What BC to use?
            for (int j=0; j<ngc; j++)
                for (int i=0; i<icells; i++)
                {
                    const int ijk = i + j*jstride_out + kstart*kstride_out;
                    const int ijk_gc = i + j*jstride_out + (kstart-1)*kstride_out;
                    fld[ijk_gc] = fld[ijk];
                }

        }

        if (location == Lbc_location::North)
        {
            const int jstride_n = icells;
            const int kstride_n = jstride_n * (ngc + nsponge);

            for (int k=kstart; k<kend; k++)
                for (int j=0; j<ngc; j++)
                    for (int i=0; i<icells; i++)
                    {
                        const int ijk_in = i + (j+nsponge)*jstride_n + k*kstride_n;
                        const int ijk_out = i + (j+jend)*jstride_out + k*kstride_out;
                        fld[ijk_out] = lbc[ijk_in];
                    }

            // Set one ghost cell below surface. What BC to use?
            for (int j=0; j<ngc; j++)
                for (int i=0; i<icells; i++)
                {
                    const int ijk = i + (j+jend)*jstride_out + kstart*kstride_out;
                    const int ijk_gc = i + (j+jend)*jstride_out + (kstart-1)*kstride_out;
                    fld[ijk_gc] = fld[ijk];
                }
        }
    };


    template<typename TF, Lbc_location lbc_location>
    void calc_div_x(
            TF& div,
            const TF* const restrict lbc_u,
            const TF* const restrict rhoref,
            const TF* const restrict dz,
            const TF dy,
            const int nsponge,
            const int ngc, const int kgc,
            const int jstart, const int jend,
            const int ktot,
            const int jcells)
    {
        const int jstride_w = ngc + nsponge + 1;
        const int kstride_w = jstride_w * jcells;

        const int jstride_e = ngc + nsponge;
        const int kstride_e = jstride_e * jcells;

        const int iw = ngc;
        const int ie = nsponge;

        for (int k=0; k<ktot; ++k)
            for (int j=jstart; j<jend; ++j)
            {
                const int ijk_w = iw + j*jstride_w + k*kstride_w;
                const int ijk_e = ie + j*jstride_e + k*kstride_e;

                // Div = east-west.
                if (lbc_location == Lbc_location::East)
                    div += rhoref[k+kgc] * dy * dz[k+kgc] * lbc_u[ijk_e];
                else
                    div -= rhoref[k+kgc] * dy * dz[k+kgc] * lbc_u[ijk_w];
            }
    }

    template<typename TF, Lbc_location lbc_location>
    void calc_div_y(
            TF& div,
            const TF* const restrict lbc_v,
            const TF* const restrict rhoref,
            const TF* const restrict dz,
            const TF dx,
            const int nsponge,
            const int ngc, const int kgc,
            const int istart, const int iend,
            const int ktot,
            const int icells)
    {
        const int jstride_s = icells;
        const int kstride_s = jstride_s * (ngc + nsponge + 1);
        const int tstride_s = kstride_s * ktot;

        const int jstride_n = icells;
        const int kstride_n = jstride_n * (ngc + nsponge);
        const int tstride_n = kstride_n * ktot;

        const int js = ngc;
        const int jn = nsponge;

        for (int k=0; k<ktot; ++k)
            for (int i=istart; i<iend; ++i)
            {
                const int ijk_s = i + js*jstride_s + k*kstride_s;
                const int ijk_n = i + jn*jstride_n + k*kstride_n;

                // Div = north-south.
                if (lbc_location == Lbc_location::North)
                    div += rhoref[k+kgc] * dx * dz[k+kgc] * lbc_v[ijk_n];
                else
                    div -= rhoref[k+kgc] * dx * dz[k+kgc] * lbc_v[ijk_s];
            }
    }

    template<typename TF>
    bool is_equal(const TF a, const TF b)
    {
        const TF epsilon = std::max(TF(10) * std::numeric_limits<TF>::epsilon(), std::max(std::abs(a), std::abs(b)) * TF(10) * std::numeric_limits<TF>::epsilon());
        return std::abs(a - b) <= epsilon;
    }
}


template<typename TF>
Boundary_lateral<TF>::Boundary_lateral(
        Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
        master(masterin), grid(gridin), fields(fieldsin), field3d_io(masterin, gridin)
{
    sw_openbc = inputin.get_item<bool>("boundary_lateral", "sw_openbc", "", false);
    sw_lbc_tf_periodic = false;
    lbc_tf_initialized = false;

    if (sw_openbc)
    {
        sw_openbc_uv = inputin.get_item<bool>("boundary_lateral", "sw_openbc_uv", "", true);
        sw_openbc_w = inputin.get_item<bool>("boundary_lateral", "sw_openbc_w", "", false);
        sw_neumann_w = inputin.get_item<bool>("boundary_lateral", "sw_neumann_w", "", true);
        sw_wtop_2d = inputin.get_item<bool>("boundary_lateral", "sw_wtop_2d", "", false);
        slist = inputin.get_list<std::string>("boundary_lateral", "slist", "", std::vector<std::string>());

        // Check....
        if (sw_openbc_w && sw_neumann_w)
            throw std::runtime_error("Cant have both \"sw_openbc_w\" and \"sw_neumann_w\" = true!");

        sw_timedep = inputin.get_item<bool>("boundary_lateral", "sw_timedep", "", false);
        loadfreq = inputin.get_item<int>("boundary_lateral", "loadfreq", "");

        // Lateral sponge / diffusion layer.
        sw_sponge = inputin.get_item<bool>("boundary_lateral", "sw_sponge", "", false);
        if (sw_sponge)
        {
            n_sponge = inputin.get_item<int>("boundary_lateral", "n_sponge", "", 5);
            tau_sponge = inputin.get_item<TF>("boundary_lateral", "tau_sponge", "", 60);
            w_diff = inputin.get_item<TF>("boundary_lateral", "w_diff", "", 0.0033);
        }

        // Turbulence recycling.
        sw_recycle[Lbc_location::North] = inputin.get_item<bool>("boundary_lateral", "sw_recycle", "north", false);
        sw_recycle[Lbc_location::East]  = inputin.get_item<bool>("boundary_lateral", "sw_recycle", "east", false);
        sw_recycle[Lbc_location::South] = inputin.get_item<bool>("boundary_lateral", "sw_recycle", "south", false);
        sw_recycle[Lbc_location::West]  = inputin.get_item<bool>("boundary_lateral", "sw_recycle", "west", false);

        if (any_true(sw_recycle))
        {
            recycle_list = inputin.get_list<std::string>(
                "boundary_lateral", "recycle_list", "", std::vector<std::string>());
            tau_recycle = inputin.get_item<TF>("boundary_lateral", "tau_recycle", "");
            recycle_offset = inputin.get_item<int>("boundary_lateral", "recycle_offset", "");
        }

        // Terrain-following periodic LBCs (patch 28): every edge is fed from
        // the opposite edge at the same height above the local surface,
        // instead of from lbc_* files. See apply_lbc_tf_periodic.py.
        sw_lbc_tf_periodic = inputin.get_item<bool>(
                "boundary_lateral", "sw_lbc_tf_periodic", "", false);

        // apply_ib_wall_kinematic.py: the terrain is out of the pressure
        // solve, so its edge faces must carry exactly the terrain mass flux.
        sw_lbc_rock_export = sw_lbc_tf_periodic && inputin.get_item<bool>(
                "IB", "sw_wall_kinematic", "", false);

        if (sw_lbc_tf_periodic)
        {
            lbc_tf_dem_file = inputin.get_item<std::string>(
                    "boundary_lateral", "lbc_tf_dem", "", "dem.0000000");

            for (auto& fld : slist)
                lbc_tf_gradient.emplace(fld, inputin.get_item<TF>(
                        "boundary_lateral", "lbc_tf_gradient", fld, TF(0)));
        }

        //sw_recycle = inputin.get_item<bool>("boundary_lateral", "sw_recycle", "", false);
        //if (sw_recycle)
        //{
        //    recycle_list = inputin.get_list<std::string>(
        //        "boundary_lateral", "recycle_list", "", std::vector<std::string>());
        //    tau_recycle = inputin.get_item<TF>("boundary_lateral", "tau_recycle", "");
        //    recycle_offset = inputin.get_item<int>("boundary_lateral", "recycle_offset", "");
        //}
    }
}


template <typename TF>
Boundary_lateral<TF>::~Boundary_lateral()
{
}


template <typename TF>
void Boundary_lateral<TF>::init()
{
    if (!sw_openbc)
        return;

    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    // Checks!
    if (sw_lbc_tf_periodic)
    {
        if (sw_timedep)
            throw std::runtime_error(
                    "[boundary_lateral] sw_lbc_tf_periodic feeds the LBCs from the opposite "
                    "edge and cannot be combined with sw_timedep=true");
        if (!sw_sponge || n_sponge < 1)
            throw std::runtime_error(
                    "[boundary_lateral] sw_lbc_tf_periodic needs sw_sponge=true and n_sponge >= 1");
        if (gd.imax < 2*n_sponge + gd.igc + 1 || gd.jmax < 2*n_sponge + gd.jgc + 1)
            throw std::runtime_error(
                    "[boundary_lateral] sw_lbc_tf_periodic needs imax, jmax >= 2*n_sponge + 4 "
                    "(the source band must sit on the edge rank)");
    }

    if (any_true(sw_recycle))
    {
        if (!sw_sponge)
            throw std::runtime_error("Turbulence recycling only works combined with sw_sponge=1");

        if ((sw_recycle[Lbc_location::West] || sw_recycle[Lbc_location::East])  && recycle_offset > gd.imax+gd.igc)
            throw std::runtime_error("Turbulence recycling offset too large for domain decomposition in x-direction");
        if ((sw_recycle[Lbc_location::North] || sw_recycle[Lbc_location::South])  && recycle_offset > gd.jmax+gd.jgc)
            throw std::runtime_error("Turbulence recycling offset too large for domain decomposition in y-direction");
    }

    auto add_lbc_var = [&](
            Lbc_map<TF>& lbc_w_in,
            Lbc_map<TF>& lbc_e_in,
            Lbc_map<TF>& lbc_s_in,
            Lbc_map<TF>& lbc_n_in,
            const std::string& name)
    {
        const int igc_pad = (name == "u") ? gd.igc+1 : gd.igc;
        const int jgc_pad = (name == "v") ? gd.jgc+1 : gd.jgc;

        const int nlbc_w = igc_pad + n_sponge;
        const int nlbc_e = gd.igc + n_sponge;
        const int nlbc_s = jgc_pad + n_sponge;
        const int nlbc_n = gd.jgc + n_sponge;

        if (md.mpicoordx == 0)
            lbc_w_in.emplace(name, std::vector<TF>(nlbc_w * gd.kcells * gd.jcells));
        if (md.mpicoordx == md.npx-1)
            lbc_e_in.emplace(name, std::vector<TF>(nlbc_e * gd.kcells * gd.jcells));
        if (md.mpicoordy == 0)
            lbc_s_in.emplace(name, std::vector<TF>(gd.icells * nlbc_s * gd.kcells));
        if (md.mpicoordy == md.npy-1)
            lbc_n_in.emplace(name, std::vector<TF>(gd.icells * nlbc_n * gd.kcells));
    };

    auto add_lbcs = [&](
            Lbc_map<TF>& lbc_w_in,
            Lbc_map<TF>& lbc_e_in,
            Lbc_map<TF>& lbc_s_in,
            Lbc_map<TF>& lbc_n_in)
    {
        if (sw_openbc_uv)
        {
            add_lbc_var(lbc_w_in, lbc_e_in, lbc_s_in, lbc_n_in, "u");
            add_lbc_var(lbc_w_in, lbc_e_in, lbc_s_in, lbc_n_in, "v");
        }

        if (sw_openbc_w)
            add_lbc_var(lbc_w_in, lbc_e_in, lbc_s_in, lbc_n_in, "w");

        for (auto& fld : slist)
            add_lbc_var(lbc_w_in, lbc_e_in, lbc_s_in, lbc_n_in, fld);
    };

    // Create fixed/constant (or time interpolated in case of `sw_timedep`) LBC arrays.
    add_lbcs(lbc_w, lbc_e, lbc_s, lbc_n);

    if (sw_timedep)
    {
        // Add LBC arrays at previous and next time steps.
        add_lbcs(lbc_w_prev, lbc_e_prev, lbc_s_prev, lbc_n_prev);
        add_lbcs(lbc_w_next, lbc_e_next, lbc_s_next, lbc_n_next);
    }
}


template <typename TF>
void Boundary_lateral<TF>::read_lbc(
        TF& div_u, TF& div_v,
        Lbc_map<TF>& lbc_w_in,
        Lbc_map<TF>& lbc_e_in,
        Lbc_map<TF>& lbc_s_in,
        Lbc_map<TF>& lbc_n_in,
        const int iotime)
{
    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    //auto dump_vector = [&](
    //        std::vector<TF>& fld,
    //        const std::string& name)
    //{
    //    std::string name_out = name + "." + std::to_string(md.mpicoordx) + "." + std::to_string(md.mpicoordy) + ".bin";

    //    FILE *pFile;
    //    pFile = fopen(name_out.c_str(), "wb");

    //    if (pFile == NULL)
    //        throw std::runtime_error("Opening raw dump field failed.");

    //    fwrite(fld.data(), sizeof(TF), fld.size(), pFile);
    //    fclose(pFile);
    //};


    auto read_binary = [&](
            std::vector<TF>& vec,
            const std::string name,
            const unsigned long size)
    {
        char filename[256];
        std::sprintf(filename, "%s.%07d", name.c_str(), iotime);

        FILE *pFile;
        pFile = fopen(filename, "rb");

        bool success = true;
        if (pFile == NULL)
            success = false;

        if (success)
        {
            // Jump to offset & read requested chunk.
            //const size_t offset = time_index * size * sizeof(TF);
            //fseek(pFile, offset, SEEK_SET);

            if (fread(vec.data(), sizeof(TF), size, pFile) != (unsigned)size)
                success = false;
        }

        if (!success)
        {
            #ifdef USEMPI
            std::cout << "SINGLE PROCESS EXCEPTION: reading binary " << filename << " failed." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            #else
            throw std::runtime_error("ERROR: reading binary failed");
            #endif
        }

        fclose(pFile);
    };

    auto copy_boundary = [&](
            std::vector<TF>& fld_out,
            const std::vector<TF>& fld_in,
            const int isize_in, const int jsize_in,
            const int isize_out, const int jsize_out,
            const int istart_in, const int jstart_in,
            const std::string& name, const std::string& loc)
    {
        // Copy LBC data from vector `fld_in`, which holds an entire
        // domain edge with data for all MPI tasks, to the `Lbc_map`
        // containing only data needed for the current MPI subdomain.

        const int size_in = gd.ktot * jsize_in * isize_in;
        const int size_out = gd.kcells * jsize_out * isize_out;

        const int jstride_in = isize_in;
        const int kstride_in = jstride_in * jsize_in;

        const int jstride_out = isize_out;
        const int kstride_out = jstride_out * jsize_out;

        for (int k=0; k<gd.ktot; k++)
            for (int j=0; j<jsize_out; j++)
                for (int i=0; i<isize_out; i++)
                {
                    const int ijk_in = i+istart_in + (j+jstart_in)*jstride_in + k*kstride_in;
                    const int ijk_out = i + j*jstride_out + (k+gd.kstart)*kstride_out;

                    fld_out[ijk_out] = fld_in[ijk_in];
                }
    };


    auto copy_boundaries = [&](const std::string& name)
    {
        // Read LBC data from binary files, and copy out the
        // part needed on the current MPI subdomain.

        // `u` at west boundary and `v` at south boundary also contain
        // `u` at `istart` and `v` at `jstart`, and are therefore larger.
        const int igc_pad = (name == "u") ? gd.igc+1 : gd.igc;
        const int jgc_pad = (name == "v") ? gd.jgc+1 : gd.jgc;

        // Number of ghost + sponge cells.
        const int nlbc_w = igc_pad + n_sponge;
        const int nlbc_e = gd.igc + n_sponge;
        const int nlbc_s = jgc_pad + n_sponge;
        const int nlbc_n = gd.jgc + n_sponge;

        const int ncells_w = gd.ktot * (gd.jtot+2*gd.jgc) * nlbc_w;
        const int ncells_e = gd.ktot * (gd.jtot+2*gd.jgc) * nlbc_e;
        const int ncells_s = gd.ktot * nlbc_s * (gd.itot+2*gd.igc);
        const int ncells_n = gd.ktot * nlbc_n * (gd.itot+2*gd.igc);

        // Arrays which hold data for the full domain edge,
        // i.e. for all MPI tasks. The `copy_boundary()` function
        // later copies out the local data needed on each MPI subdomain.
        std::vector<TF> lbc_w_full;
        std::vector<TF> lbc_e_full;
        std::vector<TF> lbc_s_full;
        std::vector<TF> lbc_n_full;

        if (md.mpicoordx == 0)
            lbc_w_full.resize(ncells_w);
        if (md.mpicoordx == md.npx-1)
            lbc_e_full.resize(ncells_e);
        if (md.mpicoordy == 0)
            lbc_s_full.resize(ncells_s);
        if (md.mpicoordy == md.npy-1)
            lbc_n_full.resize(ncells_n);


        if (md.mpicoordx == 0)
        {
            if (md.mpicoordy == 0)
                read_binary(lbc_w_full, "lbc_" + name + "_west", ncells_w);
            master.broadcast_y(lbc_w_full.data(), ncells_w, 0);

            copy_boundary(
                    lbc_w_in.at(name), lbc_w_full,
                    nlbc_w, gd.jtot+2*gd.jgc,
                    nlbc_w, gd.jcells,
                    0, md.mpicoordy*gd.jmax,
                    name, "west");

            // Calculate total inflow over west boundary.
            if (name == "u" && md.mpicoordy == 0)
                calc_div_x<TF, Lbc_location::West>(
                        div_u,
                        lbc_w_full.data(),
                        fields.rhoref.data(),
                        gd.dz.data(),
                        gd.dy,
                        n_sponge,
                        gd.igc, gd.kgc,
                        gd.jgc, gd.jtot+gd.jgc,
                        gd.ktot, gd.jtot+(2*gd.jgc));
        }

        if (md.mpicoordx == md.npx-1)
        {
            if (md.mpicoordy == md.npy-1)
                read_binary(lbc_e_full, "lbc_" + name + "_east", ncells_e);
            master.broadcast_y(lbc_e_full.data(), ncells_e, md.npy-1);

            copy_boundary(
                    lbc_e_in.at(name), lbc_e_full,
                    nlbc_e, gd.jtot+2*gd.jgc,
                    nlbc_e, gd.jcells,
                    0, md.mpicoordy*gd.jmax,
                    name, "east");

            // Calculate total outflow over east boundary.
            if (name == "u" && md.mpicoordy == md.npy-1)
                calc_div_x<TF, Lbc_location::East>(
                        div_u,
                        lbc_e_full.data(),
                        fields.rhoref.data(),
                        gd.dz.data(),
                        gd.dy,
                        n_sponge,
                        gd.igc, gd.kgc,
                        gd.jgc, gd.jtot+gd.jgc,
                        gd.ktot, gd.jtot+(2*gd.jgc));
	    }

        if (md.mpicoordy == 0)
	    {
            if (md.mpicoordx == md.npx-1)
                read_binary(lbc_s_full, "lbc_" + name + "_south", ncells_s);
            master.broadcast_x(lbc_s_full.data(), ncells_s, md.npx-1);

            copy_boundary(
                    lbc_s_in.at(name), lbc_s_full,
                    gd.itot+2*gd.igc, nlbc_s,
                    gd.icells, nlbc_s,
                    md.mpicoordx*gd.imax, 0,
                    name, "south");

            if (name == "v" && md.mpicoordx == md.npx-1)
                calc_div_y<TF, Lbc_location::South>(
                        div_v,
                        lbc_s_full.data(),
                        fields.rhoref.data(),
                        gd.dz.data(),
                        gd.dx,
                        n_sponge,
                        gd.jgc, gd.kgc,
                        gd.igc, gd.itot+gd.igc,
                        gd.ktot, gd.itot+(2*gd.igc));
	    }

        if (md.mpicoordy == md.npy-1)
	    {
            if (md.mpicoordx == 0)
                read_binary(lbc_n_full, "lbc_" + name + "_north", ncells_n);
            master.broadcast_x(lbc_n_full.data(), ncells_n, 0);

            copy_boundary(
                    lbc_n_in.at(name), lbc_n_full,
                    gd.itot+2*gd.igc, nlbc_n,
                    gd.icells, nlbc_n,
                    md.mpicoordx*gd.imax, 0,
                    name, "north");

            if (name == "v" && md.mpicoordx == 0)
                calc_div_y<TF, Lbc_location::North>(
                        div_v,
                        lbc_n_full.data(),
                        fields.rhoref.data(),
                        gd.dz.data(),
                        gd.dx,
                        n_sponge,
                        gd.jgc, gd.kgc,
                        gd.igc, gd.itot+gd.igc,
                        gd.ktot, gd.itot+(2*gd.igc));
	    }
    };

    if (sw_openbc_uv)
    {
        copy_boundaries("u");
        copy_boundaries("v");

        // Inflow is calculated at south+west edges, outflow at north+east edges.
        // Take sum, to get the net inflow in both directions.
        master.sum(&div_u, 1);
        master.sum(&div_v, 1);
    }

    if (sw_openbc_w)
        copy_boundaries("w");

    for (auto& fld : slist)
        copy_boundaries(fld);
}


template <typename TF>
void Boundary_lateral<TF>::create(
        Input& inputin,
        Timeloop<TF>& timeloop,
        Stats<TF>& stats,
        const std::string& sim_name)
{
    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    // Only proceed if open boundary conditions are enabled.
    if (!sw_openbc)
        return;

    TF* rhoref = fields.rhoref.data();

    // Domain total divergence in u and v direction.
    if (sw_openbc_uv)
    {
        w_top_2d.resize(gd.ijcells);

        if (sw_timedep)
        {
            w_top_2d_prev.resize(gd.ijcells);
            w_top_2d_next.resize(gd.ijcells);
        }
    }

    if (sw_lbc_tf_periodic)
    {
        // No lbc_* files: the buffers are refilled from the opposite edge
        // in set_ghost_cells. Only the surface height is needed.
        dem_tf.resize(gd.ijcells);
        std::fill(dem_tf.begin(), dem_tf.end(), TF(0));

        FILE* pfile = std::fopen(lbc_tf_dem_file.c_str(), "rb");
        const bool has_dem = (pfile != nullptr);
        if (pfile != nullptr)
            std::fclose(pfile);

        if (has_dem)
        {
            auto tmp = fields.get_tmp();
            if (field3d_io.load_xy_slice(dem_tf.data(), tmp->fld.data(), lbc_tf_dem_file.c_str()))
                throw std::runtime_error("sw_lbc_tf_periodic: reading " + lbc_tf_dem_file + " failed");
            fields.release_tmp(tmp);
        }

        master.print_message(
                "LBC: sw_lbc_tf_periodic ON - each edge is fed from the opposite edge (offset "
                + std::to_string(gd.itot - 2*n_sponge) + " x " + std::to_string(gd.jtot - 2*n_sponge)
                + " cells), matched on height above the surface"
                + (has_dem ? " from " + lbc_tf_dem_file : std::string(" (no DEM found: flat)")) + "\n");

        for (auto& it : lbc_tf_gradient)
            master.print_message(
                    "LBC: sw_lbc_tf_periodic - " + it.first + " shifted with the environment, d/dz = "
                    + std::to_string(it.second) + "\n");
    }
    else if (!sw_timedep)
    {
        // Read LBC data directly into `lbc_{w/e/n/s}`.
        TF div_u = 0;
        TF div_v = 0;
        const int iotime = 0;
        read_lbc(div_u, div_v, lbc_w, lbc_e, lbc_s, lbc_n, iotime);

        if (sw_wtop_2d)
            read_xy_slice(w_top_2d, "w_top", 0);
        else
        {
            const TF w_top_mean = -(div_u + div_v) / (fields.rhorefh[gd.kend] * gd.xsize * gd.ysize);
            std::fill(w_top_2d.begin(), w_top_2d.end(), w_top_mean);

            std::string message =
                    "- div(u) = " + std::to_string(div_u)
                  + ", div(v) = " + std::to_string(div_v)
                  + ", w_top = " + std::to_string(w_top_mean*100) + " cm/s";
            master.print_message(message);
        }
    }
    else
    {
        // Read previous and next input times.
        const double time = timeloop.get_time();
        const unsigned long itime = timeloop.get_itime();
        unsigned long iiotimeprec = timeloop.get_iiotimeprec();
        unsigned long iloadtime = convert_to_itime(loadfreq);

        // Determine time index for LBCs.
        const int prev_index = itime / iloadtime;
        const int next_index = prev_index + 1;

        // Determine `iotime` for `w_top` fields.
        prev_itime = prev_index * iloadtime;
        next_itime = next_index * iloadtime;

        unsigned long prev_iotime = prev_index * iloadtime / iiotimeprec;
        unsigned long next_iotime = next_index * iloadtime / iiotimeprec;

        // Read previous and next LBC values.
        TF div_u_prev = 0;
        TF div_v_prev = 0;

        TF div_u_next = 0;
        TF div_v_next = 0;

        read_lbc(div_u_prev, div_v_prev, lbc_w_prev, lbc_e_prev, lbc_s_prev, lbc_n_prev, prev_iotime);
        read_lbc(div_u_next, div_v_next, lbc_w_next, lbc_e_next, lbc_s_next, lbc_n_next, next_iotime);

        if (sw_wtop_2d)
        {
            read_xy_slice(w_top_2d_prev, "w_top", prev_iotime);
            read_xy_slice(w_top_2d_next, "w_top", next_iotime);
        }
        else
        {
            w_top_prev = -(div_u_prev + div_v_prev) / (fields.rhorefh[gd.kend] * gd.xsize * gd.ysize);
            w_top_next = -(div_u_next + div_v_next) / (fields.rhorefh[gd.kend] * gd.xsize * gd.ysize);

            std::string message1 =
                    "- div(u) = " + std::to_string(div_u_prev)
                    + ", div(v) = " + std::to_string(div_v_prev)
                    + ", w_top = " + std::to_string(w_top_prev*100)
                    + " cm/s @ t= " + std::to_string(prev_index*loadfreq) + " sec.";
            master.print_message(message1);

            std::string message2 =
                    "- div(u) = " + std::to_string(div_u_next)
                    + ", div(v) = " + std::to_string(div_v_next)
                    + ", w_top = " + std::to_string(w_top_next*100)
                    + " cm/s @ t= " + std::to_string(next_index*loadfreq) + " sec.";
            master.print_message(message2);
        }
    }

    if (sw_sponge)
    {
        stats.add_tendency(*fields.mt.at("u"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.mt.at("v"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.mt.at("w"), "zh", tend_name, tend_longname);

        for (auto& fld : slist)
            stats.add_tendency(*fields.at.at(fld), "zh", tend_name, tend_longname);
    }
}


template<typename TF>
unsigned long Boundary_lateral<TF>::get_time_limit(unsigned long itime)
{
    unsigned long idtlim = Constants::ulhuge;

    if (sw_openbc && sw_timedep)
    {
        const unsigned long ifreq = convert_to_itime(loadfreq);
        idtlim = std::min(idtlim, ifreq - itime % ifreq);
    }

    return idtlim;
}


template <typename TF>
void Boundary_lateral<TF>::set_ghost_cells_scalars()
{
    /* apply_lbc_ib_halos.py. The open-boundary ghost cells of the slist
       scalars, again. Immersed_boundary::exec_scalars ends with a
       boundary_cyclic exchange, which knows nothing about open boundaries
       and wraps the halo periodically: the ghost cells at an open edge then
       hold the field of the OPPOSITE edge, and advection, diffusion and the
       lateral sponge read that instead of the boundary values. Over sloping
       terrain the opposite edge at the same height is rock at its surface
       value. Called right after exec_scalars; refills only the scalars. */
    if (!sw_openbc)
        return;

    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    for (auto& fld : slist)
    {
        auto set = [&](std::vector<TF>& lbc, const Lbc_location location)
        {
            set_lbc_gcs(
                    fields.ap.at(fld)->fld.data(), lbc.data(),
                    gd.igc, n_sponge,
                    gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.kcells,
                    location);
        };
        if (md.mpicoordx == 0)
            set(lbc_w.at(fld), Lbc_location::West);
        if (md.mpicoordx == md.npx-1)
            set(lbc_e.at(fld), Lbc_location::East);
        if (md.mpicoordy == 0)
            set(lbc_s.at(fld), Lbc_location::South);
        if (md.mpicoordy == md.npy-1)
            set(lbc_n.at(fld), Lbc_location::North);
    }
}

template <typename TF>
void Boundary_lateral<TF>::set_ghost_cells(
        Timeloop<TF>& timeloop)
{
    if (!sw_openbc)
        return;

    // Terrain-following periodic LBCs are refreshed right before the
    // pressure solve (update_time_dependent with pres_fix), so that the
    // boundary-normal velocities the projection sees are the ones the field
    // keeps until the next solve. Refreshing them here as well, at the start
    // of a substep, changes the face velocities of a field that was just
    // made divergence free: in the B13 smoke test that left a divergence of
    // ~1e-2 1/s in the edge cells, i.e. a -thl*div source of K/s, and the
    // edges cooled by >10 K in five minutes. Only the very first call fills.
    if (sw_lbc_tf_periodic && !lbc_tf_initialized)
    {
        refresh_lbc_tf_periodic(TF(0));
        lbc_tf_initialized = true;
    }

    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    auto dump_vector = [&](
            std::vector<TF>& fld,
            const std::string& name)
    {
        std::string name_out = name + "." + std::to_string(md.mpicoordx) + "." + std::to_string(md.mpicoordy) + ".bin";

        FILE *pFile;
        pFile = fopen(name_out.c_str(), "wb");

        if (pFile == NULL)
            throw std::runtime_error("Opening raw dump field failed.");

        fwrite(fld.data(), sizeof(TF), fld.size(), pFile);
        fclose(pFile);
    };

    auto set_lbc_gcs_wrapper = [&](
            const std::string& fld,
            std::vector<TF>& lbc,
            const Lbc_location location)
    {
        int ngc = gd.igc;
        if (fld == "u" && location == Lbc_location::West)
            ngc += 1;
        if (fld == "v" && location == Lbc_location::South)
            ngc += 1;

        set_lbc_gcs(
                fields.ap.at(fld)->fld.data(),
                lbc.data(),
                ngc, n_sponge,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.jcells,
                gd.kcells,
                location);
    };

    auto set_gcs = [&](const std::string& fld)
    {
        if (md.mpicoordx == 0)
            set_lbc_gcs_wrapper(fld, lbc_w.at(fld), Lbc_location::West);
        if (md.mpicoordx == md.npx-1)
            set_lbc_gcs_wrapper(fld, lbc_e.at(fld), Lbc_location::East);
        if (md.mpicoordy == 0)
            set_lbc_gcs_wrapper(fld, lbc_s.at(fld), Lbc_location::South);
        if (md.mpicoordy == md.npy-1)
            set_lbc_gcs_wrapper(fld, lbc_n.at(fld), Lbc_location::North);
    };

    if (sw_openbc_uv)
    {
        set_gcs("u");
        set_gcs("v");
    }

    if (sw_openbc_w)
        set_gcs("w");

    for (auto& fld : slist)
        set_gcs(fld);


    // Set vertical velocity at domain top.
    if (sw_openbc_uv)
    {
        const int k = gd.kend;
        for (int j=gd.jstart; j<gd.jend; ++j)
            for (int i=gd.istart; i<gd.iend; ++i)
            {
                const int ijk = i + j*gd.icells + k*gd.ijcells;
                const int ij = i + j*gd.icells;
                fields.mp.at("w")->fld[ijk] = w_top_2d[ij];
            }
    }

    if (sw_neumann_w)
    {
        // Here, we enfore a Neumann BC of 0 over the boundaries. This works if the large scale w is approximately
        // constant in the horizontal plane. Note that w must be derived from u and v if the large-scale field is to
        // be divergence free. If there is a horizontal gradient in w_top, then it is probably better to extrapolate that
        // gradient into the ghost cells.
        auto set_ghost_cell_w_wrapper = [&]<Lbc_location location>()
        {
            set_ghost_cell_kernel_w<TF, location>(
                    fields.mp.at("w")->fld.data(),
                    gd.istart, gd.iend, gd.igc,
                    gd.jstart, gd.jend, gd.jgc,
                    gd.kstart, gd.kend+1,
                    gd.icells, gd.jcells, gd.kcells,
                    gd.ijcells);
        };

        auto set_corner_ghost_cell_wrapper = [&](
                std::vector<TF>& fld,
                const int kend)
        {
            set_corner_ghost_cell_kernel(
                    fld.data(),
                    md.mpicoordx, md.mpicoordy,
                    md.npx, md.npy,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, kend,
                    gd.icells, gd.jcells,
                    gd.kcells, gd.ijcells);
        };

        if (md.mpicoordx == 0)
            set_ghost_cell_w_wrapper.template operator()<Lbc_location::West>();
        if (md.mpicoordx == md.npx-1)
            set_ghost_cell_w_wrapper.template operator()<Lbc_location::East>();
        if (md.mpicoordy == 0)
            set_ghost_cell_w_wrapper.template operator()<Lbc_location::South>();
        if (md.mpicoordy == md.npy-1)
            set_ghost_cell_w_wrapper.template operator()<Lbc_location::North>();

        set_corner_ghost_cell_wrapper(fields.mp.at("w")->fld, gd.kend+1);
    }

    //dump_vector(fields.ap.at("u")->fld, "u");
    //dump_vector(fields.ap.at("v")->fld, "v");
    //dump_vector(fields.ap.at("w")->fld, "w");
    //dump_vector(fields.ap.at("thl")->fld, "thl");
    //dump_vector(fields.ap.at("qt")->fld, "qt");

    //throw 1;
}


template <typename TF>
void Boundary_lateral<TF>::exec_lateral_sponge(
        Stats<TF>& stats)
{
    if (!sw_openbc or !sw_sponge)
        return;

    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    auto sponge_layer_wrapper = [&]<Lbc_location location, bool sw_recycle>(
            std::map<std::string, std::vector<TF>>& lbc_map,
            const std::string& name)
    {
        const int kstart = (name == "w") ? gd.kstart+1 : gd.kstart;

        lateral_sponge_kernel_s<TF, location, sw_recycle>(
                fields.at.at(name)->fld.data(),
                fields.ap.at(name)->fld.data(),
                lbc_map.at(name).data(),
                tau_sponge,
                w_diff,
                n_sponge,
                tau_recycle,
                recycle_offset,
                md.npx, md.npy,
                md.mpicoordx, md.mpicoordy,
                gd.igc, gd.jgc,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                kstart, gd.kend,
                gd.icells, gd.jcells,
                gd.ijcells);
    };


    auto sponge_layer_u_wrapper = [&]<Lbc_location location>(
            std::map<std::string, std::vector<TF>>& lbc_map)
    {
        lateral_sponge_kernel_u<TF, location>(
                fields.mt.at("u")->fld.data(),
                fields.mp.at("u")->fld.data(),
                lbc_map.at("u").data(),
                tau_sponge,
                w_diff,
                n_sponge,
                md.npy,
                md.mpicoordy,
                gd.igc,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.jcells,
                gd.ijcells);
    };


    auto sponge_layer_v_wrapper = [&]<Lbc_location location>(
            std::map<std::string, std::vector<TF>>& lbc_map)
    {
        lateral_sponge_kernel_v<TF, location>(
                fields.mt.at("v")->fld.data(),
                fields.mp.at("v")->fld.data(),
                lbc_map.at("v").data(),
                tau_sponge,
                w_diff,
                n_sponge,
                md.npx,
                md.mpicoordx,
                gd.jgc,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.jcells,
                gd.ijcells);
    };

    if (sw_openbc_uv)
    {
        // NOTE: order of calls here is important; the `_u` and `_v` wrappers don't take
        //       the offset in corners into account, so always first call the `_u` and `_v`
        //       wrappers, followed by the generic `sponge_layer_wrapper()`.

        if (md.mpicoordx == 0)
            sponge_layer_u_wrapper.template operator()<Lbc_location::West>(lbc_w);
        if (md.mpicoordx == md.npx-1)
            sponge_layer_u_wrapper.template operator()<Lbc_location::East>(lbc_e);
        if (md.mpicoordy == 0)
            sponge_layer_wrapper.template operator()<Lbc_location::South, false>(lbc_s, "u");
        if (md.mpicoordy == md.npy-1)
            sponge_layer_wrapper.template operator()<Lbc_location::North, false>(lbc_n, "u");

        if (md.mpicoordy == 0)
            sponge_layer_v_wrapper.template operator()<Lbc_location::South>(lbc_s);
        if (md.mpicoordy == md.npy-1)
            sponge_layer_v_wrapper.template operator()<Lbc_location::North>(lbc_n);
        if (md.mpicoordx == 0)
            sponge_layer_wrapper.template operator()<Lbc_location::West, false>(lbc_w, "v");
        if (md.mpicoordx == md.npx-1)
            sponge_layer_wrapper.template operator()<Lbc_location::East, false>(lbc_e, "v");

        stats.calc_tend(*fields.mt.at("u"), tend_name);
        stats.calc_tend(*fields.mt.at("v"), tend_name);
    }

    if (sw_openbc_w)
    {
        if (md.mpicoordx == 0)
            sponge_layer_wrapper.template operator()<Lbc_location::West, false>(lbc_w, "w");
        if (md.mpicoordx == md.npx-1)
            sponge_layer_wrapper.template operator()<Lbc_location::East, false>(lbc_e, "w");
        if (md.mpicoordy == 0)
            sponge_layer_wrapper.template operator()<Lbc_location::South, false>(lbc_s, "w");
        if (md.mpicoordy == md.npy-1)
            sponge_layer_wrapper.template operator()<Lbc_location::North, false>(lbc_n, "w");

        stats.calc_tend(*fields.mt.at("w"), tend_name);
    }

    for (auto& fld : slist)
    {
        const bool sw_recycle_fld = in_list<std::string>(fld, recycle_list);

        if (md.mpicoordx == 0)
        {
            if (sw_recycle_fld && sw_recycle[Lbc_location::West])
                sponge_layer_wrapper.template operator()<Lbc_location::West, true>(lbc_w, fld);
            else
                sponge_layer_wrapper.template operator()<Lbc_location::West, false>(lbc_w, fld);
        }
        // The plain sponge runs on EVERY edge; sw_recycle only chooses the
        // kernel, inside. It used to gate the whole sponge at east, south and
        // north, so with recycling off thl/qt were relaxed at the west edge
        // only. See apply_lbc_sponge_edges.py.
        if (md.mpicoordx == md.npx-1)
        {
            if (sw_recycle_fld && sw_recycle[Lbc_location::East])
                sponge_layer_wrapper.template operator()<Lbc_location::East, true>(lbc_e, fld);
            else
                sponge_layer_wrapper.template operator()<Lbc_location::East, false>(lbc_e, fld);
        }
        if (md.mpicoordy == 0)
        {
            if (sw_recycle_fld && sw_recycle[Lbc_location::South])
                sponge_layer_wrapper.template operator()<Lbc_location::South, true>(lbc_s, fld);
            else
                sponge_layer_wrapper.template operator()<Lbc_location::South, false>(lbc_s, fld);
        }
        if (md.mpicoordy == md.npy-1)
        {
            if (sw_recycle_fld && sw_recycle[Lbc_location::North])
                sponge_layer_wrapper.template operator()<Lbc_location::North, true>(lbc_n, fld);
            else
                sponge_layer_wrapper.template operator()<Lbc_location::North, false>(lbc_n, fld);
        }

        stats.calc_tend(*fields.at.at(fld), tend_name);
    }
}


template <typename TF>
void Boundary_lateral<TF>::update_time_dependent(
        Timeloop<TF>& timeloop,
        const bool pres_fix)
{
    if (sw_openbc && sw_lbc_tf_periodic && pres_fix)
    {
        refresh_lbc_tf_periodic(TF(timeloop.get_sub_time_step()));
        lbc_tf_initialized = true;
        return;
    }

    if (!sw_openbc || !sw_timedep)
        return;

    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    // Find index in time array.
    double time = timeloop.get_time();
    double endtime = timeloop.get_endtime();
    unsigned long itime = timeloop.get_itime();
    unsigned long iiotimeprec = timeloop.get_iiotimeprec();
    unsigned long iloadtime = convert_to_itime(loadfreq);
    unsigned long iendtime = convert_to_itime(endtime);

    // CvH: this is an UGLY hack, because it only works for RK4.
    // We need to know from the time whether we are in the last iter.
    // Also, it will fail miserably if we perturb the velocities at the walls
    const int rkorder = timeloop.get_rkorder();
    const int last_substep = rkorder == 4 ? 4 : 2;
    if (pres_fix && timeloop.get_substep() == last_substep)
    {
        time += timeloop.get_dt();
        itime += timeloop.get_idt();
    }

    if (itime >= next_itime)
    {
        // BvS: We are setting `w_top` for the next time step;
        //      skip if next time is beyond the endtime.
        if (next_itime + iloadtime > iendtime)
            master.print_warning("Timedep boundary_lateral out-of-bounds at t+dt, skipping update.\n");
        else
        {
            // Advance time and read new files.
            prev_itime = next_itime;
            next_itime = prev_itime + iloadtime;

            unsigned long next_iotime = next_itime / iiotimeprec;

            // Move LBCs from next to previous values.
            for (auto& it : lbc_w_next)
                lbc_w_prev.at(it.first) = it.second;
            for (auto& it : lbc_e_next)
                lbc_e_prev.at(it.first) = it.second;
            for (auto& it : lbc_s_next)
                lbc_s_prev.at(it.first) = it.second;
            for (auto& it : lbc_n_next)
                lbc_n_prev.at(it.first) = it.second;

            // Read new LBC values.
            const int next_index = next_itime / iloadtime;

            TF div_u_next = 0;
            TF div_v_next = 0;
            read_lbc(div_u_next, div_v_next, lbc_w_next, lbc_e_next, lbc_s_next, lbc_n_next, next_iotime);

            // Read in or calculate new `w_top`.
            if (sw_wtop_2d)
            {
                w_top_2d_prev = w_top_2d_next;
                read_xy_slice(w_top_2d_next, "w_top", next_iotime);
            }
            else
            {
                w_top_prev = w_top_next;
                w_top_next = -(div_u_next + div_v_next) / (fields.rhorefh[gd.kend] * gd.xsize * gd.ysize);

                std::string message2 =
                        "- div(u) = " + std::to_string(div_u_next)
                        + ", div(v) = " + std::to_string(div_v_next)
                        + ", w_top = " + std::to_string(w_top_next*100)
                        + " cm/s @ t= " + std::to_string(next_index*loadfreq) + " sec.";
                master.print_message(message2);
            }
        }
    }

    // Interpolate LBCs and w_top to current time.
    const TF f0 = TF(1) - ((itime - prev_itime) / TF(iloadtime));
    const TF f1 = TF(1) - f0;

    // Interpolate mean domain top velocity
    if (sw_wtop_2d)
    {
        // Interpolate `w_top` field in time.
        for (int n=0; n<gd.ijcells; ++n)
            w_top_2d[n] = f0 * w_top_2d_prev[n] + f1 * w_top_2d_next[n];
    }
    else
    {
        const TF w_top = f0 * w_top_prev + f1 * w_top_next;
        std::fill(w_top_2d.begin(), w_top_2d.end(), w_top);
    }

    // Interpolate boundaries in time.
    if (md.mpicoordx == 0)
    {
        for (auto& it : lbc_w)
        {
            const int ngc = (it.first == "u") ? gd.igc+1 : gd.igc;
            interpolate_lbc_kernel(
                    lbc_w.at(it.first).data(),
                    lbc_w_prev.at(it.first).data(),
                    lbc_w_next.at(it.first).data(),
                    gd.kcells * gd.jcells * (ngc+n_sponge),
                    f0);
        }
    }

    if (md.mpicoordx == md.npx-1)
    {
        for (auto& it : lbc_e)
            interpolate_lbc_kernel(
                    lbc_e.at(it.first).data(),
                    lbc_e_prev.at(it.first).data(),
                    lbc_e_next.at(it.first).data(),
                    gd.kcells * gd.jcells * (gd.igc+n_sponge),
                    f0);
    }

    if (md.mpicoordy == 0)
    {
        for (auto& it : lbc_s)
        {
            const int ngc = (it.first == "v") ? gd.jgc+1 : gd.jgc;
            interpolate_lbc_kernel(
                    lbc_s.at(it.first).data(),
                    lbc_s_prev.at(it.first).data(),
                    lbc_s_next.at(it.first).data(),
                    gd.kcells * (ngc+n_sponge) * gd.icells,
                    f0);
        }
    }

    if (md.mpicoordy == md.npy-1)
    {
        for (auto& it : lbc_n)
            interpolate_lbc_kernel(
                    lbc_n.at(it.first).data(),
                    lbc_n_prev.at(it.first).data(),
                    lbc_n_next.at(it.first).data(),
                    gd.kcells * (gd.jgc+n_sponge) * gd.icells,
                    f0);
    }
}


template <typename TF>
void Boundary_lateral<TF>::read_xy_slice(
        std::vector<TF>& field,
        const std::string& name,
        const int time)
{
    char filename[256];
    std::sprintf(filename, "%s.%07d", name.c_str(), time);
    master.print_message("Loading \"%s\" ... ", filename);

    auto tmp  = fields.get_tmp();

    if (field3d_io.load_xy_slice(field.data(), tmp->fld.data(), filename))
        master.print_message("FAILED\n");
    else
        master.print_message("OK\n");

    fields.release_tmp(tmp);
}


template <typename TF>
void Boundary_lateral<TF>::refresh_lbc_tf_periodic(const TF dt_sub)
{
    /* Terrain-following periodic lateral boundaries (patch 28).

       The un-sponged interior is treated as periodic: the buffer of an
       edge (ghost + sponge cells) is filled from the band of the same width
       just inside the sponge of the opposite edge, an offset of
       itot - 2*n_sponge cells in x (jtot - 2*n_sponge in y). Each buffer
       column takes the source column's value at the same height ABOVE ITS
       OWN SURFACE, interpolated linearly in z, so a sloping domain can be
       closed on itself. Scalars are shifted with the environment by
       lbc_tf_gradient * (zs_buffer - zs_source), which keeps their anomaly
       relative to a stratified environment. Cells inside the terrain are
       treated the same way, so the IB ghost and blanked cells of the source
       carry over: setting them to anything else (e.g. zero momentum) breaks
       the mirror condition the IB imposes at the edge rows and drives a
       spurious vertical jet along the boundary.

       Momentum is taken from the PROVISIONAL field u + dt_sub*ut, the field
       the pressure solve is about to project. A periodic projection leaves
       the plane-mean velocity alone; an open one pins the flux through every
       plane to the boundary faces. Feeding the faces the provisional band
       flux makes the pinned value the one a periodic solve would keep, so
       the column-mean flow evolves by its own momentum budget (Coriolis,
       friction, large-scale pressure gradient) instead of being frozen at,
       or leaking from, the value of the previous step. */

    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    std::vector<std::string> names;
    if (sw_openbc_uv)
    {
        names.push_back("u");
        names.push_back("v");
    }
    if (sw_openbc_w)
        names.push_back("w");
    for (auto& fld : slist)
        names.push_back(fld);

    const int nfld = names.size();
    const int ns = n_sponge;

    // Surface height of a local column; halo columns take the nearest interior one.
    auto zs_local = [&](const int i, const int j)
    {
        const int ic = std::min(std::max(i, gd.istart), gd.iend-1);
        const int jc = std::min(std::max(j, gd.jstart), gd.jend-1);
        return dem_tf[ic + jc*gd.icells];
    };

    // Value of a source column at height zt above the domain floor. The
    // column is strided by `stride` in the packed band, one value per k.
    // Below the lowest level the lowest value is used; above the top level
    // the top value plus `grad_top`*(zt - z_top), so a scalar continues
    // along its environmental gradient and momentum stays constant.
    auto interp_column = [&](
            const TF* const col, const int stride, const std::vector<TF>& zl,
            const TF zt, const TF grad_top, TF& value)
    {
        const int kfa = gd.kstart;
        const TF z_lo = zl[kfa];
        const TF z_hi = zl[gd.kend-1];
        const TF zc = std::min(std::max(zt, z_lo), z_hi);

        int k0 = kfa;
        while (k0 < gd.kend-2 && zl[k0+1] <= zc)
            ++k0;
        const int k1 = std::min(k0+1, gd.kend-1);

        const TF f = (k1 == k0) ? TF(0) : (zc - zl[k0]) / (zl[k1] - zl[k0]);
        value = (TF(1)-f)*col[(k0-gd.kstart)*stride] + f*col[(k1-gd.kstart)*stride];

        if (zt > z_hi)
            value += grad_top * (zt - z_hi);
    };

    // Fill one buffer column (k = kstart..kend-1) from one packed source column.
    auto fill_column = [&](
            TF* const out, const int out_kstride,
            const TF* const col, const int col_kstride,
            const std::string& name,
            const TF zs_own, const TF zs_src)
    {
        const bool is_mom = (name == "u" || name == "v" || name == "w");
        const std::vector<TF>& zl = (name == "w") ? gd.zh : gd.z;
        const TF grad = is_mom ? TF(0) : lbc_tf_gradient.at(name);
        const TF offset = grad * (zs_own - zs_src);

        for (int k=gd.kstart; k<gd.kend; ++k)
        {
            TF value;
            interp_column(col, col_kstride, zl, zl[k] - zs_own + zs_src, grad, value);
            out[k*out_kstride] = value + offset;
        }
    };

    const int ktot = gd.kend - gd.kstart;

    // Exchange two packed bands between the ranks at the two ends of a
    // communicator. With one rank in that direction the copy is local.
    auto exchange = [&](
            std::vector<TF>& send_lo, std::vector<TF>& recv_lo,
            std::vector<TF>& send_hi, std::vector<TF>& recv_hi,
            const bool is_lo, const bool is_hi, const int nproc, const bool along_x)
    {
        if (nproc == 1)
        {
            recv_lo = send_hi;
            recv_hi = send_lo;
            return;
        }
        #ifdef USEMPI
        MPI_Comm comm = along_x ? md.commx : md.commy;
        if (is_lo)
            MPI_Sendrecv(
                    send_lo.data(), send_lo.size(), mpi_fp_type<TF>(), nproc-1, 28,
                    recv_lo.data(), recv_lo.size(), mpi_fp_type<TF>(), nproc-1, 28,
                    comm, MPI_STATUS_IGNORE);
        if (is_hi)
            MPI_Sendrecv(
                    send_hi.data(), send_hi.size(), mpi_fp_type<TF>(), 0, 28,
                    recv_hi.data(), recv_hi.size(), mpi_fp_type<TF>(), 0, 28,
                    comm, MPI_STATUS_IGNORE);
        #endif
    };

    // X DIRECTION. Packed band layout: [fld][k][j][c], then zs[j][c].
    {
        const bool is_w = (md.mpicoordx == 0);
        const bool is_e = (md.mpicoordx == md.npx-1);

        const int nb_w = gd.igc + 1 + ns;   // Widest west buffer (u).
        const int nb_e = gd.igc + ns;
        const int i0_for_w = gd.imax - 2*ns;    // Band on the east rank that feeds the west buffer.
        const int i0_for_e = gd.igc + ns;       // Band on the west rank that feeds the east buffer.

        auto pack = [&](std::vector<TF>& buf, const int i0, const int nb)
        {
            buf.resize((nfld*ktot + 1) * gd.jcells * nb);
            int n = 0;
            for (auto& name : names)
            {
                const TF* const fld = fields.ap.at(name)->fld.data();
                const TF* const tnd = fields.at.at(name)->fld.data();
                const TF fac = (name == "u" || name == "v" || name == "w") ? dt_sub : TF(0);
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int j=0; j<gd.jcells; ++j)
                        for (int c=0; c<nb; ++c)
                        {
                            const int ijk = (i0+c) + j*gd.icells + k*gd.ijcells;
                            buf[n++] = fld[ijk] + fac*tnd[ijk];
                        }
            }
            for (int j=0; j<gd.jcells; ++j)
                for (int c=0; c<nb; ++c)
                    buf[n++] = zs_local(i0+c, j);
        };

        std::vector<TF> send_w, recv_w, send_e, recv_e;
        if (is_e)
            pack(send_e, i0_for_w, nb_w);
        if (is_w)
            pack(send_w, i0_for_e, nb_e);
        if (is_w)
            recv_w.resize((nfld*ktot + 1) * gd.jcells * nb_w);
        if (is_e)
            recv_e.resize((nfld*ktot + 1) * gd.jcells * nb_e);

        exchange(send_w, recv_w, send_e, recv_e, is_w, is_e, md.npx, true);

        auto unpack = [&](Lbc_map<TF>& lbc, const std::vector<TF>& recv, const int nb, const bool west)
        {
            const int zs_off = nfld*ktot*gd.jcells*nb;
            for (int f=0; f<nfld; ++f)
            {
                const std::string& name = names[f];
                const int ngc_pad = (west && name == "u") ? gd.igc+1 : gd.igc;
                const int nlbc = ngc_pad + ns;
                std::vector<TF>& out = lbc.at(name);
                const TF* const band = recv.data() + f*ktot*gd.jcells*nb;

                // Rows outside [jstart, jend) are the corner regions shared
                // with the y-buffers: they take the nearest interior row of
                // this buffer, not the source's ghost rows (those are LBC
                // values themselves, and copying them closes a loop).
                for (int j=0; j<gd.jcells; ++j)
                    for (int b=0; b<nlbc; ++b)
                    {
                        const int jc = std::min(std::max(j, gd.jstart), gd.jend-1);
                        // Local column index of buffer cell b, and its source column c.
                        const int i = west ? b : gd.iend - ns + b;
                        const int c = b;
                        const TF zs_src = recv[zs_off + c + jc*nb];
                        fill_column(
                                out.data() + b + j*nlbc, nlbc*gd.jcells,
                                band + c + jc*nb, gd.jcells*nb,
                                name, zs_local(i, jc), zs_src);
                    }
            }
        };

        if (is_w)
            unpack(lbc_w, recv_w, nb_w, true);
        if (is_e)
            unpack(lbc_e, recv_e, nb_e, false);
    }

    // Y DIRECTION. Packed band layout: [fld][k][c][i], then zs[c][i].
    {
        const bool is_s = (md.mpicoordy == 0);
        const bool is_n = (md.mpicoordy == md.npy-1);

        const int nb_s = gd.jgc + 1 + ns;   // Widest south buffer (v).
        const int nb_n = gd.jgc + ns;
        const int j0_for_s = gd.jmax - 2*ns;
        const int j0_for_n = gd.jgc + ns;

        auto pack = [&](std::vector<TF>& buf, const int j0, const int nb)
        {
            buf.resize((nfld*ktot + 1) * gd.icells * nb);
            int n = 0;
            for (auto& name : names)
            {
                const TF* const fld = fields.ap.at(name)->fld.data();
                const TF* const tnd = fields.at.at(name)->fld.data();
                const TF fac = (name == "u" || name == "v" || name == "w") ? dt_sub : TF(0);
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int c=0; c<nb; ++c)
                        for (int i=0; i<gd.icells; ++i)
                        {
                            const int ijk = i + (j0+c)*gd.icells + k*gd.ijcells;
                            buf[n++] = fld[ijk] + fac*tnd[ijk];
                        }
            }
            for (int c=0; c<nb; ++c)
                for (int i=0; i<gd.icells; ++i)
                    buf[n++] = zs_local(i, j0+c);
        };

        std::vector<TF> send_s, recv_s, send_n, recv_n;
        if (is_n)
            pack(send_n, j0_for_s, nb_s);
        if (is_s)
            pack(send_s, j0_for_n, nb_n);
        if (is_s)
            recv_s.resize((nfld*ktot + 1) * gd.icells * nb_s);
        if (is_n)
            recv_n.resize((nfld*ktot + 1) * gd.icells * nb_n);

        exchange(send_s, recv_s, send_n, recv_n, is_s, is_n, md.npy, false);

        auto unpack = [&](Lbc_map<TF>& lbc, const std::vector<TF>& recv, const int nb, const bool south)
        {
            const int zs_off = nfld*ktot*gd.icells*nb;
            for (int f=0; f<nfld; ++f)
            {
                const std::string& name = names[f];
                const int ngc_pad = (south && name == "v") ? gd.jgc+1 : gd.jgc;
                const int nlbc = ngc_pad + ns;
                std::vector<TF>& out = lbc.at(name);
                const TF* const band = recv.data() + f*ktot*gd.icells*nb;

                for (int b=0; b<nlbc; ++b)
                    for (int i=0; i<gd.icells; ++i)
                    {
                        const int ic = std::min(std::max(i, gd.istart), gd.iend-1);
                        const int j = south ? b : gd.jend - ns + b;
                        const int c = b;
                        const TF zs_src = recv[zs_off + ic + c*gd.icells];
                        fill_column(
                                out.data() + i + b*gd.icells, gd.icells*nlbc,
                                band + ic + c*gd.icells, gd.icells*nb,
                                name, zs_local(ic, j), zs_src);
                    }
            }
        };

        if (is_s)
            unpack(lbc_s, recv_s, nb_s, true);
        if (is_n)
            unpack(lbc_n, recv_n, nb_n, false);
    }

    // Close the mass balance. The two faces of an edge pair are fed from
    // DIFFERENT planes (the bands just inside the opposite sponges), so their
    // instantaneous turbulent mass fluxes differ. Left to w_top, that
    // imbalance pumped the whole column at O(1 m/s) in the B13 smoke test.
    // A periodic domain has no net lateral divergence, so remove it per edge
    // pair instead: a uniform correction on the air cells of both faces, in
    // opposite directions, so that inflow equals outflow; w_top is then zero.
    if (sw_openbc_uv && !sw_wtop_2d)
    {
        auto balance = [&](
                Lbc_map<TF>& lbc_lo, Lbc_map<TF>& lbc_hi,
                const std::string& name, const bool along_x,
                const bool is_lo, const bool is_hi)
        {
            // [0] flux lo, [1] flux hi, [2] air area lo, [3] air area hi.
            TF acc[4] = {TF(0), TF(0), TF(0), TF(0)};

            const int ngc = along_x ? gd.igc : gd.jgc;
            const int nlbc_lo = ngc + 1 + ns;
            const int nlbc_hi = ngc + ns;
            const int nalong = along_x ? gd.jcells : gd.icells;
            const int along_start = along_x ? gd.jstart : gd.istart;
            const int along_end = along_x ? gd.jend : gd.iend;
            const TF dl = along_x ? gd.dy : gd.dx;

            // Buffer index of buffer cell (b, a, k) and the column it sits in.
            auto index = [&](const int b, const int a, const int k, const int nlbc)
            {
                return along_x ? b + a*nlbc + k*nlbc*nalong : a + b*nalong + k*nalong*nlbc;
            };
            auto zs_cell = [&](const int b, const int a, const bool hi)
            {
                const int n = hi ? (along_x ? gd.iend : gd.jend) - ns + b : b;
                return along_x ? zs_local(n, a) : zs_local(a, n);
            };

            // The flux is summed over the WHOLE face, terrain included: the
            // pressure solve sees the IB ghost values there too, and the
            // Neumann (DCT) problem is only solvable if the net lateral
            // inflow is exactly zero (with w_top = 0). Any residual becomes a
            // uniform divergence, i.e. a spurious -thl*div source everywhere.
            // The correction itself is spread over the air cells only.
            auto add = [&](const std::vector<TF>& buf, const int b_face, const int nlbc,
                           const bool hi, TF& flux, TF& area)
            {
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int a=along_start; a<along_end; ++a)
                    {
                        const TF w = fields.rhoref[k] * dl * gd.dz[k];
                        flux += w * buf[index(b_face, a, k, nlbc)];
                        if (gd.z[k] > zs_cell(b_face, a, hi))
                            area += w;
                    }
            };

            if (is_lo)
                add(lbc_lo.at(name), ngc, nlbc_lo, false, acc[0], acc[2]);
            if (is_hi)
                add(lbc_hi.at(name), ns, nlbc_hi, true, acc[1], acc[3]);

            master.sum(acc, 4);

            const TF area = acc[2] + acc[3];
            if (area <= TF(0))
                return;
            const TF delta = (acc[1] - acc[0]) / area;

            auto shift = [&](std::vector<TF>& buf, const int nlbc, const bool hi, const TF d)
            {
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int a=0; a<nalong; ++a)
                        for (int b=0; b<nlbc; ++b)
                            if (gd.z[k] > zs_cell(b, a, hi))
                                buf[index(b, a, k, nlbc)] += d;
            };

            if (is_lo)
                shift(lbc_lo.at(name), nlbc_lo, false, delta);
            if (is_hi)
                shift(lbc_hi.at(name), nlbc_hi, true, -delta);
        };

        // apply_ib_wall_kinematic.py: air that leaves through the floor of
        // a sloping IB surface (or enters from it) passes through the rock,
        // and the rock is closed everywhere except at the domain edges. The
        // projection would remove the net part, which on a plane slope is
        // the mean descent itself. So put the net air <- rock flux F of each
        // direction on the rock part of that edge pair, half at each face
        // (rock inflow F/2 at the low face, outflow -F/2 at the high face).
        // The whole-face balance below then hands F back to the air at the
        // opposite edge: on a plane slope this is the wrap of the
        // terrain-following periodicity, the air that slid down by
        // z_s(west) - z_s(east) re-entering at the top of the step.
        auto rock_export = [&](
                Lbc_map<TF>& lbc_lo, Lbc_map<TF>& lbc_hi,
                const std::string& name, const bool along_x,
                const bool is_lo, const bool is_hi, const TF f_rock)
        {
            const int ngc = along_x ? gd.igc : gd.jgc;
            const int nlbc_lo = ngc + 1 + ns;
            const int nlbc_hi = ngc + ns;
            const int nalong = along_x ? gd.jcells : gd.icells;
            const int along_start = along_x ? gd.jstart : gd.istart;
            const int along_end = along_x ? gd.jend : gd.iend;
            const TF dl = along_x ? gd.dy : gd.dx;

            auto index = [&](const int b, const int a, const int k, const int nlbc)
            {
                return along_x ? b + a*nlbc + k*nlbc*nalong : a + b*nalong + k*nalong*nlbc;
            };
            auto zs_cell = [&](const int b, const int a, const bool hi)
            {
                const int n = hi ? (along_x ? gd.iend : gd.jend) - ns + b : b;
                return along_x ? zs_local(n, a) : zs_local(a, n);
            };

            // [0] rock area lo, [1] rock area hi (mass weighted).
            TF acc[2] = {TF(0), TF(0)};
            if (is_lo)
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int a=along_start; a<along_end; ++a)
                        if (gd.z[k] <= zs_cell(ngc, a, false))
                            acc[0] += fields.rhoref[k] * dl * gd.dz[k];
            if (is_hi)
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int a=along_start; a<along_end; ++a)
                        if (gd.z[k] <= zs_cell(ns, a, true))
                            acc[1] += fields.rhoref[k] * dl * gd.dz[k];
            master.sum(acc, 2);

            // All of it through one face if the other has no rock.
            if (acc[0] + acc[1] <= TF(0))
                return;
            const TF f_lo = (acc[1] <= TF(0)) ? f_rock : (acc[0] <= TF(0)) ? TF(0) : TF(0.5)*f_rock;
            const TF f_hi = f_lo - f_rock;
            const TF u_lo = (acc[0] > TF(0)) ? f_lo / acc[0] : TF(0);
            const TF u_hi = (acc[1] > TF(0)) ? f_hi / acc[1] : TF(0);

            // The face and the ghost faces outside it; the sponge cells
            // inside keep the source values (the IB blanks them anyway).
            if (is_lo)
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int a=0; a<nalong; ++a)
                        for (int b=0; b<=ngc; ++b)
                            if (gd.z[k] <= zs_cell(ngc, a, false))
                                lbc_lo.at(name)[index(b, a, k, nlbc_lo)] = u_lo;
            if (is_hi)
                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int a=0; a<nalong; ++a)
                        for (int b=ns; b<nlbc_hi; ++b)
                            if (gd.z[k] <= zs_cell(ns, a, true))
                                lbc_hi.at(name)[index(b, a, k, nlbc_hi)] = u_hi;

            if (!rock_export_reported)
                master.print_message(
                        "LBC: sw_lbc_tf_periodic - terrain mass flux %.4g kg/s let "
                        "out of the rock through the %s edges (%.3g / %.3g m/s)\n",
                        double(f_rock), along_x ? "x" : "y", double(u_lo), double(u_hi));
        };

        // Also with a zero flux: the copied source values in the terrain
        // part of the faces would otherwise enter the balance of the air.
        if (sw_lbc_rock_export)
        {
            rock_export(lbc_w, lbc_e, "u", true,  md.mpicoordx == 0, md.mpicoordx == md.npx-1, rock_flux[0]);
            rock_export(lbc_s, lbc_n, "v", false, md.mpicoordy == 0, md.mpicoordy == md.npy-1, rock_flux[1]);
            rock_export_reported = true;
        }

        balance(lbc_w, lbc_e, "u", true, md.mpicoordx == 0, md.mpicoordx == md.npx-1);
        balance(lbc_s, lbc_n, "v", false, md.mpicoordy == 0, md.mpicoordy == md.npy-1);

        std::fill(w_top_2d.begin(), w_top_2d.end(), TF(0));
    }
}

#ifdef FLOAT_SINGLE
template class Boundary_lateral<float>;
#else
template class Boundary_lateral<double>;
#endif
