/*
 * MicroHH
 * Copyright (c) 2011-2024 Chiel van Heerwaarden
 * Copyright (c) 2011-2024 Thijs Heus
 * Copyright (c) 2014-2024 Bart van Stratum
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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <map> 

#include <constants.h>
#include "master.h"
#include "grid.h"
#include "fields.h"
#include "input.h"
#include "immersed_boundary.h"
#include "column.h"
#include <limits>
#include "timeloop.h"
#include "thermo.h"
#include "monin_obukhov.h"
#include "boundary_surface_kernels.h"
#include "fast_math.h"
#include "stats.h"
#include "cross.h"
#include <limits>
#include "finite_difference.h"   // interp6_ws/interp5_ws/interp4_ws/interp3_ws
#include "advec_monotonic.h"     // Koren flux_lim, as advec_2i5 uses it

namespace
{
    template<typename TF>
    TF absolute_distance(
            const TF x1, const TF y1, const TF z1,
            const TF x2, const TF y2, const TF z2)
    {
        return std::pow(Fast_math::pow2(x2-x1) + Fast_math::pow2(y2-y1) + Fast_math::pow2(z2-z1), TF(0.5));
    }

    // Help function for sorting std::vector with Neighbour points
    template<typename TF>
    bool compare_value(const Neighbour<TF>& a, const Neighbour<TF>& b)
    {
        return a.distance < b.distance;
    }

    bool has_ending(const std::string& full_string, const std::string& ending)
    {
        if (full_string.length() >= ending.length())
            return (0 == full_string.compare(full_string.length() - ending.length(), ending.length(), ending));
        else
            return false;
    };

    /* Bi-linear interpolation of the 2D IB DEM
     * onto the requested (`x_goal`, `y_goal`) location */
    template<typename TF>
    TF interp2_dem(
            const TF x_goal, const TF y_goal,
            const std::vector<TF>& x, const std::vector<TF>& y, const std::vector<TF>& dem,
            const TF dx, const TF dy,
            const int icells, const int jcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        const int ii = 1;
        const int jj = icells;

        // Indices west and south of `x_goal`, `y_goal`
        int i0 = (x_goal - TF(0.5)*dx) / dx + mpi_offset_x;
        int j0 = (y_goal - TF(0.5)*dy) / dy + mpi_offset_y;

        // Account for interpolation in last ghost cell (east and north)
        if (i0 == icells-1)
            i0 -= 1;
        if (j0 == jcells-1)
            j0 -= 1;

        const int ij = i0 + j0*jj;

        // Bounds check...
        if (i0 < 0 or i0 >= icells-1 or j0 < 0 or j0 >= jcells-1)
        {
            std::string error = "IB dem interpolation out of bounds!";
            throw std::runtime_error(error);
        }

        // Interpolation factors
        const TF f1x = (x_goal - x[i0]) / dx;
        const TF f1y = (y_goal - y[j0]) / dy;
        const TF f0x = TF(1) - f1x;
        const TF f0y = TF(1) - f1y;

        const TF z = f0y * (f0x * dem[ij   ] + f1x * dem[ij+ii   ]) +
                     f1y * (f0x * dem[ij+jj] + f1x * dem[ij+ii+jj]);
        return z;
    }

    template<typename TF>
    bool is_ghost_cell(
            const std::vector<TF>& dem,
            const std::vector<TF>& x, const std::vector<TF>& y, const std::vector<TF>& z,
            const TF dx, const TF dy,
            const int i, const int j, const int k,
            const int icells, const int jcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        const TF zdem = interp2_dem(
                x[i], y[j], x, y, dem, dx, dy,
                icells, jcells, mpi_offset_x, mpi_offset_y);

        // Check if grid point is below IB. If so; check if
        // one of the neighbouring grid points is outside.
        if (z[k] <= zdem)
        {
            // OLD METHOD:
            //if (z[k+1] > zdem)
            //    return true;

            //for (int dj = -1; dj <= 1; ++dj)
            //{
            //    const TF zdem = interp2_dem(x[i], y[j+dj], x, y, dem, dx, dy,
            //                                icells, mpi_offset_x, mpi_offset_y);
            //    if (z[k] > zdem)
            //        return true;
            //}

            //for (int di = -1; di <= 1; ++di)
            //{
            //    // Interpolate DEM to account for half-level locations x,y
            //    const TF zdem = interp2_dem(x[i+di], y[j], x, y, dem, dx, dy,
            //                                icells, mpi_offset_x, mpi_offset_y);
            //    if (z[k] > zdem)
            //        return true;
            //}

            //// NEW METHOD
            //for (int dj = -1; dj <= 1; ++dj)
            //{
            //    // Interpolate DEM to account for half-level locations x,y
            //    const TF zdem = interp2_dem(
            //            x[i], y[j+dj], x, y, dem, dx, dy,
            //            icells, jcells, mpi_offset_x, mpi_offset_y);

            //    for (int dk = -1; dk <= 1; ++dk)
            //        if (z[k + dk] > zdem)
            //            return true;
            //}

            //for (int di = -1; di <= 1; ++di)
            //{
            //    // Interpolate DEM to account for half-level locations x,y
            //    const TF zdem = interp2_dem(
            //            x[i+di], y[j], x, y, dem, dx, dy,
            //            icells, jcells, mpi_offset_x, mpi_offset_y);

            //    for (int dk = -1; dk <= 1; ++dk)
            //        if (z[k + dk] > zdem)
            //            return true;
            //}

            // NEW METHOD
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di)
                {
                    // Interpolate DEM to account for half-level locations x,y
                    const TF zdem = interp2_dem(
                            x[i+di], y[j+dj], x, y, dem, dx, dy,
                            icells, jcells, mpi_offset_x, mpi_offset_y);

                    for (int dk = -1; dk <= 1; ++dk)
                        if (z[k + dk] > zdem)
                            return true;
                }
        }

        return false;
    }

    template<typename TF>
    void find_nearest_location_wall(
            TF& xb, TF& yb, TF& zb,
            const std::vector<TF>& x, const std::vector<TF>& y, const std::vector<TF>& dem,
            const TF x0, const TF y0, const TF z0,
            const TF dx, const TF dy,
            const int icells, const int jcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        TF d_min = 1e12;
        TF x_min, y_min, z_min;
        const int n = 40;

        for (int ii = -n/2; ii < n/2+1; ++ii)
            for (int jj = -n/2; jj < n/2+1; ++jj)
            {
                const TF xc = x0 + 2 * ii / (double) n * dx;
                const TF yc = y0 + 2 * jj / (double) n * dy;
                const TF zc = interp2_dem(xc, yc, x, y, dem, dx, dy, icells, jcells, mpi_offset_x, mpi_offset_y);
                const TF d  = absolute_distance(x0, y0, z0, xc, yc, zc);

                if (d < d_min)
                {
                    d_min = d;
                    x_min = xc;
                    y_min = yc;
                    z_min = zc;
                }
            }

        xb   = x_min;
        yb   = y_min;
        zb   = z_min;
    }

    template<typename TF>
    void find_interpolation_points(
            std::vector<int>& ip_i, std::vector<int>& ip_j, std::vector<int>& ip_k,
            std::vector<TF>& ip_d, std::vector<TF>& c_idw, 
            const int index, const int n_idw,
            const std::vector<TF>& x, const std::vector<TF>& y, const std::vector<TF>& z, const std::vector<TF>& dem,
            const TF d_lim, const TF dx, const TF dy,
            const int i, const int j, const int k,
            const int kstart, const int icells, const int jcells, const int ijcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        // Vectors including all neighbours outside IB
        std::vector<Neighbour<TF>> neighbours;

        // Limit vertical stencil near surface
        const int dk0 = std::max(-2, kstart-k);

        // Find neighbouring grid points outside IB
        for (int dk=dk0; dk<6; ++dk)
            for (int dj=-1; dj<2; ++dj)
                for (int di=-1; di<2; ++di)
                {
                    const TF zd = interp2_dem(x[i+di], y[j+dj], x, y, dem, dx, dy, icells, jcells, mpi_offset_x, mpi_offset_y);

                    // Check if grid point is outside IB
                    if (z[k+dk] > zd)
                    {
                        // Calculate distance to IB
                        //TF xb, yb, zb;
                        //find_nearest_location_wall(
                        //        xb, yb, zb, x, y, dem, x[i+di], y[j+dj], z[k+dk],
                        //        dx, dy, icells, jcells, mpi_offset_x, mpi_offset_y);
                        //const TF dist = absolute_distance(xb, yb, zb, x[i+di], y[j+dj], z[k+dk]);

                        //// Exclude if grid point is too close to the IB
                        //if (dist > d_lim)
                        //{
                            const TF distance = absolute_distance(x[i], y[j], z[k], x[i+di], y[j+dj], z[k+dk]);
                            Neighbour<TF> tmp_neighbour = {i+di, j+dj, k+dk, distance};
                            neighbours.push_back(tmp_neighbour);
                        //}
                    }
                }

        // Sort them on distance
        std::sort(neighbours.begin(), neighbours.end(), compare_value<TF>);

        if (neighbours.size() < n_idw)
        {
           std::cout << "ERROR: only found " << neighbours.size() << " interpolation points @ ";
           std::cout << "i=" << i << ", j=" << j << ", k=" << k << std::endl; 
           throw 1;
        }

        // Save `n_idw` nearest neighbours
        for (int ii=0; ii<n_idw; ++ii)
        {
            const int in = ii + index*n_idw;
            ip_i[in] = neighbours[ii].i;
            ip_j[in] = neighbours[ii].j;
            ip_k[in] = neighbours[ii].k;
            ip_d[in] = neighbours[ii].distance;
        }
    }

    template<typename TF>
    void precalculate_idw(std::vector<TF>& c_idw, std::vector<TF>& c_idw_sum,
                          const std::vector<int>& ip_i, const std::vector<int>& ip_j, const std::vector<int>& ip_k,
                          const std::vector<TF>& x, const std::vector<TF>& y, const std::vector<TF>& z,
                          const TF xi, const TF yi, const TF zi,
                          const TF xb, const TF yb, const TF zb,
                          Boundary_type bc, const int index, const int n_idw)
    {
        // Dirichlet BCs use one interpolation point less, and include
        // the boundary value as an interpolation point
        const int n = (bc == Boundary_type::Dirichlet_type) ? n_idw-1 : n_idw;

        TF dist_max = TF(0);

        // Temp vector for calculations
        std::vector<TF> tmp(n_idw);

        // Calculate distances interpolation points -> image point
        for (int i=0; i<n; ++i)
        {
            const int ii  = i + index * n_idw;
            const int ipi = ip_i[ii];
            const int ipj = ip_j[ii];
            const int ipk = ip_k[ii];

            tmp[i] = absolute_distance(xi, yi, zi, x[ipi], y[ipj], z[ipk]);
            dist_max = std::max(dist_max, tmp[i]);
        }

        // For Dirichlet, add distance image point to IB
        if (bc == Boundary_type::Dirichlet_type)
        {
            tmp[n_idw-1] = std::max(absolute_distance(xi, yi, zi, xb, yb, zb), TF(1e-9));
            dist_max = std::max(dist_max, tmp[n_idw-1]);
        }

        // Calculate interpolation coefficients
        for (int i=0; i<n_idw; ++i)
        {
            const int ii = i + index * n_idw;

            c_idw[ii] = std::pow((dist_max - tmp[i]) / (dist_max * tmp[i]), 0.5) + TF(1e-9);
            c_idw_sum[index] += c_idw[ii];
        }
    }


    namespace most = Monin_obukhov;
    namespace bsk = Boundary_surface_kernels;

    /* Is this cell inside the terrain? Same test is_ghost_cell uses. */
    template<typename TF>
    bool is_solid(
            const std::vector<TF>& dem,
            const std::vector<TF>& x, const std::vector<TF>& y,
            const std::vector<TF>& z, const TF dx, const TF dy,
            const int i, const int j, const int k,
            const int icells, const int jcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        const TF zdem = interp2_dem(x[i], y[j], x, y, dem, dx, dy,
                                    icells, jcells, mpi_offset_x, mpi_offset_y);
        return z[k] <= zdem;
    }

    template<typename TF>
    void find_wall_cells(
            Wall_cells<TF>& wall, const std::vector<TF>& dem,
            const std::vector<TF>& x, const std::vector<TF>& y,
            const std::vector<TF>& z, const std::vector<TF>& dz,
            const TF dx, const TF dy, const TF z0m, const TF z0h,
            const int istart, const int jstart, const int kstart,
            const int iend,   const int jend,   const int kend,
            const int icells, const int jcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        auto solid = [&](const int i, const int j, const int k)
        {
            return is_solid(dem, x, y, z, dx, dy, i, j, k,
                            icells, jcells, mpi_offset_x, mpi_offset_y);
        };

        auto add = [&](const int i, const int j, const int k,
                       const int axis, const int sgn, const TF da)
        {
            wall.i.push_back(i);
            wall.j.push_back(j);
            wall.k.push_back(k);
            wall.axis.push_back(axis);
            wall.sign.push_back(sgn);
            wall.dn.push_back(TF(0.5) * da);
            wall.da.push_back(da);
            wall.z0m.push_back(z0m);
            wall.z0h.push_back(z0h);
        };

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                for (int i=istart; i<iend; ++i)
                {
                    if (solid(i, j, k))
                        continue;               // this cell is rock

                    if (k > kstart && solid(i, j, k-1))
                        add(i, j, k, 2, -1, dz[k]);

                    if (solid(i-1, j, k)) add(i, j, k, 0, -1, dx);
                    if (solid(i+1, j, k)) add(i, j, k, 0, +1, dx);
                    if (solid(i, j-1, k)) add(i, j, k, 1, -1, dy);
                    if (solid(i, j+1, k)) add(i, j, k, 1, +1, dy);
                }

        wall.n = wall.i.size();
        wall.obuk .resize(wall.n, TF(-1));   // slightly unstable first guess
        wall.ustar.resize(wall.n, TF(0));
        wall.utan .resize(wall.n, TF(0));
    }

    template<typename TF>
    void wall_similarity_kernel(
            std::vector<TF>& ustar, std::vector<TF>& obuk,
            std::vector<TF>& utan,
            const TF* const restrict u, const TF* const restrict v,
            const TF* const restrict b, const TF* const restrict b_wall,
            const std::vector<int>& wi, const std::vector<int>& wj,
            const std::vector<int>& wk, const std::vector<int>& waxis,
            const std::vector<TF>& wdn,
            const std::vector<TF>& wz0m, const std::vector<TF>& wz0h,
            const bool sw_stability, const bool sw_stab_vertical,
            int& n_zl_clamped,
            const int n, const int icells, const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        for (int m=0; m<n; ++m)
        {
            const int ijk = wi[m] + wj[m]*jj + wk[m]*kk;

            // Horizontal speed at the CELL CENTRE, from its two faces.
            const TF uc = TF(0.5) * (u[ijk] + u[ijk+ii]);
            const TF vc = TF(0.5) * (v[ijk] + v[ijk+jj]);
            const TF du = std::max(std::sqrt(uc*uc + vc*vc),
                                   TF(Constants::dsmall));

            const TF zsl = wdn[m];

            TF L;
            if (sw_stability && (waxis[m] == 2 || sw_stab_vertical))
            {
                const TF db = b[ijk] - b_wall[m];
                L = bsk::calc_obuk_noslip_dirichlet_iterative(
                        obuk[m], du, db, zsl, wz0m[m], wz0h[m]);
            }
            else
                L = TF(Constants::dbig);      // neutral: psi -> 0

            if (L < TF(0) && zsl/L < Constants::zL_min<TF>)
            {
                L = zsl / Constants::zL_min<TF>;
                ++n_zl_clamped;
            }

            obuk[m]  = L;
            ustar[m] = du * most::fm(zsl, wz0m[m], L);
            utan[m]  = du;
        }
    }



    template<typename TF>
    void find_wall_cells_stag(
            Wall_cells<TF>& wall, const std::vector<TF>& dem,
            const std::vector<TF>& xg, const std::vector<TF>& yg,
            const std::vector<TF>& zg,
            const std::vector<TF>& x, const std::vector<TF>& y,
            const std::vector<TF>& dz,
            const TF dx, const TF dy, const TF z0m, const TF z0h,
            const int istart, const int jstart, const int kstart,
            const int iend,   const int jend,   const int kend,
            const int icells, const int jcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        auto solid = [&](const int i, const int j, const int k)
        {
            const TF zdem = interp2_dem(xg[i], yg[j], x, y, dem, dx, dy,
                                        icells, jcells,
                                        mpi_offset_x, mpi_offset_y);
            return zg[k] <= zdem;
        };

        auto add = [&](const int i, const int j, const int k,
                       const int axis, const int sgn, const TF da)
        {
            wall.i.push_back(i);
            wall.j.push_back(j);
            wall.k.push_back(k);
            wall.axis.push_back(axis);
            wall.sign.push_back(sgn);
            wall.dn.push_back(TF(0.5) * da);
            wall.da.push_back(da);
            wall.z0m.push_back(z0m);
            wall.z0h.push_back(z0h);
        };

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                for (int i=istart; i<iend; ++i)
                {
                    if (solid(i, j, k))
                        continue;

                    if (k > kstart && solid(i, j, k-1))
                        add(i, j, k, 2, -1, dz[k]);

                    if (solid(i-1, j, k)) add(i, j, k, 0, -1, dx);
                    if (solid(i+1, j, k)) add(i, j, k, 0, +1, dx);
                    if (solid(i, j-1, k)) add(i, j, k, 1, -1, dy);
                    if (solid(i, j+1, k)) add(i, j, k, 1, +1, dy);
                }

        wall.n = wall.i.size();
        wall.obuk .resize(wall.n, TF(-1));
        wall.ustar.resize(wall.n, TF(0));
        wall.utan .resize(wall.n, TF(0));
    }

    template<typename TF>
    void wall_momentum_flux_kernel(
            TF* const restrict at,
            const TF* const restrict fld,        // the component being forced
            const TF* const restrict oth,        // the other horizontal one
            const TF* const restrict w,
            const TF* const restrict evisc,
            const std::vector<int>& wi, const std::vector<int>& wj,
            const std::vector<int>& wk, const std::vector<int>& waxis,
            const std::vector<int>& wsign, const std::vector<TF>& wda,
            const std::vector<int>& wfloor,      // -> index into ustar
            const std::vector<TF>& ustar,
            const TF* const restrict dzi, const TF* const restrict dzhi,
            const TF* const restrict rhoref, const TF* const restrict rhorefh,
            const TF dxi, const TF dyi, const TF visc,
            const int comp, const bool sw_vert,
            TF& tau_max, TF& undo_max, int& n_bad,
            const int n, const int icells, const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        // Along-component and cross-component strides and metrics. For u the
        // component axis is x; for v it is y.
        const int  cc  = (comp == 0) ? ii  : jj;    // stride along the component
        const int  oo  = (comp == 0) ? jj  : ii;    // stride across it
        const TF   ci  = (comp == 0) ? dxi : dyi;   // 1/d along the component
        const TF   oi  = (comp == 0) ? dyi : dxi;   // 1/d across it

        for (int m=0; m<n; ++m)
        {
            const int i = wi[m];
            const int j = wj[m];
            const int k = wk[m];
            const int ijk = i + j*jj + k*kk;

            // --- 1. undo the operator's stress across this face -------------
            TF undo;
            if (waxis[m] == comp)
            {
                undo = TF(0);
            }
            else if (waxis[m] == 2)
            {
                // The floor/ceiling face: eviscb * (dc/dz + dw/dc).
                if (wsign[m] < 0)
                {
                    const TF Kf = TF(0.25)*(evisc[ijk-cc-kk] + evisc[ijk-kk]
                                          + evisc[ijk-cc   ] + evisc[ijk   ]) + visc;
                    undo = -rhorefh[k] * Kf*((fld[ijk] - fld[ijk-kk])*dzhi[k]
                                           + (w[ijk] - w[ijk-cc])*ci)
                           / rhoref[k] * dzi[k];
                }
                else
                {
                    const TF Kf = TF(0.25)*(evisc[ijk-cc   ] + evisc[ijk   ]
                                          + evisc[ijk-cc+kk] + evisc[ijk+kk]) + visc;
                    undo = +rhorefh[k+1] * Kf*((fld[ijk+kk] - fld[ijk])*dzhi[k+1]
                                             + (w[ijk+kk] - w[ijk-cc+kk])*ci)
                           / rhoref[k] * dzi[k];
                }
            }
            else
            {
                // A face normal to the OTHER horizontal axis:
                // eviscn/s * (dc/do + do/dc).
                if (wsign[m] < 0)
                {
                    const TF Kf = TF(0.25)*(evisc[ijk-cc-oo] + evisc[ijk-oo]
                                          + evisc[ijk-cc   ] + evisc[ijk   ]) + visc;
                    undo = -Kf*((fld[ijk] - fld[ijk-oo])*oi
                              + (oth[ijk] - oth[ijk-cc])*ci) * oi;
                }
                else
                {
                    const TF Kf = TF(0.25)*(evisc[ijk-cc   ] + evisc[ijk   ]
                                          + evisc[ijk-cc+oo] + evisc[ijk+oo]) + visc;
                    undo = +Kf*((fld[ijk+oo] - fld[ijk])*oi
                              + (oth[ijk+oo] - oth[ijk-cc+oo])*ci) * oi;
                }
            }

            if (waxis[m] == comp || (waxis[m] != 2 && !sw_vert))
            {
                at[ijk] -= undo;
                continue;
            }

            const int mf = wfloor[m];
            if (mf < 0)
            {
                // No floor face in this column: nothing to take u* from. Leave
                // the operator's stress in place rather than invent one.
                continue;
            }

            // The horizontal wind AT THIS POINT. fld is already there; the
            // other component needs the four-point staggered average.
            const TF fc = fld[ijk];
            const TF oc = TF(0.25)*(oth[ijk] + oth[ijk-cc]
                                  + oth[ijk+oo] + oth[ijk-cc+oo]);
            const TF uh = std::sqrt(fc*fc + oc*oc);

            TF tau = TF(0);
            if (uh > TF(Constants::dsmall))
                tau = -ustar[mf] * ustar[mf] * (fc / uh);   // opposes the wind

            if (!std::isfinite(tau) || !std::isfinite(undo))
            {
                ++n_bad;
                continue;
            }

            tau_max  = std::max(tau_max,  std::abs(tau));
            undo_max = std::max(undo_max, std::abs(undo));

            at[ijk] -= undo;
            at[ijk] += tau / wda[m];
        }
    }
    template<typename TF>
    void wall_scalar_flux_kernel(
            TF* const restrict at,
            std::vector<TF>& flux_out,
            const TF* const restrict a,
            const TF* const restrict evisc,
            const std::vector<TF>& phi_wall,
            const std::vector<int>& wi, const std::vector<int>& wj,
            const std::vector<int>& wk, const std::vector<int>& waxis,
            const std::vector<int>& wsign,
            const std::vector<TF>& wdn, const std::vector<TF>& wda,
            const std::vector<TF>& wz0h,
            const std::vector<TF>& ustar, const std::vector<TF>& obuk,
            const TF* const restrict dzi, const TF* const restrict dzhi,
            const TF* const restrict rhoref, const TF* const restrict rhorefh,
            const TF dxidxi, const TF dyidyi,
            const TF tPr_i, const TF visc,
            TF& ch_min, TF& ch_max, TF& src_max, TF& undo_max, int& n_bad,
            const bool sw_vert,
            const int n, const int icells, const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        for (int m=0; m<n; ++m)
        {
            const int i = wi[m];
            const int j = wj[m];
            const int k = wk[m];
            const int ijk = i + j*jj + k*kk;

            // --- 1. undo the operator's flux across this face ---------------
            TF undo;
            if (waxis[m] == 0)
            {
                if (wsign[m] < 0)
                {
                    const TF Kf = TF(0.5)*(evisc[ijk-ii] + evisc[ijk])*tPr_i + visc;
                    undo = -Kf*(a[ijk] - a[ijk-ii]) * dxidxi;
                }
                else
                {
                    const TF Kf = TF(0.5)*(evisc[ijk] + evisc[ijk+ii])*tPr_i + visc;
                    undo = +Kf*(a[ijk+ii] - a[ijk]) * dxidxi;
                }
            }
            else if (waxis[m] == 1)
            {
                if (wsign[m] < 0)
                {
                    const TF Kf = TF(0.5)*(evisc[ijk-jj] + evisc[ijk])*tPr_i + visc;
                    undo = -Kf*(a[ijk] - a[ijk-jj]) * dyidyi;
                }
                else
                {
                    const TF Kf = TF(0.5)*(evisc[ijk] + evisc[ijk+jj])*tPr_i + visc;
                    undo = +Kf*(a[ijk+jj] - a[ijk]) * dyidyi;
                }
            }
            else
            {
                if (wsign[m] < 0)
                {
                    const TF Kf = TF(0.5)*(evisc[ijk-kk] + evisc[ijk])*tPr_i + visc;
                    undo = -rhorefh[k] * Kf*(a[ijk] - a[ijk-kk])*dzhi[k]
                           / rhoref[k] * dzi[k];
                }
                else
                {
                    const TF Kf = TF(0.5)*(evisc[ijk] + evisc[ijk+kk])*tPr_i + visc;
                    undo = +rhorefh[k+1] * Kf*(a[ijk+kk] - a[ijk])*dzhi[k+1]
                           / rhoref[k] * dzi[k];
                }
            }
            if (waxis[m] != 2 && !sw_vert)
            {
                flux_out[m] = TF(0);
                at[ijk] -= undo;
                continue;
            }

            const TF fh = most::fh(wdn[m], wz0h[m], obuk[m]);
            const TF ch = ustar[m] * fh;
            const TF F  = ch * (phi_wall[m] - a[ijk]);

            if (!std::isfinite(F) || !std::isfinite(undo))
            {
                ++n_bad;
                flux_out[m] = TF(0);
                continue;
            }

            ch_min   = std::min(ch_min, ch);
            ch_max   = std::max(ch_max, ch);
            src_max  = std::max(src_max,  std::abs(F / wda[m]));
            undo_max = std::max(undo_max, std::abs(undo));

            at[ijk] -= undo;
            flux_out[m] = F;
            at[ijk] += F / wda[m];
        }
    }

    template<typename TF>
    void wall_impermeable_kernel(
            TF* const restrict u, TF* const restrict v, TF* const restrict w,
            const std::vector<int>& wi, const std::vector<int>& wj,
            const std::vector<int>& wk, const std::vector<int>& waxis,
            const std::vector<int>& wsign,
            const int n, const int icells, const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        for (int m=0; m<n; ++m)
        {
            const int ijk = wi[m] + wj[m]*jj + wk[m]*kk;

            if (waxis[m] == 0)
                u[wsign[m] < 0 ? ijk : ijk+ii] = TF(0);
            else if (waxis[m] == 1)
                v[wsign[m] < 0 ? ijk : ijk+jj] = TF(0);
            else
                w[wsign[m] < 0 ? ijk : ijk+kk] = TF(0);
        }
    }
    template<typename TF>
    void wall_coeff_kernel(
            std::vector<TF>& a_wall,
            const std::vector<int>& widx, const std::vector<TF>& nsq,
            const std::vector<TF>& di,
            const std::vector<int>& wi, const std::vector<int>& wj,
            const std::vector<int>& wk,
            const std::vector<TF>& ustar, const std::vector<TF>& utan,
            const std::vector<TF>& obuk, const std::vector<TF>& wdn,
            const std::vector<TF>& wz0h,
            const TF* const restrict evisc,
            const TF visc, const TF tPr_i, const bool is_scalar,
            const int nghost, const int icells, const int ijcells)
    {
        for (int n=0; n<nghost; ++n)
        {
            const int m = widx[n];
            if (m < 0)
            {
                a_wall[n] = TF(2);            // upstream Dirichlet mirror
                continue;
            }

            const int ijk = wi[m] + wj[m]*icells + wk[m]*ijcells;

            TF a;
            if (is_scalar)
            {
                const TF K = evisc[ijk] * tPr_i + visc;
                a = di[n] * ustar[m]
                    * most::fh(wdn[m], wz0h[m], obuk[m]) / K;
            }
            else
            {
                const TF K = evisc[ijk] + visc;
                const TF at = di[n] * ustar[m] * ustar[m]
                            / (K * std::max(utan[m], TF(Constants::dsmall)));
                a = TF(2)*nsq[n] + (TF(1) - nsq[n]) * std::min(at, TF(2));
            }

            a_wall[n] = std::min(std::max(a, TF(0)), TF(2));
        }
    }
    template<typename TF>
    void calc_ghost_cells(
            Ghost_cells<TF>& ghost, const std::vector<TF>& dem, 
            const std::vector<TF>& x, const std::vector<TF>& y, const std::vector<TF>& z,
            Boundary_type bc, const TF dx, const TF dy, const std::vector<TF>& dz, 
            const int n_idw,
            const int istart, const int jstart, const int kstart,
            const int iend,   const int jend,   const int kend,
            const int icells, const int jcells, const int ijcells,
            const int mpi_offset_x, const int mpi_offset_y)
    {
        // 1. Find the IB ghost cells
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                for (int i=istart; i<iend; ++i)
                    if (is_ghost_cell(dem, x, y, z, dx, dy, i, j, k,
                                      icells, jcells, mpi_offset_x, mpi_offset_y))
                    {
                        ghost.i.push_back(i);
                        ghost.j.push_back(j);
                        ghost.k.push_back(k);
                    }

        const int nghost = ghost.i.size();
        ghost.nghost = nghost;

        // 2. For each ghost cell, find the nearest location on the wall,
        // the image point (ghost cell mirrored across the IB),
        // and distance between image point and ghost cell.
        ghost.xb.resize(nghost);
        ghost.yb.resize(nghost);
        ghost.zb.resize(nghost);

        ghost.xi.resize(nghost);
        ghost.yi.resize(nghost);
        ghost.zi.resize(nghost);

        ghost.di.resize(nghost);

        for (int n=0; n<nghost; ++n)
        {
            // Indices ghost cell in 3D field
            const int i = ghost.i[n];
            const int j = ghost.j[n];
            const int k = ghost.k[n];

            find_nearest_location_wall(
                    ghost.xb[n], ghost.yb[n], ghost.zb[n],
                    x, y, dem, x[i], y[j], z[k],
                    dx, dy, icells, jcells, mpi_offset_x, mpi_offset_y);

            // Image point
            ghost.xi[n] = 2*ghost.xb[n] - x[i];
            ghost.yi[n] = 2*ghost.yb[n] - y[j];
            ghost.zi[n] = 2*ghost.zb[n] - z[k];

            // Distance image point -> ghost cell
            ghost.di[n] = absolute_distance(ghost.xi[n], ghost.yi[n], ghost.zi[n], x[i], y[j], z[k]);
        }

        // 3. Find N interpolation points outside of IB
        ghost.ip_i .resize(nghost*n_idw);
        ghost.ip_j .resize(nghost*n_idw);
        ghost.ip_k .resize(nghost*n_idw);
        ghost.ip_d .resize(nghost*n_idw);
        ghost.c_idw.resize(nghost*n_idw);

        for (int n=0; n<nghost; ++n)
        {
            // Exclude interpolation points closer than `d_lim` to IB
            const TF dist_lim = 0.1 * std::min(std::min(dx, dy), dz[ghost.k[n]]);

            find_interpolation_points(
                    ghost.ip_i, ghost.ip_j, ghost.ip_k, ghost.ip_d, ghost.c_idw,
                    n, n_idw, x, y, z, dem, dist_lim, dx, dy,
                    ghost.i[n], ghost.j[n], ghost.k[n], kstart,
                    icells, jcells, ijcells,
                    mpi_offset_x, mpi_offset_y);
        }

        // 4. Calculate interpolation coefficients
        ghost.c_idw.resize(nghost*n_idw);
        ghost.c_idw_sum.resize(nghost);

        for (int n=0; n<nghost; ++n)
        {
            precalculate_idw(
                    ghost.c_idw, ghost.c_idw_sum,
                    ghost.ip_i, ghost.ip_j, ghost.ip_k, x, y, z,
                    ghost.xi[n], ghost.yi[n], ghost.zi[n],
                    ghost.xb[n], ghost.yb[n], ghost.zb[n],
                    bc, n, n_idw);
        }
    }

    void print_statistics(std::vector<int>& ghost_i, std::string name, Master& master)
    {
        int nghost = ghost_i.size();
        master.sum(&nghost, 1);

        if (master.get_mpiid() == 0)
        {
            std::string message = "Found: " + std::to_string(nghost) + " IB ghost cells at the " + name + " location";
            master.print_message(message);
        }
    }

    template<typename TF>
    void set_ghost_cells(
            TF* const restrict fld, const TF* const restrict boundary_value,
            const TF* const restrict c_idw, const TF* const restrict c_idw_sum,
            const TF* const restrict di,
            const int* const restrict gi, const int* const restrict gj, const int* const restrict gk,
            const int* const restrict ipi, const int* const restrict ipj, const int* const restrict ipk,
            Boundary_type bc, const TF visc, const int n_ghostcells, const int n_idw,
            const int icells, const int ijcells,
            const TF* const restrict a_wall = nullptr)
    {
        const int n_idw_loc = (bc == Boundary_type::Dirichlet_type) ? n_idw-1 : n_idw;

        for (int n=0; n<n_ghostcells; ++n)
        {
            const int ijkg = gi[n] + gj[n]*icells + gk[n]*ijcells;

            // Sum the IDW coefficient times the value at the neighbouring grid points
            TF vI = TF(0);
            for (int i=0; i<n_idw_loc; ++i)
            {
                const int ii = i + n*n_idw;
                const int ijki = ipi[ii] + ipj[ii]*icells + ipk[ii]*ijcells;
                vI += c_idw[ii] * fld[ijki];
            }

            // For Dirichlet BCs, add the boundary value
            if (bc == Boundary_type::Dirichlet_type)
            {
                const int ii = n_idw-1 + n*n_idw;
                vI += c_idw[ii] * boundary_value[n];
            }

            vI /= c_idw_sum[n];

            // Set the ghost cells, depending on the IB boundary conditions
            if (a_wall != nullptr)
            {
                fld[ijkg] = vI + a_wall[n] * (boundary_value[n] - vI);
            }
            else if (bc == Boundary_type::Dirichlet_type)
                fld[ijkg] = 2*boundary_value[n] - vI;
            else if (bc == Boundary_type::Neumann_type)
                fld[ijkg] = vI - boundary_value[n] * di[n]; // Image value minus gradient times distance
            else if (bc == Boundary_type::Flux_type)
            {
                const TF grad = -boundary_value[n] / visc;
                fld[ijkg] = vI - grad * di[n];              // Image value minus gradient times distance
            }
        }
    }

    template<typename TF>
    void calc_mask(
            TF* const restrict mask, TF* const restrict maskh,
            const TF* const restrict z_dem,
            const TF* const restrict z, const TF* const restrict zh,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const int ij  = i + j*jj;

                    const int is_not_ib   = z [k] > z_dem[ij];
                    const int is_not_ib_h = zh[k] > z_dem[ij];

                    mask[ijk]  = static_cast<TF>(is_not_ib);
                    maskh[ijk] = static_cast<TF>(is_not_ib_h);
                }
    }

    template<typename TF>
    void find_k_dem(
            unsigned int* const restrict k_dem,
            const TF* const restrict dem,
            const TF* const restrict z,
            const int istart, const int iend, 
            const int jstart, const int jend, 
            const int kstart, const int kend,
            const int jj)
    {
        for (int j=jstart; j<jend; ++j)
            for (int i=istart; i<iend; ++i)
            {
                const int ij = i + j*jj;
                int k = -1;
                for (k=kstart; k<kend; ++k)
                {
                    if (z[k] > dem[ij])
                        break;
                }
                k_dem[ij] = k;
            }
    }

    template<typename TF>
    void calc_fluxes(
            TF* const restrict flux,
            const unsigned int* const restrict k_dem,
            const TF* restrict s,
            const TF* const restrict evisc,
            const TF tPr_i,
            const TF dx, const TF dy, const TF* const restrict dz,
            const TF dxi, const TF dyi, const TF* const restrict dzhi,
            const TF svisc,
            const int istart, const int iend, 
            const int jstart, const int jend, 
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int ii = 1;

        for (int j=jstart; j<jend; ++j)
            for (int i=istart; i<iend; ++i)
            {
                // Weight all fluxes by the respective area of the face through which they go.
                // Fluxes are only exchanged with neighbors that are a ghost cell. This requires
                // a drawing...

                auto Kf = [&](const int a, const int b)
                {
                    return TF(0.5)*(evisc[a] + evisc[b])*tPr_i + svisc;
                };

                // Add the vertical flux.
                const int ij  = i + j*jj;
                {
                    const int ijk = i + j*jj + k_dem[ij]*kk;
                    flux[ij] = -Kf(ijk, ijk-kk)*(s[ijk]-s[ijk-kk])*dzhi[k_dem[ij]] * dx*dy;
                }

                // West flux.
                for (int k=k_dem[ij]; k<k_dem[ij-ii]; ++k)
                {
                    const int ijk = i + j*jj + k*kk;
                    flux[ij] += -Kf(ijk, ijk-ii)*(s[ijk]-s[ijk-ii])*dxi * dy*dz[k];
                }
                // East flux.
                for (int k=k_dem[ij]; k<k_dem[ij+ii]; ++k)
                {
                    const int ijk = i + j*jj + k*kk;
                    flux[ij] += Kf(ijk, ijk+ii)*(s[ijk+ii]-s[ijk])*dxi * dy*dz[k];
                }
                // South flux.
                for (int k=k_dem[ij]; k<k_dem[ij-jj]; ++k)
                {
                    const int ijk = i + j*jj + k*kk;
                    flux[ij] += -Kf(ijk, ijk-jj)*(s[ijk]-s[ijk-jj])*dyi * dx*dz[k];
                }
                // North flux.
                for (int k=k_dem[ij]; k<k_dem[ij+jj]; ++k)
                {
                    const int ijk = i + j*jj + k*kk;
                    flux[ij] += Kf(ijk, ijk+jj)*(s[ijk+jj]-s[ijk])*dyi * dx*dz[k];
                }

                // Normalize the fluxes back to the correct units.
                flux[ij] /= dx*dy;
            }
    }

}

template<typename TF>
Immersed_boundary<TF>::Immersed_boundary(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
    master(masterin), grid(gridin), fields(fieldsin),
    field3d_io(masterin, gridin), boundary_cyclic(masterin, gridin)
{
    // Read IB switch from namelist, and set internal `sw_ib` switch
    std::string sw_ib_str = inputin.get_item<std::string>("IB", "sw_immersed_boundary", "", "0");

    if (sw_ib_str == "0")
        sw_ib = IB_type::Disabled;
    else if (sw_ib_str == "dem")
        sw_ib = IB_type::DEM;
    else
    {
        std::string error = "\"" + sw_ib_str + "\" is an illegal value for \"sw_ib\"";
        throw std::runtime_error(error);
    }

    if (sw_ib != IB_type::Disabled)
    {
        // Set a minimum of 2 ghost cells in the horizontal
        const int ijgc = 2;
        grid.set_minimum_ghost_cells(ijgc, ijgc, 0);

        // Read additional settings
        n_idw_points = inputin.get_item<int>("IB", "n_idw_points", "");

        sw_dem_halo_replicate = inputin.get_item<bool>(
                "IB", "sw_dem_halo_replicate", "", false);

        // Monin-Obukhov wall model. See IB_WALL_MODEL.md.
        const std::string wm = inputin.get_item<std::string>(
                "IB", "sw_wall_model", "", "0");
        if (wm == "0" || wm == "false")
            sw_wall_model = IB_wall_type::Disabled;
        else if (wm == "neutral")
            sw_wall_model = IB_wall_type::Neutral;
        else if (wm == "most")
            sw_wall_model = IB_wall_type::Most;
        else
            throw std::runtime_error(
                    "\"" + wm + "\" is not a valid \"sw_wall_model\"; use "
                    "0, neutral or most");

        if (sw_wall_model != IB_wall_type::Disabled)
        {
            z0m_ib = inputin.get_item<TF>("IB", "z0m", "");
            z0h_ib = inputin.get_item<TF>("IB", "z0h", "");
        }
        else
        {
            z0m_ib = TF(0);
            z0h_ib = TF(0);
        }

        // The scalar diffusivity diff_smag2 applies is evisc/tPr + svisc, so
        // the wall model has to read the same tPr rather than assume one.
        tPr_wm = inputin.get_item<TF>("diff", "tPr", "", TF(1./3.));
        sw_scalar_flux = inputin.get_item<bool>(
                "IB", "sw_scalar_flux", "", sw_wall_model != IB_wall_type::Disabled);
        sw_momentum_flux = inputin.get_item<bool>(
                "IB", "sw_momentum_flux", "",
                sw_wall_model != IB_wall_type::Disabled);
        sw_momentum_flux_vertical = inputin.get_item<bool>(
                "IB", "sw_momentum_flux_vertical", "", true);
        diag_z_mom    = inputin.get_item<TF>("IB", "diag_z_mom", "", TF(10));
        diag_z_scalar = inputin.get_item<TF>("IB", "diag_z_scalar", "", TF(2));
        sw_strain_most = inputin.get_item<bool>(
                "IB", "sw_strain_most", "",
                sw_wall_model != IB_wall_type::Disabled);
        sw_strain_most_vertical = inputin.get_item<bool>(
                "IB", "sw_strain_most_vertical", "", false);
        strain_most_min = inputin.get_item<TF>(
                "IB", "strain_most_min", "", TF(0.05));
        strain_most_reported = false;
        strain_most_built = false;

        if (sw_strain_most)
        {
            const std::string swdiff_sm = inputin.get_item<std::string>(
                    "diff", "swdiff", "", "0");
            if (swdiff_sm != "smag2")
                throw std::runtime_error(
                        "[IB] sw_strain_most rescales smag2's own evisc field. "
                        "With [diff] swdiff != smag2 there is nothing for it "
                        "to rescale and it would do nothing silently.");
            const bool aniso = inputin.get_item<bool>(
                    "diff", "swanisotropic", "", false);
            if (aniso)
                throw std::runtime_error(
                        "[IB] sw_strain_most does not handle the anisotropic "
                        "evisc_h / evisc_v split. Set [diff] "
                        "swanisotropic=false or sw_strain_most=false.");
        }
        mom_flux_reported = false;
        wall_flux_reported = false;
        n_zl_clamped_last  = 0;

        sw_scalar_flux_vertical = inputin.get_item<bool>(
                "IB", "sw_scalar_flux_vertical", "", true);

        sw_wall_stability_vertical = inputin.get_item<bool>(
                "IB", "sw_wall_stability_vertical", "", false);

        sw_impermeable = inputin.get_item<bool>(
                "IB", "sw_impermeable", "", false);

        if (sw_scalar_flux)
        {
            const std::string swdiff = inputin.get_item<std::string>(
                    "diff", "swdiff", "", "0");
            const std::string sworder = inputin.get_item<std::string>(
                    "grid", "swspatialorder", "", "2");
            if (swdiff != "smag2")
                throw std::runtime_error(
                        "[IB] sw_scalar_flux reproduces diff_c's own flux "
                        "expression in order to subtract it, and that "
                        "expression is smag2's. With [diff] swdiff != smag2 it "
                        "would subtract the wrong thing silently.");
            if (sworder != "2")
                throw std::runtime_error(
                        "[IB] sw_scalar_flux assumes the 2nd-order diffusion "
                        "kernels ([grid] swspatialorder=2).");
        }

        // Set available masks
        available_masks.insert(available_masks.end(), {"ib"});
    }
}

template <typename TF>
Immersed_boundary<TF>::~Immersed_boundary()
{
}


#ifndef USECUDA
template<typename TF>
void Immersed_boundary<TF>::exec_wall_model(Thermo<TF>& thermo, Stats<TF>& stats)
{
    // Cache the base-state exner while a Thermo is in reach; T_2m_ib needs it
    // and exec_cross has no Thermo of its own.
    if (exner_ref.empty())
        exner_ref = thermo.get_basestate_vector("exner");

    if (sw_ib == IB_type::Disabled || sw_wall_model == IB_wall_type::Disabled)
        return;
    if (wall.n == 0)
        return;

    auto& gd = grid.get_grid_data();

    const bool sw_stability = (sw_wall_model == IB_wall_type::Most);

    std::vector<TF> b_wall(wall.n, TF(0));
    std::shared_ptr<Field3d<TF>> buoy;

    if (sw_stability)
    {
        buoy = fields.get_tmp();
        thermo.get_thermo_field(*buoy, "b", false, false);

        const std::vector<TF>& thvref = thermo.get_basestate_vector("thv");

        const bool has_qt  = (fields.sp.find("qt")  != fields.sp.end());
        const bool has_thl = (fields.sp.find("thl") != fields.sp.end());

        auto wall_value = [&](const std::string& name, const int m) -> TF
        {
            auto it2 = sbot_2d.find(name);
            if (it2 != sbot_2d.end())
                return it2->second[wall.i[m] + wall.j[m]*gd.icells];
            return sbc.count(name) ? sbc.at(name) : TF(0);
        };

        for (int m=0; m<wall.n; ++m)
        {
            const int k = wall.k[m];
            const TF thl_w = has_thl ? wall_value("thl", m) : TF(0);
            const TF qt_w  = has_qt  ? wall_value("qt",  m) : TF(0);
            const TF thv_w = thl_w * (TF(1) + TF(0.61) * qt_w);
            b_wall[m] = Constants::grav<TF> * (thv_w - thvref[k]) / thvref[k];
        }
    }

    // nullptr is safe: the kernel only dereferences b when sw_stability.
    const TF* b_ptr = sw_stability ? buoy->fld.data() : nullptr;

    int n_zl_clamped = 0;

    wall_similarity_kernel<TF>(
            wall.ustar, wall.obuk, wall.utan,
            fields.mp.at("u")->fld.data(),
            fields.mp.at("v")->fld.data(),
            b_ptr, b_wall.data(),
            wall.i, wall.j, wall.k, wall.axis,
            wall.dn, wall.z0m, wall.z0h,
            sw_stability, sw_wall_stability_vertical,
            n_zl_clamped,
            wall.n, gd.icells, gd.ijcells);
    n_zl_clamped_last = n_zl_clamped;

    if (sw_stability)
        fields.release_tmp(buoy);

    std::shared_ptr<Field3d<TF>> evisc_tmp;
    const TF* evisc_ptr;
    if (fields.sd.count("evisc"))
        evisc_ptr = fields.sd.at("evisc")->fld.data();
    else
    {
        evisc_tmp = fields.get_tmp();
        std::fill(evisc_tmp->fld.begin(), evisc_tmp->fld.end(), TF(0));
        evisc_ptr = evisc_tmp->fld.data();
    }

    auto fill = [&](const std::string& name, const TF visc, const bool scalar)
    {
        Ghost_cells<TF>& gh = ghost.at(name);
        wall_coeff_kernel<TF>(
                gh.a_wall, gh.wall_idx, gh.nsq, gh.di,
                wall.i, wall.j, wall.k,
                wall.ustar, wall.utan, wall.obuk, wall.dn, wall.z0h,
                evisc_ptr, visc, TF(1) / tPr_wm, scalar,
                gh.nghost, gd.icells, gd.ijcells);
    };

    fill("u", fields.visc, false);
    fill("v", fields.visc, false);
    fill("w", fields.visc, false);
    if (fields.sp.size() > 0)
    {
        if (sw_scalar_flux)
        {
            std::fill(ghost.at("s").a_wall.begin(),
                      ghost.at("s").a_wall.end(), TF(0));
        }
        else
            fill("s", fields.sp.begin()->second->visc, true);
    }

    if (evisc_tmp)
        fields.release_tmp(evisc_tmp);
}
#endif

#ifndef USECUDA
template<typename TF>
void Immersed_boundary<TF>::sbot_2d_to_ghosts()
{
    if (fields.sp.size() == 0)
        return;

    auto& gd = grid.get_grid_data();
    const auto& mpi = master.get_MPI_data();
    const int mpi_offset_x = -mpi.mpicoordx * gd.imax + gd.igc;
    const int mpi_offset_y = -mpi.mpicoordy * gd.jmax + gd.jgc;

    Ghost_cells<TF>& gh = ghost.at("s");

    for (auto& it : sbot_2d)
    {
        auto its = gh.sbot.find(it.first);
        if (its == gh.sbot.end())
            continue;

        for (int i=0; i<gh.nghost; ++i)
            its->second[i] = interp2_dem(
                    gh.xb[i], gh.yb[i], gd.x, gd.y, it.second,
                    gd.dx, gd.dy, gd.icells, gd.jcells,
                    mpi_offset_x, mpi_offset_y);
    }
}

template<typename TF>
void Immersed_boundary<TF>::update_time_dependent(Timeloop<TF>& timeloop)
{
    if (sw_ib == IB_type::Disabled || !sw_timedep_sbot || sbot_2d.empty())
        return;

    auto& gd = grid.get_grid_data();
    const unsigned long itime = timeloop.get_itime();
    const unsigned long iiotimeprec = timeloop.get_iiotimeprec();

    auto tmp = fields.get_tmp();
    int nerror = 0;

    auto load_2d = [&](std::vector<TF>& fld, const std::string& name,
                       const unsigned long it_load)
    {
        char filename[256];
        const std::string base = name + "_sbot";
        std::snprintf(filename, 256, "%s.%07d", base.c_str(),
                      int(it_load / iiotimeprec));
        master.print_message("Loading \"%s\" ... ", filename);

        if (field3d_io.load_xy_slice(fld.data(), tmp->fld.data(), filename))
        {
            master.print_message("FAILED\n");
            nerror += 1;
        }
        else
            master.print_message("OK\n");

        boundary_cyclic.exec_2d(fld.data());
    };

    if (!sbot_timedep_init)
    {
        // Snap to the input grid the way boundary.cxx does, so a restart at a
        // time that is not a multiple of sbot_loadtime still brackets it.
        itime_sbot_prev = itime / iloadtime_sbot * iloadtime_sbot;
        itime_sbot_next = itime_sbot_prev + iloadtime_sbot;

        for (auto& it : sbot_2d)
        {
            sbot_2d_prev.emplace(it.first, std::vector<TF>(gd.ijcells, TF(0)));
            sbot_2d_next.emplace(it.first, std::vector<TF>(gd.ijcells, TF(0)));
            load_2d(sbot_2d_prev.at(it.first), it.first, itime_sbot_prev);
            load_2d(sbot_2d_next.at(it.first), it.first, itime_sbot_next);
        }

        sbot_timedep_init = true;

        master.print_message(
                "IB: 2-D surface condition follows time, "
                "<scalar>_sbot.<iotime> every %d s\n",
                int(iloadtime_sbot / iiotimeprec));
    }
    else if (itime > itime_sbot_next)
    {
        itime_sbot_prev = itime_sbot_next;
        itime_sbot_next = itime_sbot_prev + iloadtime_sbot;

        for (auto& it : sbot_2d)
        {
            sbot_2d_prev.at(it.first) = sbot_2d_next.at(it.first);
            load_2d(sbot_2d_next.at(it.first), it.first, itime_sbot_next);
        }
    }

    fields.release_tmp(tmp);
    master.sum(&nerror, 1);
    if (nerror)
        throw std::runtime_error(
                "[IB] sw_timedep_sbot: could not read a <scalar>_sbot field. "
                "There is no extrapolation - write fields covering the whole "
                "run, at every multiple of sbot_loadtime.");

    // Linear in time, as boundary.cxx does for the flat bottom.
    const TF f1 = TF(itime - itime_sbot_prev)
                / TF(itime_sbot_next - itime_sbot_prev);
    const TF f0 = TF(1) - f1;

    for (auto& it : sbot_2d)
    {
        std::vector<TF>& sb = it.second;
        const std::vector<TF>& p = sbot_2d_prev.at(it.first);
        const std::vector<TF>& q = sbot_2d_next.at(it.first);

        for (int n=0; n<gd.ijcells; ++n)
            sb[n] = f0 * p[n] + f1 * q[n];
    }

    sbot_2d_to_ghosts();
}
#endif

#ifndef USECUDA
template<typename TF>
void Immersed_boundary<TF>::exec_scalar_flux(Thermo<TF>& thermo, Stats<TF>& stats)
{
    if (sw_ib == IB_type::Disabled || !sw_scalar_flux)
        return;
    if (sw_wall_model == IB_wall_type::Disabled || wall.n == 0)
        return;
    if (fields.sp.size() == 0)
        return;

    auto& gd = grid.get_grid_data();

    // The SGS diffusivity the operator used. Without smag2 there is no evisc
    // and the molecular value is what diff_c applied.
    std::shared_ptr<Field3d<TF>> evisc_tmp;
    const TF* evisc_ptr;
    if (fields.sd.count("evisc"))
        evisc_ptr = fields.sd.at("evisc")->fld.data();
    else
    {
        evisc_tmp = fields.get_tmp();
        std::fill(evisc_tmp->fld.begin(), evisc_tmp->fld.end(), TF(0));
        evisc_ptr = evisc_tmp->fld.data();
    }

    const TF dxidxi = TF(1) / (gd.dx * gd.dx);
    const TF dyidyi = TF(1) / (gd.dy * gd.dy);

    TF ch_min = TF(1e30), ch_max = TF(-1e30);
    TF src_max = TF(0), undo_max = TF(0);
    int n_bad = 0;

    for (auto& it : fields.sp)
    {
        const std::string& name = it.first;

        std::vector<TF>& pw = wall_value_face.at(name);
        auto it2d = sbot_2d.find(name);
        for (int m=0; m<wall.n; ++m)
            pw[m] = (it2d != sbot_2d.end())
                  ? it2d->second[wall.i[m] + wall.j[m]*gd.icells]
                  : sbc.at(name);

        wall_scalar_flux_kernel<TF>(
                fields.st.at(name)->fld.data(),
                wall_flux.at(name),
                it.second->fld.data(),
                evisc_ptr,
                pw,
                wall.i, wall.j, wall.k, wall.axis, wall.sign,
                wall.dn, wall.da, wall.z0h,
                wall.ustar, wall.obuk,
                gd.dzi.data(), gd.dzhi.data(),
                fields.rhoref.data(), fields.rhorefh.data(),
                dxidxi, dyidyi,
                TF(1) / tPr_wm, it.second->visc,
                ch_min, ch_max, src_max, undo_max, n_bad,
                sw_scalar_flux_vertical,
                wall.n, gd.icells, gd.ijcells);
    }


    if (hfss_ij.size() == size_t(gd.ijcells))
    {
        const std::vector<TF>& exn = thermo.get_basestate_vector("exner");
        std::fill(hfss_ij.begin(), hfss_ij.end(), TF(0));
        std::fill(hfls_ij.begin(), hfls_ij.end(), TF(0));

        for (int m=0; m<wall.n; ++m)
        {
            const int ij = wall.i[m] + wall.j[m]*gd.icells;
            const int k  = wall.k[m];
            const TF w_area = (wall.axis[m] == 2) ? TF(1)
                            : (wall.axis[m] == 0) ? gd.dz[k]*gd.dxi
                                                  : gd.dz[k]*gd.dyi;
            const TF rho = fields.rhoref[k];
            if (wall_flux.count("thl"))
                hfss_ij[ij] += rho * Constants::cp<TF> * exn[k]
                             * wall_flux.at("thl")[m] * w_area;
            if (wall_flux.count("qt"))
                hfls_ij[ij] += rho * Constants::Lv<TF>
                             * wall_flux.at("qt")[m] * w_area;
        }
    }
    if (evisc_tmp)
        fields.release_tmp(evisc_tmp);

    master.sum(&n_bad, 1);

    if (!wall_flux_reported)
    {
        wall_flux_reported = true;
        master.print_message(
                "IB: surface exchange velocity u* f_h = %.4g .. %.4g m/s "
                "(MOST expects ~0.01; the ghost cell it replaces implied ~6)\n",
                double(ch_min), double(ch_max));
        master.print_message(
                "IB: max |source| %.3g K/s, max |diffusive term removed| "
                "%.3g K/s, %d face(s) with z/L clamped, %d non-finite\n",
                double(src_max), double(undo_max), n_zl_clamped_last, n_bad);
    }

    if (n_bad > 0)
        throw std::runtime_error(
                "[IB] sw_scalar_flux produced a non-finite surface flux. That "
                "is u* f_h (phi_wall - phi_cell) or the diffusive term being "
                "subtracted, so look at ustar_ib and obuk_ib first: L -> 0- "
                "over near-zero wind sends f_h to infinity. Faces affected is "
                "reported above.");
}

template<typename TF>
void Immersed_boundary<TF>::exec_impermeable()
{
    if (sw_ib == IB_type::Disabled || !sw_impermeable || wall.n == 0)
        return;

    auto& gd = grid.get_grid_data();

    wall_impermeable_kernel<TF>(
            fields.mp.at("u")->fld.data(),
            fields.mp.at("v")->fld.data(),
            fields.mp.at("w")->fld.data(),
            wall.i, wall.j, wall.k, wall.axis, wall.sign,
            wall.n, gd.icells, gd.ijcells);
}
#endif

#ifndef USECUDA
template<typename TF>
void Immersed_boundary<TF>::exec_momentum_flux(Stats<TF>& stats)
{
    if (sw_ib == IB_type::Disabled || !sw_momentum_flux)
        return;
    if (sw_wall_model == IB_wall_type::Disabled || wall.n == 0)
        return;

    auto& gd = grid.get_grid_data();

    std::shared_ptr<Field3d<TF>> evisc_tmp;
    const TF* evisc_ptr;
    if (fields.sd.count("evisc"))
        evisc_ptr = fields.sd.at("evisc")->fld.data();
    else
    {
        evisc_tmp = fields.get_tmp();
        std::fill(evisc_tmp->fld.begin(), evisc_tmp->fld.end(), TF(0));
        evisc_ptr = evisc_tmp->fld.data();
    }

    TF tau_max = TF(0), undo_max = TF(0);
    int n_bad = 0;

    const TF* const u = fields.mp.at("u")->fld.data();
    const TF* const v = fields.mp.at("v")->fld.data();
    const TF* const w = fields.mp.at("w")->fld.data();

    wall_momentum_flux_kernel<TF>(
            fields.mt.at("u")->fld.data(), u, v, w, evisc_ptr,
            wall_u.i, wall_u.j, wall_u.k, wall_u.axis, wall_u.sign, wall_u.da,
            wall_u_floor, wall.ustar,
            gd.dzi.data(), gd.dzhi.data(),
            fields.rhoref.data(), fields.rhorefh.data(),
            gd.dxi, gd.dyi, fields.visc,
            0, sw_momentum_flux_vertical,
            tau_max, undo_max, n_bad,
            wall_u.n, gd.icells, gd.ijcells);

    wall_momentum_flux_kernel<TF>(
            fields.mt.at("v")->fld.data(), v, u, w, evisc_ptr,
            wall_v.i, wall_v.j, wall_v.k, wall_v.axis, wall_v.sign, wall_v.da,
            wall_v_floor, wall.ustar,
            gd.dzi.data(), gd.dzhi.data(),
            fields.rhoref.data(), fields.rhorefh.data(),
            gd.dxi, gd.dyi, fields.visc,
            1, sw_momentum_flux_vertical,
            tau_max, undo_max, n_bad,
            wall_v.n, gd.icells, gd.ijcells);

    if (evisc_tmp)
        fields.release_tmp(evisc_tmp);

    master.sum(&n_bad, 1);

    if (!mom_flux_reported)
    {
        mom_flux_reported = true;
        master.print_message(
                "IB: momentum through MOST on %d u-faces and %d v-faces; "
                "max |u*^2| %.4g m2/s2, max |diffusive stress removed| "
                "%.4g m2/s2, %d non-finite\n",
                wall_u.n, wall_v.n, double(tau_max), double(undo_max), n_bad);
        if (undo_max > TF(20) * tau_max)
            master.print_message(
                    "IB: *** the stress being REMOVED is %.0fx the stress "
                    "being applied. That is the over-coupling this patch "
                    "exists to fix, so a large ratio here is expected at t=0 - "
                    "but if it stays large the subtraction is not matching "
                    "diff_u/diff_v and the run will not survive it.\n",
                    double(undo_max / std::max(tau_max, TF(1e-12))));
    }

    if (n_bad > 0)
        throw std::runtime_error(
                "[IB] sw_momentum_flux produced a non-finite stress. Look at "
                "ustar_ib first.");
}
#endif
#ifndef USECUDA
template <typename TF>
void Immersed_boundary<TF>::exec_momentum()
{
    if (sw_ib == IB_type::Disabled)
        return;

    auto& gd = grid.get_grid_data();

    set_ghost_cells(
            fields.mp.at("u")->fld.data(), ghost.at("u").mbot.data(),
            ghost.at("u").c_idw.data(), ghost.at("u").c_idw_sum.data(), ghost.at("u").di.data(),
            ghost.at("u").i.data(), ghost.at("u").j.data(), ghost.at("u").k.data(),
            ghost.at("u").ip_i.data(), ghost.at("u").ip_j.data(), ghost.at("u").ip_k.data(),
            Boundary_type::Dirichlet_type, fields.visc, ghost.at("u").i.size(), n_idw_points,
            gd.icells, gd.ijcells,
            (sw_wall_model != IB_wall_type::Disabled)
                ? ghost.at("u").a_wall.data() : nullptr);

    set_ghost_cells(
            fields.mp.at("v")->fld.data(), ghost.at("v").mbot.data(),
            ghost.at("v").c_idw.data(), ghost.at("v").c_idw_sum.data(), ghost.at("v").di.data(),
            ghost.at("v").i.data(), ghost.at("v").j.data(), ghost.at("v").k.data(),
            ghost.at("v").ip_i.data(), ghost.at("v").ip_j.data(), ghost.at("v").ip_k.data(),
            Boundary_type::Dirichlet_type, fields.visc, ghost.at("v").i.size(), n_idw_points,
            gd.icells, gd.ijcells,
            (sw_wall_model != IB_wall_type::Disabled)
                ? ghost.at("v").a_wall.data() : nullptr);

    set_ghost_cells(
            fields.mp.at("w")->fld.data(), ghost.at("w").mbot.data(),
            ghost.at("w").c_idw.data(), ghost.at("w").c_idw_sum.data(), ghost.at("w").di.data(),
            ghost.at("w").i.data(), ghost.at("w").j.data(), ghost.at("w").k.data(),
            ghost.at("w").ip_i.data(), ghost.at("w").ip_j.data(), ghost.at("w").ip_k.data(),
            Boundary_type::Dirichlet_type, fields.visc, ghost.at("w").i.size(), n_idw_points,
            gd.icells, gd.ijcells,
            (sw_wall_model != IB_wall_type::Disabled)
                ? ghost.at("w").a_wall.data() : nullptr);

    boundary_cyclic.exec(fields.mp.at("u")->fld.data());
    boundary_cyclic.exec(fields.mp.at("v")->fld.data());
    boundary_cyclic.exec(fields.mp.at("w")->fld.data());

    blank_solid_momentum();
}

template <typename TF>
void Immersed_boundary<TF>::exec_scalars()
{
    if (sw_ib == IB_type::Disabled)
        return;

    auto& gd = grid.get_grid_data();

    for (auto& it : fields.sp)
    {
        set_ghost_cells(
                it.second->fld.data(), ghost.at("s").sbot.at(it.first).data(),
                ghost.at("s").c_idw.data(), ghost.at("s").c_idw_sum.data(), ghost.at("s").di.data(),
                ghost.at("s").i.data(), ghost.at("s").j.data(), ghost.at("s").k.data(),
                ghost.at("s").ip_i.data(), ghost.at("s").ip_j.data(), ghost.at("s").ip_k.data(),
                sbcbot, it.second->visc, ghost.at("s").i.size(), n_idw_points,
                gd.icells, gd.ijcells,
                (sw_wall_model != IB_wall_type::Disabled)
                    ? ghost.at("s").a_wall.data() : nullptr);

        boundary_cyclic.exec(it.second->fld.data());
    }

    blank_solid_scalars();
}

template<typename TF>
void Immersed_boundary<TF>::exec_strain_most()
{
    if (sw_ib == IB_type::Disabled || !sw_strain_most)
        return;
    if (!fields.sd.count("evisc"))
        return;

    auto& gd = grid.get_grid_data();
    TF* const restrict evisc = fields.sd.at("evisc")->fld.data();

    if (!strain_most_built)
    {
        strain_most_built = true;

        std::map<int, int> pick;            // ijk -> face index
        for (int m=0; m<wall.n; ++m)
        {
            if (!sw_strain_most_vertical && wall.axis[m] != 2)
                continue;
            const int ijk = wall.i[m] + wall.j[m]*gd.icells + wall.k[m]*gd.ijcells;
            auto it = pick.find(ijk);
            if (it == pick.end())
                pick[ijk] = m;
            else if (wall.axis[m] == 2 && wall.axis[it->second] != 2)
                it->second = m;             // prefer the floor
        }

        strain_most_m.clear();
        strain_most_m.reserve(pick.size());
        for (const auto& kv : pick)
            strain_most_m.push_back(kv.second);
    }

    const TF kappa = Constants::kappa<TF>;

    TF r_lo = TF(1);
    TF r_hi = TF(0);
    int n_cells = 0;

    for (const int m : strain_most_m)
    {
        const TF us = wall.ustar[m];
        if (!(us > TF(0)))
            continue;                       // before the first exec_wall_model

        const TF L    = wall.obuk[m];
        const TF zeta = (std::abs(L) > TF(1e-12)) ? wall.dn[m] / L : TF(0);

        // r = u* phim / (kappa |U|) = phim * fm / kappa. |U| cancels.
        TF r = most::phim(zeta) * most::fm(wall.dn[m], wall.z0m[m], L) / kappa;
        if (!std::isfinite(r))
            continue;
        r = std::min(TF(1), std::max(strain_most_min, r));

        const int ijk = wall.i[m] + wall.j[m]*gd.icells + wall.k[m]*gd.ijcells;
        evisc[ijk] *= r;

        r_lo = std::min(r_lo, r);
        r_hi = std::max(r_hi, r);
        ++n_cells;
    }

    // The diffusion operator reads evisc from the neighbouring ranks too.
    boundary_cyclic.exec(evisc);

    if (!strain_most_reported && n_cells > 0)
    {
        strain_most_reported = true;
        master.sum(&n_cells, 1);
        master.min(&r_lo, 1);
        master.max(&r_hi, 1);
        master.print_message(
                "IB: sw_strain_most rescaling evisc on %d wall cell(s), "
                "factor %.4f .. %.4f (neutral limit is 1/ln(dn/z0m)); "
                "vertical faces %s\n",
                n_cells, double(r_lo), double(r_hi),
                sw_strain_most_vertical ? "included" : "skipped");
        master.print_message(
                "IB: expect DNUM to stop being set by the wall cells. If dt "
                "does not rise, the diffusion limit has moved into the "
                "interior and this patch has done all it can.\n");
    }
}
template<typename TF>
void Immersed_boundary<TF>::blank_solid_scalars()
{
    if (sw_ib == IB_type::Disabled || !sw_blank_solid)
        return;

    for (auto& it : fields.sp)
    {
        TF* const restrict fld = it.second->fld.data();

        auto it2d = sbot_2d.find(it.first);
        const bool has_2d = (it2d != sbot_2d.end());
        const TF s_const = sbc.count(it.first) ? sbc.at(it.first) : TF(0);

        if (has_2d)
        {
            const std::vector<TF>& s2d = it2d->second;
            for (std::size_t n=0; n<blank_s.size(); ++n)
                fld[blank_s[n]] = s2d[blank_s_ij[n]];
        }
        else
        {
            for (std::size_t n=0; n<blank_s.size(); ++n)
                fld[blank_s[n]] = s_const;
        }

        boundary_cyclic.exec(fld);
    }
}

template<typename TF>
void Immersed_boundary<TF>::blank_solid_momentum()
{
    if (sw_ib == IB_type::Disabled || !sw_blank_solid)
        return;

    TF* const restrict u = fields.mp.at("u")->fld.data();
    TF* const restrict v = fields.mp.at("v")->fld.data();
    TF* const restrict w = fields.mp.at("w")->fld.data();

    for (std::size_t n=0; n<blank_u.size(); ++n)
        u[blank_u[n]] = TF(0);
    for (std::size_t n=0; n<blank_v.size(); ++n)
        v[blank_v[n]] = TF(0);
    for (std::size_t n=0; n<blank_w.size(); ++n)
        w[blank_w[n]] = TF(0);

    boundary_cyclic.exec(u);
    boundary_cyclic.exec(v);
    boundary_cyclic.exec(w);
}
#endif

template <typename TF>
void Immersed_boundary<TF>::init(Input& inputin, Cross<TF>& cross)
{
    auto& gd = grid.get_grid_data();

    if (sw_ib == IB_type::Disabled)
        return;
    else if (sw_ib == IB_type::DEM)
    {
        dem  .resize(gd.ijcells);
        k_dem.resize(gd.ijcells);
    }

    // Process the boundary conditions for scalars
    if (fields.sp.size() > 0)
    {
        // All scalars have the same boundary type (for now)
        std::string swbot = inputin.get_item<std::string>("IB", "sbcbot", "");

        if (swbot == "flux")
            sbcbot = Boundary_type::Flux_type;
        else if (swbot == "dirichlet")
            sbcbot = Boundary_type::Dirichlet_type;
        else if (swbot == "neumann")
            sbcbot = Boundary_type::Neumann_type;
        else
        {
            std::string error = "IB sbcbot=" + swbot + " is not a valid choice (options: dirichlet, neumann, flux)";
            throw std::runtime_error(error);
        }

        // Process boundary values per scalar
        for (auto& it : fields.sp)
            sbc.emplace(it.first, inputin.get_item<TF>("IB", "sbot", it.first));

        // Read the scalars with spatial patterns
        sbot_spatial_list = inputin.get_list<std::string>("IB", "sbot_spatial", "", std::vector<std::string>());
        sw_timedep_sbot = inputin.get_item<bool>(
                "IB", "sw_timedep_sbot", "", false);
        sbot_timedep_init = false;
        iloadtime_sbot = 0;
        itime_sbot_prev = 0;
        itime_sbot_next = 0;
        if (sw_timedep_sbot)
        {
            if (sbot_spatial_list.empty())
                throw std::runtime_error(
                        "[IB] sw_timedep_sbot needs sbot_spatial: there is no "
                        "2-D surface field to advance in time");
            iloadtime_sbot = convert_to_itime(
                    inputin.get_item<int>("IB", "sbot_loadtime", ""));
        }
    }

    sw_blank_solid = inputin.get_item<bool>("IB", "sw_blank_solid", "", false);

    // [IB] sw_advec_wall - apply_ib_advec_wall.py
    sw_advec_wall = inputin.get_item<bool>("IB", "sw_advec_wall", "", false);
    advec_wall_reported = false;
    if (sw_advec_wall)
    {
        // The correction recomputes advec_2i5's face fluxes to replace them
        // exactly, so it is only valid for that scheme.
        const std::string swadvec = inputin.get_item<std::string>("advec", "swadvec", "", "");
        if (swadvec != "2i5")
            throw std::runtime_error(
                    "[IB] sw_advec_wall reproduces the face fluxes of "
                    "[advec] swadvec=2i5 and cannot be used with swadvec=" + swadvec);
        advec_wall_limited = inputin.get_list<std::string>(
                "advec", "fluxlimit_list", "", std::vector<std::string>());
    }

    tPr_ib = inputin.get_item<TF>("diff", "tPr", "", TF(1./3.));

    // Terrain-following xy cross-sections. The heights live here rather than
    // in the crosslist names, so that changing them does not mean editing
    // [cross] crosslist. See apply_ib_tf_cross.py.
    sw_tf_cross = inputin.get_item<bool>("IB", "sw_tf_cross", "", false);
    tf_cross_heights = inputin.get_list<TF>(
            "IB", "tf_cross_heights", "", std::vector<TF>());


    // ---- [IB] columnlist ---------------------------------------------------
    // Aliases are accepted so the .ini can use the short names people write on
    // a whiteboard; the canonical name is what lands in the netCDF, so two
    // spellings can never produce two differently-named copies of one field.
    if (sw_ib != IB_type::Disabled)
    {
        std::vector<std::string> raw =
                inputin.get_list<std::string>("IB", "columnlist", "",
                                              std::vector<std::string>());

        static const struct { const char* alias; const char* canonical; }
        alias_table[] = {
            {"wt_ib",   "thl_fluxbot_ib"}, {"wq_ib",   "qt_fluxbot_ib"},
            {"obl_ib",  "obuk_ib"},
            {"th2m",    "thl_2m_ib"},      {"T2m",     "T_2m_ib"},
            {"qt2m",    "qt_2m_ib"},       {"qt2mm",   "qt_2m_ib"},
            {"u10m",    "u_10m_ib"},       {"v10m",    "v_10m_ib"},
            {"ws10m",   "wspd_10m_ib"},    {"wd10m",   "wdir_10m_ib"},
            {"thls_ib", "thl_sbot_ib"},    {"qts_ib",  "qt_sbot_ib"},
        };

        for (std::string s : raw)
        {
            for (const auto& a : alias_table)
                if (s == a.alias)
                {
                    master.print_message("IB: columnlist \"%s\" -> \"%s\"\n",
                                         s.c_str(), a.canonical);
                    s = a.canonical;
                    break;
                }
            if (std::find(columnlist.begin(), columnlist.end(), s)
                    == columnlist.end())
                columnlist.push_back(s);
        }
    }
    // Check input list of cross variables (crosslist)
    std::vector<std::string>& crosslist_global = cross.get_crosslist();
    std::vector<std::string>::iterator it = crosslist_global.begin();
    while (it != crosslist_global.end())
    {

        // Terrain-following xy planes: <var>_tf (one plane per
        // [IB] tf_cross_heights) or <var>_tf<h>. See apply_ib_tf_cross.py.
        if (sw_tf_cross)
        {
            const size_t p_tf = it->rfind("_tf");
            const bool shaped = p_tf != std::string::npos
                    && (p_tf + 3 == it->size()
                        || it->find_first_not_of("0123456789", p_tf+3)
                           == std::string::npos);
            if (shaped)
            {
                const std::string var = it->substr(0, p_tf);
                const bool known_var = (var == "u" || var == "v" || var == "w"
                        || var == "wspd" || var == "wdir"
                        || fields.sp.find(var) != fields.sp.end());
                const bool have_h = (p_tf + 3 < it->size())
                        || !tf_cross_heights.empty();
                if (known_var && have_h)
                {
                    crosslist.push_back(*it);
                    it = crosslist_global.erase(it);
                    continue;
                }
            }
        }

        // Terrain-following surface diagnostics: taken verbatim, no <scalar>
        // prefix to strip. See apply_ib_surface_diag.py.
        {
            static const char* diag_names[] = {
                "u_ib1", "v_ib1", "thl_ib1", "qt_ib1",
                "u_10m_ib", "v_10m_ib", "wspd_10m_ib", "wdir_10m_ib",
                "thl_2m_ib", "T_2m_ib", "qt_2m_ib",
                "thl_sbot_ib", "qt_sbot_ib",
                "thl_sbot_ib_floor", "thl_sbot_ib_wall",
                "qt_sbot_ib_floor", "qt_sbot_ib_wall",
                "hfss_ib", "hfls_ib",
                "hfss_ib_floor", "hfss_ib_wall",
                "hfls_ib_floor", "hfls_ib_wall",
                "thl_fluxbot_ib_floor", "thl_fluxbot_ib_wall",
                "qt_fluxbot_ib_floor", "qt_fluxbot_ib_wall",
                "ustar_ib_floor", "ustar_ib_wall",
                "obuk_ib_floor", "obuk_ib_wall",
                "ch_ib_floor", "ch_ib_wall",
                "nface", "nface_floor", "nface_wall", "area_wall"};
            bool hit = false;
            for (const char* nm : diag_names)
                if (*it == nm) { hit = true; break; }
            if (hit)
            {
                crosslist.push_back(*it);
                crosslist_global.erase(it);
                continue;
            }
        }
        if (*it == "ustar_ib" || *it == "obuk_ib" || *it == "ch_ib")
        {
            crosslist_wall.push_back(*it);
            it = crosslist_global.erase(it);
            continue;
        }
        const std::string dsurf_ib_string = "dsurf_ib";
        if (has_ending(*it, dsurf_ib_string))
        {
            std::string scalar = *it;
            scalar.erase(it->length() - dsurf_ib_string.length());
            if (fields.sp.find(scalar) != fields.sp.end())
            {
                crosslist.push_back(*it);
                crosslist_global.erase(it);
            }
            else
                ++it;
            continue;
        }

        const std::string fluxbot_ib_string = "fluxbot_ib";
        if (has_ending(*it, fluxbot_ib_string))
        {
            // Strip the ending.
            std::string scalar = *it;
            scalar.erase(it->length() - fluxbot_ib_string.length());

            // Check if array is exists, else cycle.
            if (fields.sp.find(scalar) != fields.sp.end())
            {
                // Remove variable from global list, put in local list
                crosslist.push_back(*it);
                crosslist_global.erase(it); // erase() returns iterator of next element..
            }
            else
                ++it;
        }
        else
            ++it;
    }
}

template <typename TF>
void Immersed_boundary<TF>::create()
{
    if (sw_ib == IB_type::Disabled)
        return;

    // Init the toolbox classes.
    boundary_cyclic.init();

    // Get grid and MPI information
    auto& gd  = grid.get_grid_data();
    auto& mpi = master.get_MPI_data();

    if (sw_ib == IB_type::DEM)
    {
        // Offsets used in the 2D DEM interpolation
        const int mpi_offset_x = -mpi.mpicoordx * gd.imax + gd.igc;
        const int mpi_offset_y = -mpi.mpicoordy * gd.jmax + gd.jgc;

        // Read the IB height (DEM) map
        char filename[256] = "dem.0000000";
        auto tmp = fields.get_tmp();
        master.print_message("Loading \"%s\" ... ", filename);

        if (field3d_io.load_xy_slice(dem.data(), tmp->fld.data(), filename))
        {
            master.print_message("FAILED\n");
            throw std::runtime_error("Reading input DEM field failed");
        }
        else
        {
            master.print_message("OK\n");
        }

        fields.release_tmp(tmp);
        boundary_cyclic.exec_2d(dem.data());

        if (sw_dem_halo_replicate)
        {
            const int jj = gd.icells;

            if (mpi.mpicoordx == 0)
                for (int j=0; j<gd.jcells; ++j)
                    for (int i=0; i<gd.igc; ++i)
                        dem[i + j*jj] = dem[gd.istart + j*jj];

            if (mpi.mpicoordx == mpi.npx-1)
                for (int j=0; j<gd.jcells; ++j)
                    for (int i=gd.iend; i<gd.icells; ++i)
                        dem[i + j*jj] = dem[(gd.iend-1) + j*jj];

            if (mpi.mpicoordy == 0)
                for (int j=0; j<gd.jgc; ++j)
                    for (int i=0; i<gd.icells; ++i)
                        dem[i + j*jj] = dem[i + gd.jstart*jj];

            if (mpi.mpicoordy == mpi.npy-1)
                for (int j=gd.jend; j<gd.jcells; ++j)
                    for (int i=0; i<gd.icells; ++i)
                        dem[i + j*jj] = dem[i + (gd.jend-1)*jj];

            master.print_message(
                    "IB: DEM halo replicated at the outer domain edges "
                    "(sw_dem_halo_replicate=true)\n");
        }

        // Find ghost cells (grid points inside IB, which have at least one
        // neighbouring grid point outside of IB). Different for each
        // location on staggered grid.
        ghost.emplace("u", Ghost_cells<TF>());
        ghost.emplace("v", Ghost_cells<TF>());
        ghost.emplace("w", Ghost_cells<TF>());

        master.print_message("Calculating ghost cells u\n");
        calc_ghost_cells(
                ghost.at("u"), dem, gd.xh, gd.y, gd.z,
                Boundary_type::Dirichlet_type,
                gd.dx, gd.dy, gd.dz, n_idw_points,
                gd.istart, gd.jstart, gd.kstart,
                gd.iend,   gd.jend,   gd.kend,
                gd.icells, gd.jcells, gd.ijcells,
                mpi_offset_x, mpi_offset_y);

        master.print_message("Calculating ghost cells v\n");
        calc_ghost_cells(
                ghost.at("v"), dem, gd.x, gd.yh, gd.z,
                Boundary_type::Dirichlet_type,
                gd.dx, gd.dy, gd.dz, n_idw_points,
                gd.istart, gd.jstart, gd.kstart,
                gd.iend,   gd.jend,   gd.kend,
                gd.icells, gd.jcells, gd.ijcells,
                mpi_offset_x, mpi_offset_y);

        master.print_message("Calculating ghost cells w\n");
        calc_ghost_cells(
                ghost.at("w"), dem, gd.x, gd.y, gd.zh,
                Boundary_type::Dirichlet_type,
                gd.dx, gd.dy, gd.dzh, n_idw_points,
                gd.istart, gd.jstart, gd.kstart,
                gd.iend,   gd.jend,   gd.kend,
                gd.icells, gd.jcells, gd.ijcells,
                mpi_offset_x, mpi_offset_y);

        // Print some statistics (number of ghost cells)
        print_statistics(ghost.at("u").i, std::string("u"), master);
        print_statistics(ghost.at("v").i, std::string("v"), master);
        print_statistics(ghost.at("w").i, std::string("w"), master);

        // Momentum boundary condition
        ghost.at("u").mbot.resize(ghost.at("u").nghost);
        ghost.at("v").mbot.resize(ghost.at("v").nghost);
        ghost.at("w").mbot.resize(ghost.at("w").nghost);

        std::fill(ghost.at("u").mbot.begin(), ghost.at("u").mbot.end(), 0.);
        std::fill(ghost.at("v").mbot.begin(), ghost.at("v").mbot.begin(), 0.);
        std::fill(ghost.at("w").mbot.begin(), ghost.at("w").mbot.begin(), 0.);

        if (fields.sp.size() > 0)
        {
            ghost.emplace("s", Ghost_cells<TF>());

            master.print_message("Calculating ghost cells s\n");
            calc_ghost_cells(
                    ghost.at("s"), dem, gd.x, gd.y, gd.z, sbcbot,
                    gd.dx, gd.dy, gd.dz, n_idw_points,
                    gd.istart, gd.jstart, gd.kstart,
                    gd.iend,   gd.jend,   gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    mpi_offset_x, mpi_offset_y);

            print_statistics(ghost.at("s").i, std::string("s"), master);

            // Read spatially varying boundary conditions (if necessary)
            for (auto& scalar : fields.sp)
            {
                ghost.at("s").sbot.emplace(scalar.first, std::vector<TF>(ghost.at("s").nghost));

                if (std::find(sbot_spatial_list.begin(), sbot_spatial_list.end(), scalar.first) != sbot_spatial_list.end())
                {
                    // Read 2D sbot into tmp field
                    auto tmp = fields.get_tmp();

                    std::string sbot_file = scalar.first + "_sbot.0000000";
                    master.print_message("Loading \"%s\" ... ", sbot_file.c_str());

                    if (field3d_io.load_xy_slice(tmp->fld_bot.data(), tmp->fld.data(), sbot_file.c_str()))
                    {
                        master.print_message("FAILED\n");
                        throw std::runtime_error("Reading input sbot field failed");
                    }
                    else
                        master.print_message("OK\n");


                    sbot_2d.emplace(scalar.first,
                                    std::vector<TF>(gd.ijcells, TF(0)));
                    std::copy(tmp->fld_bot.begin(),
                              tmp->fld_bot.begin() + gd.ijcells,
                              sbot_2d.at(scalar.first).begin());
                    // Interpolate 2D sbot onto the ghost cell boundary locations

                    for (int i=0; i<ghost.at("s").nghost; ++i)
                    {
                        ghost.at("s").sbot.at(scalar.first)[i] =
                            interp2_dem(ghost.at("s").xb[i], ghost.at("s").yb[i],
                                   gd.x, gd.y, tmp->fld_bot, gd.dx, gd.dy,
                                   gd.icells, gd.jcells, mpi_offset_x, mpi_offset_y);
                    }
                }
                else
                {
                    for (int i=0; i<ghost.at("s").nghost; ++i)
                        ghost.at("s").sbot.at(scalar.first)[i] = sbc.at(scalar.first);
                }
            }

        }

        // Create the array with vertical indices that give the first cell above the DEM. 

        if (sw_wall_model != IB_wall_type::Disabled)
        {
            find_wall_cells<TF>(
                    wall, dem, gd.x, gd.y, gd.z, gd.dz, gd.dx, gd.dy,
                    z0m_ib, z0h_ib,
                    gd.istart, gd.jstart, gd.kstart,
                    gd.iend,   gd.jend,   gd.kend,
                    gd.icells, gd.jcells,
                    mpi_offset_x, mpi_offset_y);

            // The floor face of each column, by 2-D index.
            std::vector<int> wall_floor(gd.ijcells, -1);
            for (int m=0; m<wall.n; ++m)
                if (wall.axis[m] == 2)
                    wall_floor[wall.i[m] + wall.j[m]*gd.icells] = m;

            // Kept as a member: the momentum wall faces (sw_momentum_flux)
            // need it to find the u* of the column they sit in.
            wall_floor_ij = wall_floor;
            hfss_ij.assign(gd.ijcells, TF(0));
            hfls_ij.assign(gd.ijcells, TF(0));

            for (auto& sp : fields.sp)
            {
                wall_flux.emplace(sp.first, std::vector<TF>(wall.n, TF(0)));
                wall_value_face.emplace(sp.first, std::vector<TF>(wall.n, TF(0)));
            }


            if (sw_momentum_flux)
            {
                find_wall_cells_stag<TF>(
                        wall_u, dem, gd.xh, gd.y, gd.z, gd.x, gd.y, gd.dz,
                        gd.dx, gd.dy, z0m_ib, z0h_ib,
                        gd.istart, gd.jstart, gd.kstart,
                        gd.iend,   gd.jend,   gd.kend,
                        gd.icells, gd.jcells, mpi_offset_x, mpi_offset_y);

                find_wall_cells_stag<TF>(
                        wall_v, dem, gd.x, gd.yh, gd.z, gd.x, gd.y, gd.dz,
                        gd.dx, gd.dy, z0m_ib, z0h_ib,
                        gd.istart, gd.jstart, gd.kstart,
                        gd.iend,   gd.jend,   gd.kend,
                        gd.icells, gd.jcells, mpi_offset_x, mpi_offset_y);

                auto map_floor = [&](const Wall_cells<TF>& wc,
                                     std::vector<int>& out, const int di,
                                     const int dj)
                {
                    out.assign(wc.n, -1);
                    for (int m=0; m<wc.n; ++m)
                    {
                        const int ij = wc.i[m] + wc.j[m]*gd.icells;
                        int mf = wall_floor_ij[ij];
                        if (mf < 0)
                            mf = wall_floor_ij[(wc.i[m]-di)
                                             + (wc.j[m]-dj)*gd.icells];
                        out[m] = mf;
                    }
                };
                map_floor(wall_u, wall_u_floor, 1, 0);
                map_floor(wall_v, wall_v_floor, 0, 1);

                int nu_orphan = 0, nv_orphan = 0;
                for (int m=0; m<wall_u.n; ++m) if (wall_u_floor[m] < 0) ++nu_orphan;
                for (int m=0; m<wall_v.n; ++m) if (wall_v_floor[m] < 0) ++nv_orphan;

                master.print_message(
                        "IB: momentum wall faces: %d on the u grid, %d on the "
                        "v grid (%d and %d with no floor face to take u* "
                        "from, left on the operator's stress)\n",
                        wall_u.n, wall_v.n, nu_orphan, nv_orphan);
            }
            if (sw_scalar_flux)
            {
                int nfloor = 0;
                for (int m=0; m<wall.n; ++m)
                    if (wall.axis[m] == 2) ++nfloor;
                master.print_message(
                        "IB: scalars exchange through a MOST SOURCE TERM on "
                        "%d wall faces (%d floor, %d riser); stability on "
                        "%s\n",
                        wall.n, nfloor, wall.n - nfloor,
                        sw_wall_stability_vertical ? "all faces"
                                                   : "floor faces only");
            }


            auto map_ghosts = [&](
                    Ghost_cells<TF>& gh, const int axis,
                    const std::vector<TF>& xl, const std::vector<TF>& yl,
                    const std::vector<TF>& zl)
            {
                gh.a_wall  .assign(gh.nghost, TF(2));
                gh.nsq     .assign(gh.nghost, TF(0));
                gh.wall_idx.assign(gh.nghost, -1);

                for (int n=0; n<gh.nghost; ++n)
                {
                    const int i = gh.i[n];
                    const int j = gh.j[n];
                    const int k = gh.k[n];

                    // Wall normal, from the ghost cell to its wall point.
                    const TF nx = gh.xb[n] - xl[i];
                    const TF ny = gh.yb[n] - yl[j];
                    const TF nz = gh.zb[n] - zl[k];
                    const TF nn = nx*nx + ny*ny + nz*nz;

                    if (axis >= 0)
                    {
                        if (nn < TF(1e-12))
                            gh.nsq[n] = (axis == 2) ? TF(1) : TF(0);
                        else
                        {
                            const TF nc = (axis == 0) ? nx
                                        : (axis == 1) ? ny : nz;
                            gh.nsq[n] = nc*nc / nn;
                        }
                    }

                    // The floor face of this column, then of its neighbours.
                    int m = wall_floor[i + j*gd.icells];
                    for (int dj=-1; dj<2 && m<0; ++dj)
                        for (int di2=-1; di2<2 && m<0; ++di2)
                            m = wall_floor[(i+di2) + (j+dj)*gd.icells];
                    gh.wall_idx[n] = m;
                }
            };

            map_ghosts(ghost.at("u"), 0, gd.xh, gd.y,  gd.z);
            map_ghosts(ghost.at("v"), 1, gd.x,  gd.yh, gd.z);
            map_ghosts(ghost.at("w"), 2, gd.x,  gd.y,  gd.zh);
            if (fields.sp.size() > 0)
                map_ghosts(ghost.at("s"), -1, gd.x, gd.y, gd.z);

            int nsum = wall.n;
            int nfloor = 0;
            for (int m=0; m<wall.n; ++m)
                if (wall.axis[m] == 2) ++nfloor;
            master.sum(&nsum, 1);
            master.sum(&nfloor, 1);
            master.print_message(
                    "IB wall model: %s, z0m=%.3e m, z0h=%.3e m, "
                    "%d exposed faces of which %d floors\n",
                    (sw_wall_model == IB_wall_type::Most
                        ? "Monin-Obukhov with stability"
                        : "neutral log law"),
                    double(z0m_ib), double(z0h_ib), nsum, nfloor);
            master.print_message(
                    "IB wall model: the ghost cells blend between zero flux "
                    "(a=0) and the Dirichlet mirror (a=2); the wall-normal "
                    "momentum component always keeps a=2, so no-penetration "
                    "is unchanged\n");
        }
        if (sw_blank_solid)
        {
            auto build = [&](
                    std::vector<int>& idx, std::vector<int>& idx_ij,
                    const std::vector<TF>& xl,
                    const std::vector<TF>& yl,
                    const std::vector<TF>& zl,
                    const Ghost_cells<TF>* gh)
            {
                std::vector<char> is_gh(gd.ncells, 0);
                if (gh != nullptr)
                    for (int n=0; n<gh->nghost; ++n)
                        is_gh[gh->i[n] + gh->j[n]*gd.icells
                              + gh->k[n]*gd.ijcells] = 1;

                idx.clear();
                idx_ij.clear();

                for (int k=gd.kstart; k<gd.kend; ++k)
                    for (int j=gd.jstart; j<gd.jend; ++j)
                        for (int i=gd.istart; i<gd.iend; ++i)
                        {
                            const int ijk = i + j*gd.icells + k*gd.ijcells;

                            if (is_gh[ijk])
                                continue;

                            if (!is_solid(dem, xl, yl, zl, gd.dx, gd.dy,
                                          i, j, k, gd.icells, gd.jcells,
                                          mpi_offset_x, mpi_offset_y))
                                continue;

                            idx.push_back(ijk);
                            idx_ij.push_back(i + j*gd.icells);
                        }
            };

            std::vector<int> unused;

            build(blank_s, blank_s_ij, gd.x, gd.y, gd.z,
                  fields.sp.size() > 0 ? &ghost.at("s") : nullptr);
            build(blank_u, unused, gd.xh, gd.y,  gd.z,  &ghost.at("u"));
            build(blank_v, unused, gd.x,  gd.yh, gd.z,  &ghost.at("v"));
            build(blank_w, unused, gd.x,  gd.y,  gd.zh, &ghost.at("w"));

            int ns = blank_s.size();
            int nu = blank_u.size();
            master.sum(&ns, 1);
            master.sum(&nu, 1);
            master.print_message(
                    "IB: sw_blank_solid ON - %d scalar and %d u cells inside "
                    "the terrain are held at their surface value / at rest "
                    "every substep (ghost cells excluded)\n", ns, nu);
        }

        // [IB] sw_advec_wall: classify every face whose 2i5 stencil touches
        // the terrain. See apply_ib_advec_wall.py and README.md section 10.
        if (sw_advec_wall)
        {
            // Solid at scalar cell centres, halo included. is_solid() at a
            // cell centre is `z[k] <= dem[i,j]` (interp2_dem at a node returns
            // the node value), so the mask is taken straight from the DEM,
            // which carries its halo (cyclic, or replicated with patch 1).
            std::vector<char> solid(gd.ncells, 0);
            for (int k=gd.kstart; k<gd.kend; ++k)
                for (int j=0; j<gd.jcells; ++j)
                    for (int i=0; i<gd.icells; ++i)
                    {
                        const int ij = i + j*gd.icells;
                        solid[ij + k*gd.ijcells] = (gd.z[k] <= dem[ij]) ? 1 : 0;
                    }

            const int dd[3] = {1, gd.icells, gd.ijcells};
            int ncls[4] = {0, 0, 0, 0};   // third, upwind, wall_hi, wall_lo
            int nskip = 0;

            for (int a=0; a<3; ++a)
            {
                advec_face_ijk[a].clear();
                advec_face_cls[a].clear();

                const int d  = dd[a];
                // Faces c = istart..iend along the face axis (iend is the
                // face between the last interior cell and the halo). Vertical
                // faces: only those strictly inside the domain; kstart and
                // kend are MicroHH's own walls and advec treats them itself.
                const int i1 = (a == 0) ? gd.iend + 1 : gd.iend;
                const int j1 = (a == 1) ? gd.jend + 1 : gd.jend;
                const int k0 = (a == 2) ? gd.kstart + 1 : gd.kstart;

                for (int k=k0; k<gd.kend; ++k)
                    for (int j=gd.jstart; j<j1; ++j)
                        for (int i=gd.istart; i<i1; ++i)
                        {
                            const int c = i + j*gd.icells + k*gd.ijcells;
                            const bool air_lo = !solid[c-d];
                            const bool air_hi = !solid[c];
                            if (!air_lo && !air_hi)
                                continue;                      // rock-rock

                            // Does the 6-point stencil (c-3d .. c+2d), or its
                            // 4-point core (c-2d .. c+d), touch a solid cell?
                            // Vertically, cells outside kstart..kend-1 are the
                            // domain's own ghost levels, not terrain.
                            bool touch6 = false, touch4 = false;
                            for (int m=-3; m<=2; ++m)
                            {
                                if (a == 2 && (k+m < gd.kstart || k+m >= gd.kend))
                                    continue;
                                if (solid[c + m*d])
                                {
                                    touch6 = true;
                                    if (m >= -2 && m <= 1)
                                        touch4 = true;
                                }
                            }

                            signed char cls;
                            if (air_lo != air_hi)
                                cls = air_hi ? 2 : 3;          // wall face
                            else if (!touch6)
                                continue;                      // untouched
                            else if (!touch4)
                                cls = 0;                       // 3rd order
                            else
                                cls = 1;                       // 1st order

                            // Vertically, advec_2i5 uses its interior formula
                            // for both cells of face k only for
                            // kstart+3 <= k < kend-3; nearer its own walls it
                            // has already lowered the order itself.
                            if (a == 2 && (k < gd.kstart+3 || k >= gd.kend-3))
                            {
                                ++nskip;
                                continue;
                            }

                            advec_face_ijk[a].push_back(c);
                            advec_face_cls[a].push_back(cls);
                            ++ncls[cls];
                        }
            }

            master.sum(ncls, 4);
            master.sum(&nskip, 1);
            master.print_message(
                    "IB: sw_advec_wall ON - scalar faces whose 2i5 stencil "
                    "reaches into the terrain: %d to 3rd order, %d to 1st-order "
                    "upwind, %d wall faces carrying the air value only\n",
                    ncls[0], ncls[1], ncls[2] + ncls[3]);
            if (nskip > 0)
                master.print_message(
                    "IB: sw_advec_wall - %d face(s) within 3 levels of the domain "
                    "bottom/top left to advec_2i5's own near-wall treatment. Raise "
                    "the terrain off the floor (IB_MIN_SOLID_LEVELS >= 4) to "
                    "cover them.\n", nskip);
        }

        find_k_dem(
                k_dem.data(), dem.data(), gd.z.data(),
                gd.istart, gd.iend, 
                gd.jstart, gd.jend, 
                gd.kstart, gd.kend,
                gd.icells);

        boundary_cyclic.exec_2d(k_dem.data());
    }
}

template<typename TF>
bool Immersed_boundary<TF>::has_mask(std::string mask_name)
{
    if (std::find(available_masks.begin(), available_masks.end(), mask_name) != available_masks.end())
        return true;
    else
        return false;
}

template<typename TF>
void Immersed_boundary<TF>::get_mask(Stats<TF>& stats, std::string mask_name)
{
    auto& gd = grid.get_grid_data();

    auto mask  = fields.get_tmp();
    auto maskh = fields.get_tmp();

    calc_mask(
            mask->fld.data(), maskh->fld.data(), dem.data(), gd.z.data(), gd.zh.data(),
            gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
            gd.icells, gd.ijcells);

    stats.set_mask_thres("ib", *mask, *maskh, TF(0.5), Stats_mask_type::Plus);

    fields.release_tmp(mask );
    fields.release_tmp(maskh);
}


namespace
{
    // Meteorological wind direction: the direction the wind blows FROM, in
    // degrees clockwise from north. Not averageable - see the header of
    // apply_ib_column_diag.py.
    template<typename TF>
    TF wind_from_deg(const TF u, const TF v)
    {
        if (std::abs(u) < TF(1e-12) && std::abs(v) < TF(1e-12))
            return TF(0);
        TF d = TF(270) - std::atan2(v, u) * TF(180) / TF(M_PI);
        while (d < TF(0))    d += TF(360);
        while (d >= TF(360)) d -= TF(360);
        return d;
    }
}

template<typename TF>
bool Immersed_boundary<TF>::calc_surface_diag(
        const std::string& name_in, TF* const restrict out)
{
    auto& gd = grid.get_grid_data();
    const int ii = 1;
    const int jj = gd.icells;

    std::fill(out, out + gd.ijcells, TF(0));

    /*
     * A name may carry a FACE SELECTOR:
     *
     *   <name>          every wall face of the column
     *   <name>_floor    the horizontal top of the step only
     *   <name>_wall     the vertical risers only
     *
     * The split is the whole point for a staircase IB. A column's floor face
     * is where Monin-Obukhov properly applies; its risers are the staircase
     * artefact, they carry the neutral log law, and with
     * sw_scalar_flux_vertical=0 they carry nothing at all while still having
     * their diffusive flux removed. Summing the two together hides exactly
     * the term worth looking at.
     */
    int sel = 0;                                    // 0 all, 1 floor, 2 riser
    std::string name = name_in;
    {
        const std::string f = "_floor", w = "_wall";
        if (name.size() > f.size()
                && name.compare(name.size()-f.size(), f.size(), f) == 0)
        {
            sel = 1;
            name = name.substr(0, name.size()-f.size());
        }
        else if (name.size() > w.size()
                 && name.compare(name.size()-w.size(), w.size(), w) == 0)
        {
            sel = 2;
            name = name.substr(0, name.size()-w.size());
        }
    }
    auto face_ok = [sel](const int axis)
    {
        return sel == 0 || (sel == 1 && axis == 2) || (sel == 2 && axis != 2);
    };
    // Every face is weighted by its area RELATIVE TO THE FLOOR face, so a
    // flux comes out per unit GROUND area and the numbers of one column add
    // up: <x>_floor + <x>_wall == <x>.
    auto w_area = [&](const int m)
    {
        return (wall.axis[m] == 2) ? TF(1)
             : (wall.axis[m] == 0) ? gd.dz[wall.k[m]]*gd.dxi
                                   : gd.dz[wall.k[m]]*gd.dyi;
    };

    // ---- pure geometry: no wall model needed -------------------------------
    if (name == "z_dem" || name == "k_dem")
    {
        const bool want_z = (name == "z_dem");
        for (int j=0; j<gd.jcells; ++j)
            for (int i=0; i<gd.icells; ++i)
            {
                const int ij = i + j*jj;
                out[ij] = want_z ? dem[ij] : TF(k_dem[ij]);
            }
        return true;
    }

    // How many faces of each kind the column has, and how much riser area it
    // presents per unit ground area. `area_wall` is the factor by which a
    // riser exchange coefficient is multiplied before it reaches the cell.
    if (name_in == "nface" || name_in == "nface_floor"
            || name_in == "nface_wall" || name_in == "area_wall")
    {
        const bool want_area = (name_in == "area_wall");
        for (int m=0; m<wall.n; ++m)
        {
            if (want_area)
            {
                if (wall.axis[m] == 2) continue;
                out[wall.i[m] + wall.j[m]*jj] += w_area(m);
            }
            else if (face_ok(wall.axis[m]))
                out[wall.i[m] + wall.j[m]*jj] += TF(1);
        }
        boundary_cyclic.exec_2d(out);
        return true;
    }

    // ---- fluxes summed over the selected faces -----------------------------
    {
        const std::string tag = "_fluxbot_ib";
        if (name.size() > tag.size()
                && name.compare(name.size()-tag.size(), tag.size(), tag) == 0)
        {
            const std::string scalar = name.substr(0, name.size()-tag.size());
            if (!(sw_scalar_flux && wall.n > 0 && wall_flux.count(scalar)))
                return true;                        // left at zero, honestly
            const std::vector<TF>& F = wall_flux.at(scalar);
            for (int m=0; m<wall.n; ++m)
                if (face_ok(wall.axis[m]))
                    out[wall.i[m] + wall.j[m]*jj] += F[m] * w_area(m);
            boundary_cyclic.exec_2d(out);
            return true;
        }
    }

    if (name == "hfss_ib" || name == "hfls_ib")
    {
        const bool sens = (name == "hfss_ib");
        // Unsplit, the value exec_wall_model already accumulated is the same
        // number and cheaper; split, it has to be rebuilt from wall_flux,
        // which is where that accumulation came from in the first place.
        if (sel == 0)
        {
            const std::vector<TF>& src = sens ? hfss_ij : hfls_ij;
            if (src.size() == size_t(gd.ijcells))
                std::copy(src.begin(), src.end(), out);
            return true;
        }
        const std::string sc = sens ? "thl" : "qt";
        if (!wall_flux.count(sc) || exner_ref.empty())
            return true;
        const std::vector<TF>& F = wall_flux.at(sc);
        for (int m=0; m<wall.n; ++m)
        {
            if (!face_ok(wall.axis[m]))
                continue;
            const int k = wall.k[m];
            const TF c = sens ? (Constants::cp<TF> * exner_ref[k])
                              :  Constants::Lv<TF>;
            out[wall.i[m] + wall.j[m]*jj] +=
                    fields.rhoref[k] * c * F[m] * w_area(m);
        }
        boundary_cyclic.exec_2d(out);
        return true;
    }

    // ---- per-face quantities: AREA-WEIGHTED MEAN over the selection --------
    // ustar, obuk, ch and the wall value are not extensive, so they are
    // averaged, not summed - weighted by face area, which is how they enter
    // the flux. A column with no face of the requested kind gets NaN rather
    // than 0: an Obukhov length of zero would look like a number.
    if (name == "ustar_ib" || name == "obuk_ib" || name == "ch_ib"
            || name == "thl_sbot_ib" || name == "qt_sbot_ib")
    {
        if (sw_wall_model == IB_wall_type::Disabled && name[0] != 't'
                && name[0] != 'q')
            return true;

        const bool is_sbot = (name == "thl_sbot_ib" || name == "qt_sbot_ib");
        const std::string sc = (name == "qt_sbot_ib") ? "qt" : "thl";
        auto itv = wall_value_face.find(sc);
        auto it2d = sbot_2d.find(sc);

        std::vector<TF> wsum(gd.ijcells, TF(0));
        for (int m=0; m<wall.n; ++m)
        {
            if (!face_ok(wall.axis[m]))
                continue;
            const int ij = wall.i[m] + wall.j[m]*jj;
            const TF a = w_area(m);

            TF val;
            if (name == "ustar_ib")      val = wall.ustar[m];
            else if (name == "obuk_ib")  val = wall.obuk[m];
            else if (name == "ch_ib")    val = wall.ustar[m]
                    * most::fh(wall.dn[m], wall.z0h[m], wall.obuk[m]);
            else if (itv != wall_value_face.end()
                     && itv->second.size() == size_t(wall.n))
                // The value the kernel actually used on THIS face. For a
                // riser that is not the same as the column's own surface
                // value, and seeing the difference is the point.
                val = itv->second[m];
            else
                val = (it2d != sbot_2d.end()) ? it2d->second[ij]
                    : (sbc.count(sc) ? sbc.at(sc) : TF(0));

            out[ij] += val * a;
            wsum[ij] += a;
        }
        for (int n=0; n<gd.ijcells; ++n)
            out[n] = (wsum[n] > TF(0)) ? out[n] / wsum[n]
                                       : std::numeric_limits<TF>::quiet_NaN();
        (void)is_sbot;
        boundary_cyclic.exec_2d(out);
        return true;
    }

    // ---- the per-column MOST diagnostics -----------------------------------
    // These describe the FIRST FLUID CELL of the column, not a face, so a
    // floor/wall split is meaningless and the selector is refused rather
    // than silently ignored.
    {
        static const char* diag_names[] = {
            "u_ib1", "v_ib1", "thl_ib1", "qt_ib1",
            "u_10m_ib", "v_10m_ib", "wspd_10m_ib", "wdir_10m_ib",
            "thl_2m_ib", "T_2m_ib", "qt_2m_ib"};

        bool known = false;
        for (const char* nm : diag_names)
            if (name == nm) { known = true; break; }
        if (!known)
            return false;
        if (sel != 0)
            throw std::runtime_error(
                    "[IB] \"" + name_in + "\": " + name + " is a property of "
                    "the first fluid cell, not of a wall face - it has no "
                    "_floor / _wall split");

        const TF* const u = fields.mp.at("u")->fld.data();
        const TF* const v = fields.mp.at("v")->fld.data();

        for (int m=0; m<wall.n; ++m)
        {
            if (wall.axis[m] != 2)
                continue;                       // floor faces only
            const int i = wall.i[m];
            const int j = wall.j[m];
            const int k = wall.k[m];
            const int ij  = i + j*jj;
            const int ijk = ij + k*gd.ijcells;

            // u and v brought to the cell centre, so every variable here
            // sits on the same point and a wind speed can be formed without
            // further staggering.
            const TF uc = TF(0.5)*(u[ijk] + u[ijk+ii]);
            const TF vc = TF(0.5)*(v[ijk] + v[ijk+jj]);

            const TF z1 = wall.dn[m];
            const TF L  = wall.obuk[m];

            TF val = TF(0);
            if (name == "u_ib1")        val = uc;
            else if (name == "v_ib1")   val = vc;
            else if (name == "thl_ib1" && fields.sp.count("thl"))
                val = fields.sp.at("thl")->fld[ijk];
            else if (name == "qt_ib1" && fields.sp.count("qt"))
                val = fields.sp.at("qt")->fld[ijk];
            else if (name == "u_10m_ib" || name == "v_10m_ib"
                     || name == "wspd_10m_ib" || name == "wdir_10m_ib")
            {
                const TF zd = std::max(diag_z_mom, wall.z0m[m]*TF(1.001));
                const TF r  = most::fm(z1, wall.z0m[m], L)
                            / most::fm(zd, wall.z0m[m], L);
                const TF ud = uc * r;
                const TF vd = vc * r;
                if (name == "u_10m_ib")         val = ud;
                else if (name == "v_10m_ib")    val = vd;
                else if (name == "wspd_10m_ib") val = std::sqrt(ud*ud + vd*vd);
                else                            val = wind_from_deg(ud, vd);
            }
            else if (name == "thl_2m_ib" || name == "qt_2m_ib"
                     || name == "T_2m_ib")
            {
                const std::string s2 = (name == "qt_2m_ib") ? "qt" : "thl";
                if (!fields.sp.count(s2))
                    continue;
                auto i2 = sbot_2d.find(s2);
                const TF pw = (i2 != sbot_2d.end()) ? i2->second[ij]
                            : (sbc.count(s2) ? sbc.at(s2) : TF(0));
                const TF p1 = fields.sp.at(s2)->fld[ijk];
                const TF zd = std::max(diag_z_scalar, wall.z0h[m]*TF(1.001));
                const TF r  = most::fh(z1, wall.z0h[m], L)
                            / most::fh(zd, wall.z0h[m], L);
                val = pw + (p1 - pw) * r;

                // T_2m_ib is the DRY temperature: exner(p_k) * thl_2m, with
                // p_k the base state of the first fluid cell. The condensate
                // terms of T = exner*thl + Lv/cp*ql + Ls/cp*qi are left out
                // on purpose - this is a MOST extrapolation toward the wall,
                // not a thermodynamic state - so in fog it reads low.
                if (name == "T_2m_ib")
                {
                    if (k < int(exner_ref.size()))
                        val *= exner_ref[k];
                    else
                        val = TF(0);
                }
            }

            out[ij] = val;
        }
        boundary_cyclic.exec_2d(out);
        return true;
    }
}


template<typename TF>
void Immersed_boundary<TF>::create_column(Column<TF>& column)
{
    if (sw_ib == IB_type::Disabled)
        return;

    // The mask first: a column file that carries a profile without saying
    // which part of it is rock invites exactly the mistake this whole
    // exercise is about.
    column.add_prof("ib_mask", "1 where the cell is air, 0 where it is inside "
                               "the immersed boundary", "-", "z");

    static const struct { const char* name; const char* unit;
                          const char* longname; } known[] = {
        {"thl_fluxbot_ib", "K m s-1",  "IB surface kinematic heat flux"},
        {"qt_fluxbot_ib",  "m s-1",    "IB surface kinematic moisture flux"},
        {"hfss_ib",        "W m-2",    "IB sensible heat flux"},
        {"hfls_ib",        "W m-2",    "IB latent heat flux"},
        {"ustar_ib",       "m s-1",    "IB friction velocity"},
        {"obuk_ib",        "m",        "IB Obukhov length"},
        {"ch_ib",          "m s-1",    "IB scalar exchange coefficient"},
        {"thl_2m_ib",      "K",        "thl at diag_z_scalar above the IB"},
        {"T_2m_ib",        "K",        "dry temperature at diag_z_scalar above the IB"},
        {"qt_2m_ib",       "kg kg-1",  "qt at diag_z_scalar above the IB"},
        {"u_10m_ib",       "m s-1",    "u at diag_z_mom above the IB"},
        {"v_10m_ib",       "m s-1",    "v at diag_z_mom above the IB"},
        {"wspd_10m_ib",    "m s-1",    "wind speed at diag_z_mom above the IB"},
        {"wdir_10m_ib",    "degrees",  "wind direction (from) at diag_z_mom above the IB"},
        {"thl_sbot_ib",    "K",        "IB surface thl"},
        {"qt_sbot_ib",     "kg kg-1",  "IB surface qt"},
        {"u_ib1",          "m s-1",    "u at the first fluid cell"},
        {"v_ib1",          "m s-1",    "v at the first fluid cell"},
        {"thl_ib1",        "K",        "thl at the first fluid cell"},
        {"qt_ib1",         "kg kg-1",  "qt at the first fluid cell"},
        {"z_dem",          "m",        "terrain height of the column"},
        {"k_dem",          "-",        "first air level of the column (0-based)"},
        {"nface",          "-",        "wall faces of the column"},
        {"nface_floor",    "-",        "floor faces of the column"},
        {"nface_wall",     "-",        "riser faces of the column"},
        {"area_wall",      "-",        "riser area per unit ground area"},
    };

    // The per-face split. Anything that is carried BY a wall face can be
    // asked for on the floor alone or the risers alone; the two add up to
    // the unsplit value for the fluxes and are area-weighted means for the
    // rest. Registered automatically so the table above stays readable.
    static const char* splittable[] = {
        "thl_fluxbot_ib", "qt_fluxbot_ib", "hfss_ib", "hfls_ib",
        "ustar_ib", "obuk_ib", "ch_ib", "thl_sbot_ib", "qt_sbot_ib"};

    for (const std::string& s : columnlist)
    {
        bool found = false;
        for (const auto& e : known)
            if (s == e.name)
            {
                column.add_time_series(e.name, e.longname, e.unit);
                found = true;
                break;
            }
        if (found)
            continue;

        // <base>_floor / <base>_wall
        for (const char* b : splittable)
        {
            const std::string base(b);
            if (s != base + "_floor" && s != base + "_wall")
                continue;
            const bool floor = (s.size() > 6
                    && s.compare(s.size()-6, 6, "_floor") == 0);
            for (const auto& e : known)
                if (base == e.name)
                {
                    column.add_time_series(
                            s, std::string(e.longname) + (floor
                                ? " (floor faces only)"
                                : " (riser faces only, area-weighted; NaN "
                                  "where the column has none)"),
                            e.unit);
                    found = true;
                    break;
                }
            break;
        }
        if (!found)
            throw std::runtime_error(
                    "[IB] columnlist: \"" + s + "\" is not an IB diagnostic");
    }

    master.print_message(
            "IB: %d diagnostic(s) added to the column output, plus ib_mask\n",
            int(columnlist.size()));
}


#ifndef USECUDA
template<typename TF>
void Immersed_boundary<TF>::exec_column(Column<TF>& column, Thermo<TF>& thermo)
{
    if (sw_ib == IB_type::Disabled)
        return;

    auto& gd = grid.get_grid_data();

    if (exner_ref.empty())
        exner_ref = thermo.get_basestate_vector("exner");

    // ---- the mask, as a 3-D field the column machinery can slice ----------
    // MicroHH's own test: solid where z[k] <= dem (immersed_boundary.cxx:353).
    {
        auto tmp = fields.get_tmp();
        for (int k=0; k<gd.kcells; ++k)
            for (int j=0; j<gd.jcells; ++j)
                for (int i=0; i<gd.icells; ++i)
                {
                    const int ij  = i + j*gd.icells;
                    const int ijk = ij + k*gd.ijcells;
                    tmp->fld[ijk] = (gd.z[k] <= dem[ij]) ? TF(0) : TF(1);
                }
        column.calc_column("ib_mask", tmp->fld.data(), TF(0));
        fields.release_tmp(tmp);
    }

    if (columnlist.empty())
        return;

    auto tmp = fields.get_tmp();
    for (const std::string& s : columnlist)
    {
        std::fill(tmp->flux_bot.begin(), tmp->flux_bot.end(), TF(0));
        if (calc_surface_diag(s, tmp->flux_bot.data()))
            column.calc_time_series(s, tmp->flux_bot.data(), TF(0));
    }
    fields.release_tmp(tmp);
}
#endif

template<typename TF>
void Immersed_boundary<TF>::exec_cross(Cross<TF>& cross, unsigned long iotime)
{
    auto& gd = grid.get_grid_data();
    TF no_offset = 0.;

    if (cross.get_switch())
    {
        for (auto& s : crosslist)
        {
            // ---- terrain-following xy planes: <var>_tf / <var>_tf<h> -----
            // A horizontal slice at a fixed height ABOVE THE LOCAL SURFACE.
            // `<var>_tf` writes one plane per [IB] tf_cross_heights entry;
            // `<var>_tf<h>` writes that one height. See apply_ib_tf_cross.py.
            {
                const size_t p_tf = s.rfind("_tf");
                const bool is_tf = sw_tf_cross && p_tf != std::string::npos
                        && (p_tf + 3 == s.size()
                            || s.find_first_not_of("0123456789", p_tf+3)
                               == std::string::npos);
                if (is_tf)
                {
                    const std::string var = s.substr(0, p_tf);

                    std::vector<TF> heights;
                    if (p_tf + 3 == s.size())
                        heights = tf_cross_heights;
                    else
                        heights.push_back(TF(std::stod(s.substr(p_tf+3))));

                    const bool want_wspd = (var == "wspd");
                    const bool want_wdir = (var == "wdir");
                    const bool want_uv   = (var == "u" || var == "v"
                                            || want_wspd || want_wdir);
                    const bool want_w    = (var == "w");

                    const TF* fld = nullptr;
                    if (!want_uv && !want_w)
                    {
                        if (fields.sp.find(var) == fields.sp.end())
                            continue;               // unknown: nothing to do
                        fld = fields.sp.at(var)->fld.data();
                    }

                    const TF* const uf = fields.mp.at("u")->fld.data();
                    const TF* const vf = fields.mp.at("v")->fld.data();
                    const TF* const wf = fields.mp.at("w")->fld.data();
                    const std::vector<TF>& zax = want_w ? gd.zh : gd.z;
                    const TF rad2deg = TF(180) / std::acos(TF(-1));

                    auto tmpd = fields.get_tmp();
                    const int ii = 1;
                    const int jj = gd.icells;

                    for (size_t q=0; q<heights.size(); ++q)
                    {
                        const TF h_tf = heights[q];
                        std::fill(tmpd->flux_bot.begin(),
                                  tmpd->flux_bot.end(), TF(0));

                        for (int j=gd.jstart; j<gd.jend; ++j)
                            for (int i=gd.istart; i<gd.iend; ++i)
                            {
                                const int ij = i + j*jj;
                                const TF zt = dem[ij] + h_tf;

                                // The two levels bracketing zt, never below
                                // the column's first AIR level: below that is
                                // rock, and the value there is not a value.
                                int k0 = std::max(int(k_dem[ij]), gd.kstart);
                                while (k0+1 < gd.kend && zax[k0+1] <= zt)
                                    ++k0;
                                const int k1 = std::min(k0+1, gd.kend-1);
                                TF f = TF(0);
                                if (k1 > k0 && zax[k1] > zax[k0])
                                    f = std::min(std::max(
                                            (zt - zax[k0])/(zax[k1] - zax[k0]),
                                            TF(0)), TF(1));

                                const int ijk0 = ij + k0*gd.ijcells;
                                const int ijk1 = ij + k1*gd.ijcells;

                                TF val = TF(0);
                                if (want_w)
                                    val = (TF(1)-f)*wf[ijk0] + f*wf[ijk1];
                                else if (want_uv)
                                {
                                    // to the cell centre, so u, v, wspd and
                                    // wdir all sit on the same point
                                    const TF u0 = TF(0.5)*(uf[ijk0] + uf[ijk0+ii]);
                                    const TF u1 = TF(0.5)*(uf[ijk1] + uf[ijk1+ii]);
                                    const TF v0 = TF(0.5)*(vf[ijk0] + vf[ijk0+jj]);
                                    const TF v1 = TF(0.5)*(vf[ijk1] + vf[ijk1+jj]);
                                    const TF uc = (TF(1)-f)*u0 + f*u1;
                                    const TF vc = (TF(1)-f)*v0 + f*v1;
                                    const TF sp = std::sqrt(uc*uc + vc*vc);
                                    if (var == "u")          val = uc;
                                    else if (var == "v")     val = vc;
                                    else if (want_wspd)      val = sp;
                                    else
                                        // meteorological: the direction it
                                        // comes FROM, clockwise from north
                                        val = (sp > TF(0))
                                            ? std::fmod(TF(270)
                                                - std::atan2(vc, uc)*rad2deg
                                                + TF(360), TF(360))
                                            : TF(0);
                                }
                                else
                                    val = (TF(1)-f)*fld[ijk0] + f*fld[ijk1];

                                tmpd->flux_bot[ij] = val;
                            }

                        boundary_cyclic.exec_2d(tmpd->flux_bot.data());
                        const std::string name_h = var + "_tf"
                                + std::to_string(int(std::lround(h_tf)));
                        cross.cross_plane(tmpd->flux_bot.data(), no_offset,
                                          name_h, iotime);
                    }
                    fields.release_tmp(tmpd);
                    continue;
                }
            }


            {
                // One implementation, two outputs. Everything the inline
                // block here used to do - and the three diagnostics added
                // since - now lives in calc_surface_diag, which exec_column
                // calls as well, so a cross-section and a column time series
                // of the same quantity cannot drift apart.
                auto tmpc = fields.get_tmp();
                std::fill(tmpc->flux_bot.begin(),
                          tmpc->flux_bot.end(), TF(0));
                const bool handled =
                        calc_surface_diag(s, tmpc->flux_bot.data());
                if (handled)
                    cross.cross_plane(tmpc->flux_bot.data(), no_offset,
                                      s, iotime);
                fields.release_tmp(tmpc);
                if (handled)
                    continue;
            }
            const std::string dsurf_ib_string = "dsurf_ib";
            if (has_ending(s, dsurf_ib_string))
            {
                std::string scalar = s;
                scalar.erase(s.length() - dsurf_ib_string.length());

                auto tmpd = fields.get_tmp();
                std::fill(tmpd->flux_bot.begin(), tmpd->flux_bot.end(), TF(0));

                const TF* const restrict fld = fields.sp.at(scalar)->fld.data();
                auto it2d = sbot_2d.find(scalar);

                for (int m=0; m<wall.n; ++m)
                {
                    if (wall.axis[m] != 2)
                        continue;
                    const int ij  = wall.i[m] + wall.j[m]*gd.icells;
                    const int ijk = ij + wall.k[m]*gd.ijcells;
                    const TF pw = (it2d != sbot_2d.end()) ? it2d->second[ij]
                                                          : sbc.at(scalar);
                    tmpd->flux_bot[ij] = pw - fld[ijk];
                }

                boundary_cyclic.exec_2d(tmpd->flux_bot.data());
                cross.cross_plane(tmpd->flux_bot.data(), no_offset,
                                  scalar+"_dsurf_ib", iotime);
                fields.release_tmp(tmpd);
                continue;
            }

            const std::string fluxbot_ib_string = "fluxbot_ib";
            if (has_ending(s, fluxbot_ib_string))
            {
                // Strip the scalar from the fluxbot_ib
                std::string scalar = s;
                scalar.erase(s.length() - fluxbot_ib_string.length());

                if (sw_scalar_flux && wall.n > 0 && wall_flux.count(scalar))
                {
                    auto tmpw = fields.get_tmp();
                    std::fill(tmpw->flux_bot.begin(),
                              tmpw->flux_bot.end(), TF(0));

                    const std::vector<TF>& F = wall_flux.at(scalar);
                    for (int m=0; m<wall.n; ++m)
                    {
                        const int ij = wall.i[m] + wall.j[m]*gd.icells;
                        const TF w_area =
                                (wall.axis[m] == 2) ? TF(1)
                              : (wall.axis[m] == 0) ? gd.dz[wall.k[m]]*gd.dxi
                                                    : gd.dz[wall.k[m]]*gd.dyi;
                        tmpw->flux_bot[ij] += F[m] * w_area;
                    }

                    boundary_cyclic.exec_2d(tmpw->flux_bot.data());
                    cross.cross_plane(tmpw->flux_bot.data(), no_offset,
                                      scalar+"_fluxbot_ib", iotime);
                    fields.release_tmp(tmpw);
                    continue;
                }

                auto tmp = fields.get_tmp();

                calc_fluxes(
                        tmp->flux_bot.data(), k_dem.data(),
                        fields.sp.at(scalar)->fld.data(),
                        fields.sd.at("evisc")->fld.data(),
                        TF(1) / tPr_ib,
                        gd.dx,  gd.dy,  gd.dz.data(),
                        gd.dxi, gd.dyi, gd.dzhi.data(),
                        fields.sp.at(scalar)->visc,
                        gd.istart, gd.iend, 
                        gd.jstart, gd.jend, 
                        gd.kstart, gd.kend,
                        gd.icells, gd.ijcells);

                cross.cross_plane(tmp->flux_bot.data(), no_offset, scalar+"_fluxbot_ib", iotime);

                fields.release_tmp(tmp);
            }
        }

        if (sw_wall_model != IB_wall_type::Disabled)
        {
            for (const std::string& name : {"ustar_ib", "obuk_ib", "ch_ib"})
            {
                if (std::find(crosslist_wall.begin(), crosslist_wall.end(),
                              name) == crosslist_wall.end())
                    continue;

                auto tmp = fields.get_tmp();
                std::fill(tmp->flux_bot.begin(), tmp->flux_bot.end(), TF(0));

                if (name == "ch_ib")
                {
                    for (int m=0; m<wall.n; ++m)
                        if (wall.axis[m] == 2)
                            tmp->flux_bot[wall.i[m] + wall.j[m]*gd.icells] =
                                    wall.ustar[m] * most::fh(wall.dn[m],
                                                             wall.z0h[m],
                                                             wall.obuk[m]);
                }
                else
                {
                const std::vector<TF>& src =
                        (name == "ustar_ib") ? wall.ustar : wall.obuk;

                for (int m=0; m<wall.n; ++m)
                    if (wall.axis[m] == 2)
                        tmp->flux_bot[wall.i[m] + wall.j[m]*gd.icells] = src[m];
                }

                cross.cross_plane(tmp->flux_bot.data(), no_offset, name, iotime);
                fields.release_tmp(tmp);
            }
        }
    }
}

// ===========================================================================
//   [IB] sw_advec_wall - apply_ib_advec_wall.py, README.md section 10
// ===========================================================================
#ifndef USECUDA
namespace
{
    // Replace advec_2i5's flux through each listed face by an air-only one.
    //
    // Face n lies between cells c-d and c (c = face_ijk[n]). The flux
    // advec_2i5 put there is recomputed with the scheme's own formula:
    //   unlimited: F = u*interp6_ws(s[c-3d..c+2d]) - |u|*interp5_ws(...)
    //              (advec_2i5.cxx advec_s, all horizontal faces and vertical
    //               faces kstart+3 <= k < kend-3)
    //   limited:   F = flux_lim(u, s[c-2d], s[c-d], s[c], s[c+d])
    //              (advec_monotonic.h advec_s_lim)
    // and the difference to the replacement is added to both cells, so the
    // net effect is exactly as if advec had used the replacement.
    template<typename TF>
    void advec_wall_correct(
            TF* const restrict st, const TF* const restrict s,
            const TF* const restrict vel,
            const std::vector<int>& face_ijk,
            const std::vector<signed char>& face_cls,
            const int d, const bool vertical, const bool limited,
            const TF dxi, const TF* const restrict dzi,
            const TF* const restrict rhoref, const TF* const restrict rhorefh,
            const int ijcells, TF& max_dst)
    {
        namespace fd4 = Finite_difference::O4;
        namespace fd6 = Finite_difference::O6;
        const int d2 = 2*d;
        const int d3 = 3*d;

        for (std::size_t n=0; n<face_ijk.size(); ++n)
        {
            const int c   = face_ijk[n];
            const int cls = face_cls[n];
            const TF un   = vel[c];

            // 3rd order is what Koren already is: nothing to replace.
            if (limited && cls == 0)
                continue;

            const TF f_old = limited
                ? Advec_monotonic::flux_lim(un, s[c-d2], s[c-d], s[c], s[c+d])
                : un * fd6::interp6_ws(s[c-d3], s[c-d2], s[c-d], s[c], s[c+d], s[c+d2])
                  - std::abs(un) * fd6::interp5_ws(s[c-d3], s[c-d2], s[c-d], s[c], s[c+d], s[c+d2]);

            TF f_new;
            if (cls == 0)          // 4-point core all air: 3rd-order upwind
                f_new = un * fd4::interp4_ws(s[c-d2], s[c-d], s[c], s[c+d])
                      - std::abs(un) * fd4::interp3_ws(s[c-d2], s[c-d], s[c], s[c+d]);
            else if (cls == 1)     // both face cells air: 1st-order upwind
                f_new = (un > TF(0)) ? un * s[c-d] : un * s[c];
            else if (cls == 2)     // wall, air on the HIGH side
                f_new = un * s[c];
            else                   // wall, air on the LOW side
                f_new = un * s[c-d];

            const TF df = f_new - f_old;
            TF dhi, dlo;
            if (vertical)
            {
                const int k = c / ijcells;
                dhi =  rhorefh[k] * df / rhoref[k  ] * dzi[k  ];
                dlo = -rhorefh[k] * df / rhoref[k-1] * dzi[k-1];
            }
            else
            {
                dhi =  df * dxi;
                dlo = -df * dxi;
            }
            st[c  ] += dhi;
            st[c-d] += dlo;

            // largest change to an AIR cell, for the log
            if (cls != 3) max_dst = std::max(max_dst, std::abs(dhi));
            if (cls != 2) max_dst = std::max(max_dst, std::abs(dlo));
        }
    }
}

template<typename TF>
void Immersed_boundary<TF>::exec_advec_wall()
{
    if (sw_ib == IB_type::Disabled || !sw_advec_wall)
        return;

    auto& gd = grid.get_grid_data();

    const TF* vel[3] = {
            fields.mp.at("u")->fld.data(),
            fields.mp.at("v")->fld.data(),
            fields.mp.at("w")->fld.data()};
    const int dd[3] = {1, gd.icells, gd.ijcells};
    const TF dxi[3] = {TF(1)/gd.dx, TF(1)/gd.dy, TF(0)};

    for (auto& it : fields.sp)
    {
        const bool limited = std::find(
                advec_wall_limited.begin(), advec_wall_limited.end(),
                it.first) != advec_wall_limited.end();

        TF max_dst = TF(0);
        for (int a=0; a<3; ++a)
            advec_wall_correct<TF>(
                    fields.st.at(it.first)->fld.data(), it.second->fld.data(),
                    vel[a], advec_face_ijk[a], advec_face_cls[a],
                    dd[a], a == 2, limited, dxi[a], gd.dzi.data(),
                    fields.rhoref.data(), fields.rhorefh.data(),
                    gd.ijcells, max_dst);

        if (!advec_wall_reported)
        {
            master.max(&max_dst, 1);
            master.print_message(
                    "IB: sw_advec_wall - %s: largest change to an air cell's "
                    "advective tendency on the first call %.3g /s%s\n",
                    it.first.c_str(), max_dst,
                    limited ? " (flux-limited field)" : "");
        }
    }
    advec_wall_reported = true;
}
#else
template<typename TF>
void Immersed_boundary<TF>::exec_advec_wall()
{
    if (sw_ib != IB_type::Disabled && sw_advec_wall)
        throw std::runtime_error("[IB] sw_advec_wall is not implemented for GPU builds");
}
#endif

#ifdef FLOAT_SINGLE
template class Immersed_boundary<float>;
#else
template class Immersed_boundary<double>;
#endif
