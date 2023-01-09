#!/usr/bin/env python 

# code to plot the residual between two maps/output files

import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs 
from matplotlib.backends.backend_pdf import PdfPages


slip_system='010'
rad = 3530
nside = 24

xsize = 2000
ysize = xsize / 2.
theta = np.linspace(np.pi, 0, int(ysize))
phi = np.linspace(-np.pi, np.pi, int(xsize))
longitude = np.linspace(-180, 180, int(xsize))
latitude = np.linspace(-90, 90, int(ysize))

PHI, THETA = np.meshgrid(phi, theta)
grid_pix = hp.ang2pix(nside, THETA, PHI)

# sample the flow field at grid points

lons = np.arange(-180, 185, 5)
lats = np.arange(-90, 95, 5)
grid_lons, grid_lats = np.meshgrid(lons, lats)


NPIX = hp.nside2npix(nside)
colats_hp, lons_hp = np.degrees(hp.pix2ang(nside=nside, ipix=np.arange(NPIX)))
lats_hp = 90 - colats_hp

dep_file = f"/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/cij_summary_{slip_system}.txt"
indep_file = f"/Users/earjwara/work/anisotropy_flow/model01/time_indep/cij_summary_{slip_system}.txt"

path_file_dep = '/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/path_summary.txt'
path_file_indep = '/Users/earjwara/work/anisotropy_flow/model01/time_indep/path_summary.txt'


data_array_timedep = np.loadtxt(dep_file)
data_array_timeindep = np.loadtxt(indep_file)

path_array_timedep = np.loadtxt(path_file_dep)
path_array_timeindep = np.loadtxt(path_file_indep)

data_array_timedep = data_array_timedep[np.lexsort((data_array_timedep[:, 1], data_array_timedep[:, 0], data_array_timedep[:, 2]))]
data_array_timeindep = data_array_timeindep[np.lexsort((data_array_timeindep[:, 1], data_array_timeindep[:, 0], data_array_timeindep[:, 2]))]
path_array_timedep = path_array_timedep[np.lexsort((path_array_timedep[:, 1], path_array_timedep[:, 0], path_array_timedep[:, 2]))]
path_array_timeindep = path_array_timeindep[np.lexsort((path_array_timeindep[:, 1], path_array_timeindep[:, 0], path_array_timeindep[:, 2]))]

print(data_array_timedep)
print(path_array_timedep)


R, lat, lon, P, T, xi, phi_ani, eta = data_array_timedep[:,0], data_array_timedep[:,1], data_array_timedep[:,2], data_array_timedep[:,3], data_array_timedep[:,4], data_array_timedep[:,5], data_array_timedep[:,6], data_array_timedep[:,7]
R_indep, lat_indep, lon_indep, P_indep, T_indep, xi_indep, phi_ani_indep, eta_indep = data_array_timeindep[:,0], data_array_timeindep[:,1], data_array_timeindep[:,2], data_array_timeindep[:,3], data_array_timeindep[:,4], data_array_timeindep[:,5], data_array_timeindep[:,6], data_array_timeindep[:,7]

lats = lat[data_array_timedep[:,0] == int(rad)]
lons = lon[data_array_timedep[:,0] == int(rad)]
colat_rad = np.radians(90 - lats)
lon_rad = np.radians(lons)

pixel_indices = hp.ang2pix(nside, colat_rad, lon_rad)

lat_r = lat[data_array_timedep[:,0] == int(rad)]
lat_r_indep = lat_indep[data_array_timeindep[:,0] == int(rad)]

lon_r = lon[data_array_timedep[:,0] == int(rad)]
lon_r_indep = lon_indep[data_array_timeindep[:,0] == int(rad)]

pathlen_dep = path_array_timedep[path_array_timedep[:,0] == int(rad)][:,3]
pathlen_indep = path_array_timeindep[path_array_timeindep[:,0] == int(rad)][:,3]

tort_dep = path_array_timedep[path_array_timedep[:,0] == int(rad)][:,4]
tort_indep = path_array_timeindep[path_array_timeindep[:,0] == int(rad)][:,4]

mean_vel_dep = path_array_timedep[path_array_timedep[:,0] == int(rad)][:,7]
mean_vel_indep = path_array_timeindep[path_array_timeindep[:,0] == int(rad)][:,7]



xi_r = xi[data_array_timedep[:,0] == int(rad)]
phi_r = phi_ani[data_array_timedep[:,0] == int(rad)]
eta_r = eta[data_array_timedep[:,0] == int(rad)]

xi_r_indep = xi_indep[data_array_timeindep[:,0] == int(rad)]
phi_r_indep = phi_ani_indep[data_array_timeindep[:,0] == int(rad)]
eta_r_indep = eta_indep[data_array_timeindep[:,0] == int(rad)]

# print(np.unique(lon_r, return_counts=True))
# print(np.unique(lon_r_indep, return_counts=True))

# xi_r_use = []
# phi_r_use = []
# eta_r_use = []

# xi_r_use_indep = []
# phi_r_use_indep = []
# eta_r_use_indep = []


# for i,lo in enumerate(lon_r):
#     if np.round(lo, 4) in lons_hp or np.round(lat_r[i], 4) in lats_hp:
#         print(lo, lat_r[i])
#         xi_r_use.append(xi_r[i])
#         phi_r_use.append(phi_r[i])
#         eta_r_use.append(eta_r[i])

# for j,lo in enumerate(lon_r_indep):   
#     if np.round(lon_r_indep[j], 4) in lons_hp or np.round(lat_r_indep[j], 4) in lats_hp:
#         xi_r_use_indep.append(xi_r_indep[j])
#         phi_r_use_indep.append(phi_r_indep[j])
#         eta_r_use_indep.append(eta_r_indep[j])

# print(len(xi_r_use))


xi_diff = np.subtract(xi_r, xi_r_indep)
phi_diff = np.subtract(phi_r, phi_r_indep)
eta_diff = np.subtract(eta_r, eta_r_indep)

pathlen_diff = np.subtract(pathlen_dep, pathlen_indep)
tort_diff = np.subtract(tort_dep, tort_indep)

# initialise maps
m_xi = np.zeros(hp.nside2npix(nside))
m_phi = np.zeros(hp.nside2npix(nside))
m_eta = np.zeros(hp.nside2npix(nside))

m_pathlen = np.zeros(hp.nside2npix(nside))
m_tort = np.zeros(hp.nside2npix(nside))

m_xi[pixel_indices] = xi_diff
m_phi[pixel_indices] = phi_diff
m_eta[pixel_indices] = eta_diff

m_pathlen[pixel_indices] = pathlen_diff
m_tort[pixel_indices] = tort_diff


grid_map_xi = m_xi[grid_pix]
grid_map_phi = m_phi[grid_pix]
grid_map_eta = m_eta[grid_pix]

grid_map_pathlen = m_pathlen[grid_pix]
grid_map_tort = m_tort[grid_pix]

# plot all the maps 
with PdfPages(f"residual_maps_{slip_system}.pdf") as pdf:

    for grid in [[grid_map_xi, 'xi'], [grid_map_phi, 'phi'], [grid_map_eta, 'eta']]:
            print(grid[1])
            fig = plt.figure(figsize=(13,6))
            ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))
            # flip longitude to the astro convention
            image = ax.pcolormesh(longitude, latitude, grid[0],
                                   rasterized=True,
                                   transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
                                   cmap='seismic')
            
            if grid[1] == 'xi':
                plt.colorbar(image, label = "$\\xi$", ax=ax)
            elif grid[1] == 'eta':
                plt.colorbar(image, label = "$\eta$", ax=ax)
            elif grid[1] == 'phi':
                plt.colorbar(image, label = "$\phi$", ax=ax)

            ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

            gl = ax.gridlines(draw_labels=True, linewidth=0)

            gl.top_label = True
            gl.right_label = True
            gl.bottom_label = True
            gl.left_label = True

            mean_ani = np.mean(grid[0])
            ani_param = grid[1]

            ax.set_title(f"$\\{ani_param}$ difference between time indep and dep | Radius {rad} km")

            pdf.savefig()
            plt.close()

    print('pathlen difference')
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))
    # flip longitude to the astro convention
    image = ax.pcolormesh(longitude, latitude, grid_map_pathlen,
                           rasterized=True,
                           transform=ccrs.PlateCarree(), 
                           vmin = grid_map_pathlen.min(), vmax = grid_map_pathlen.max(),
                           cmap='seismic')
  
    plt.colorbar(image, label = "$\delta path length (km)$", ax=ax)

    ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0)

    gl.top_label = True
    gl.right_label = True
    gl.bottom_label = True
    gl.left_label = True

    ax.set_title(f"Path length difference between time dep and indep | Radius {rad} km")

    pdf.savefig()
    plt.close()

    print('tort difference')
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))
    # flip longitude to the astro convention
    image = plt.pcolormesh(longitude, latitude, grid_map_tort,
                           rasterized=True,
                           transform=ccrs.PlateCarree(),
                           vmin = grid_map_tort.min(), vmax = grid_map_tort.max(),
                           cmap='inferno')
  
    plt.colorbar(image, label = "$\delta path tort (km)$", ax=ax)

    ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0)

    gl.top_label = True
    gl.right_label = True
    gl.bottom_label = True
    gl.left_label = True

    ax.set_title(f"Path tortuosity difference between time dep and indep | Radius {rad} km")

    pdf.savefig()
    plt.close()
