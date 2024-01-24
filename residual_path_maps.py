#!/usr/bin/env python 

# code to plot the residual between two maps/output files

import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
import matplotlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-pf_dep', '--pathfile_dep', type=str, help='absolute path to path_summary.txt for time dep.')
parser.add_argument('-pf_indep', '--pathfile_indep', type=str, help='absolute path to path_summary.txt for time indep.')
parser.add_argument('-nside', '--nside', type=int, required=False, help='healpix nside value for resolution')
parser.add_argument('-r', '--rad', required=False, help='Radius to plot at')

args = parser.parse_args()

rad = args.rad
nside = args.nside

xsize = 2000
ysize = xsize / 2.
theta = np.linspace(np.pi, 0, int(ysize))
phi = np.linspace(-np.pi, np.pi, int(xsize))
longitude = np.linspace(-180, 180, int(xsize))
latitude = np.linspace(-90, 90, int(ysize))

PHI, THETA = np.meshgrid(phi, theta)
grid_pix = hp.ang2pix(nside, THETA, PHI)
VMIN=0
VMAX=8000

# sample the flow field at grid points

lons = np.arange(-180, 185, 5)
lats = np.arange(-90, 95, 5)
grid_lons, grid_lats = np.meshgrid(lons, lats)


NPIX = hp.nside2npix(nside)
colats_hp, lons_hp = np.degrees(hp.pix2ang(nside=nside, ipix=np.arange(NPIX)))
lats_hp = 90 - colats_hp

path_file_dep = args.pathfile_dep
path_file_indep = args.pathfile_indep



path_array_timedep = np.loadtxt(path_file_dep)
path_array_timeindep = np.loadtxt(path_file_indep)

print(path_array_timedep.shape)
print(path_array_timeindep.shape)
path_array_timedep = path_array_timedep[np.lexsort((path_array_timedep[:, 1], path_array_timedep[:, 0], path_array_timedep[:, 2]))]
path_array_timeindep = path_array_timeindep[np.lexsort((path_array_timeindep[:, 1], path_array_timeindep[:, 0], path_array_timeindep[:, 2]))]

print(path_array_timedep.shape)
print(path_array_timeindep.shape)

print(path_array_timedep)
lat = path_array_timedep[:,1]
lon = path_array_timedep[:,2]

lats = lat[path_array_timedep[:,0] == int(rad)]
lons = lon[path_array_timedep[:,0] == int(rad)]
colat_rad = np.radians(90 - lats)
lon_rad = np.radians(lons)

colat_rad = np.radians(90 - lats)
lon_rad = np.radians(lons)

pixel_indices = hp.ang2pix(nside, colat_rad, lon_rad)

pathlen_dep = path_array_timedep[path_array_timedep[:,0] == int(rad)][:,3]
pathlen_indep = path_array_timeindep[path_array_timeindep[:,0] == int(rad)][:,3]

tort_dep = path_array_timedep[path_array_timedep[:,0] == int(rad)][:,4]
tort_indep = path_array_timeindep[path_array_timeindep[:,0] == int(rad)][:,4]

mean_vel_dep = path_array_timedep[path_array_timedep[:,0] == int(rad)][:,7]
mean_vel_indep = path_array_timeindep[path_array_timeindep[:,0] == int(rad)][:,7]

print(path_array_timedep.shape)
print(path_array_timeindep.shape)

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
pathlen_dep[pathlen_dep == 0] = np.nan
pathlen_indep[pathlen_indep == 0] = np.nan

pathlen_diff = np.subtract(pathlen_dep, pathlen_indep)
tort_diff = np.subtract(tort_dep, tort_indep)

m_pathlen_dep = np.zeros(hp.nside2npix(nside))
m_pathlen_indep = np.zeros(hp.nside2npix(nside))

m_tort_dep = np.zeros(hp.nside2npix(nside))
m_tort_indep = np.zeros(hp.nside2npix(nside))


m_pathlen = np.zeros(hp.nside2npix(nside))
m_tort = np.zeros(hp.nside2npix(nside))

m_pathlen[pixel_indices] = pathlen_diff
m_pathlen_dep[pixel_indices] = pathlen_dep
m_pathlen_indep[pixel_indices] = pathlen_indep


m_tort[pixel_indices] = tort_diff
m_tort_dep[pixel_indices] = tort_dep
m_tort_indep[pixel_indices] = tort_indep



print(pathlen_indep)
print(pathlen_indep)

grid_map_pathlen = m_pathlen[grid_pix]
grid_map_path_dep = m_pathlen_dep[grid_pix]
grid_map_path_indep = m_pathlen_indep[grid_pix]

grid_map_tort = m_tort[grid_pix]
grid_map_tort_dep = m_tort_dep[grid_pix]
grid_map_tort_indep = m_tort_indep[grid_pix]


# # plot all the maps 
# with PdfPages(f"path_residual_maps.pdf") as pdf:
cmap = matplotlib.cm.get_cmap('inferno')
cmap.set_bad(color='dimgray')

fig = plt.figure(figsize=(10,13))
ax = fig.add_subplot(311,projection=ccrs.Robinson(central_longitude=120))
ax2 = fig.add_subplot(312,projection=ccrs.Robinson(central_longitude=120))
ax3 = fig.add_subplot(313,projection=ccrs.Robinson(central_longitude=120))


image = ax.pcolormesh(longitude, latitude, grid_map_path_indep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap)

image = ax.pcolormesh(longitude, latitude, grid_map_path_indep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap='inferno', vmin=VMIN, vmax=VMAX)

plt.colorbar(image, label = "Pathlength (km)", ax=ax)

ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax.set_title(f"Pathlength Time Constant Flowfield")


image = ax2.pcolormesh(longitude, latitude, grid_map_path_dep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap)

image = ax2.pcolormesh(longitude, latitude, grid_map_path_dep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap='inferno', vmin=VMIN, vmax=VMAX)

plt.colorbar(image, label = "Pathlength (km)", ax=ax2)

ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax2.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax2.set_title(f"Pathlength Time Varying Flowfield")






# pdf.savefig()
# flip longitude to the astro convention
image = ax3.pcolormesh(longitude, latitude, grid_map_pathlen,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        norm=colors.CenteredNorm(),
                        cmap=cmap)

image = ax3.pcolormesh(longitude, latitude, grid_map_pathlen,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        norm=colors.CenteredNorm(),
                        cmap='seismic')

plt.colorbar(image, label = "$\delta$Pathlength (km)", ax=ax3)

ax3.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax3.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax3.set_title(f"$\delta$Pathlength")

plt.tight_layout(pad = 0.5)
ax.text(0.0, 0.94, 'a)', transform=ax.transAxes,
        size=20)

ax2.text(0.0, 0.94, 'b)', transform=ax2.transAxes,
size=20)

ax3.text(0.0, 0.94, 'c)', transform=ax3.transAxes,
size=20)


plt.savefig('pathlength_summary.pdf')
plt.show()
plt.close()

# print('tort difference')
fig = plt.figure(figsize=(10,13))
ax = fig.add_subplot(311,projection=ccrs.Robinson(central_longitude=120))
ax2 = fig.add_subplot(312,projection=ccrs.Robinson(central_longitude=120))
ax3 = fig.add_subplot(313,projection=ccrs.Robinson(central_longitude=120))


image = ax.pcolormesh(longitude, latitude, grid_map_tort_indep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap)

image = ax.pcolormesh(longitude, latitude, grid_map_tort_indep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap='inferno')

plt.colorbar(image, label = "Tortuosity", ax=ax)

ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax.set_title(f"Tortuosity Time Constant Flowfield")


image = ax2.pcolormesh(longitude, latitude, grid_map_tort_dep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap)

image = ax2.pcolormesh(longitude, latitude, grid_map_tort_dep,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        cmap='inferno')

plt.colorbar(image, label = "Tortuosity", ax=ax2)

ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax2.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax2.set_title(f"Tortuosity Time Varying Flowfield")






# pdf.savefig()
# flip longitude to the astro convention
image = ax3.pcolormesh(longitude, latitude, grid_map_tort,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        norm=colors.CenteredNorm(),
                        cmap=cmap)

image = ax3.pcolormesh(longitude, latitude, grid_map_tort,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), 
                        norm=colors.CenteredNorm(),
                        cmap='seismic')

plt.colorbar(image, label = "$\delta$ tortuosity (km)", ax=ax3)

ax3.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax3.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax3.set_title(f"$\delta$ tortuosity")

plt.tight_layout(pad = 0.5)
ax.text(0.0, 0.94, 'a)', transform=ax.transAxes,
        size=20)

ax2.text(0.0, 0.94, 'b)', transform=ax2.transAxes,
size=20)

ax3.text(0.0, 0.94, 'c)', transform=ax3.transAxes,
size=20)


plt.savefig('tort_summary.pdf')
plt.show()
plt.close()
