#!/usr/bin/env python 

import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs 
import glob 
import os
from tex2elas import read_cij_file
from decompose import bc_decomp

ss = '010'

path_dir_dep = '/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/_L_files/'
path_dir_indep = '/Users/earjwara/work/anisotropy_flow/model01/time_indep/_L_files/'

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

lats = []
lons = []
diffs = []

for pathfile in glob.glob(str(path_dir_dep)+'_L_*'):
    print(pathfile)
    pathfile_indep = path_dir_indep + pathfile.split('/')[-1]
    lat = float(os.path.basename(pathfile).split('_')[3])
    lon = float(os.path.basename(pathfile).split('_')[4])
    rad = float(os.path.basename(pathfile).split('_')[2])

    cij_dep = read_cij_file(pathfile + '/avg_cij')
    cij_indep = read_cij_file(pathfile_indep + '/avg_cij')

    cij_decomp_dep = read_cij_file(pathfile + '/avg_cij')
    cij_decomp_indep = read_cij_file(pathfile_indep + '/avg_cij')

    iso_dep = bc_decomp(cij_decomp_dep)[0]
    iso_indep = bc_decomp(cij_decomp_indep)[0]

    cij_no_iso_dep = cij_dep - iso_dep
    cij_no_iso_indep = cij_indep - iso_indep


    diff = np.sqrt(np.sum((cij_no_iso_dep - cij_no_iso_indep)**2))

    print('cij')
    print(cij_dep)
    print('iso')
    print(iso_dep)
    print('cij - iso')
    print(cij_no_iso_dep)
    

    print('cij')
    print(cij_indep)
    print('iso')
    print(iso_indep)
    print('cij - iso')
    print(cij_no_iso_indep)

    print(diff)
    

    diffs.append(diff)
    lons.append(lon)
    lats.append(lat)
    




lats = np.array(lats)
lons = np.array(lons)
diffs = np.array(diffs)

colat_rad = np.radians(90 - lats)
lon_rad = np.radians(lons)

pixel_indices = hp.ang2pix(nside, colat_rad, lon_rad)

m_dist = np.zeros(hp.nside2npix(nside))
m_dist[pixel_indices] = diffs

grid_map_dist = m_dist[grid_pix]




fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))

# flip longitude to the astro convention
image = ax.pcolormesh(longitude, latitude, grid_map_dist,
                        rasterized=True,
                        transform=ccrs.PlateCarree(),
                        cmap='gist_heat_r')

plt.colorbar(image, label = "$\delta$ anisotropy", ax=ax)

ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax.set_title(f"l2 norm between dep and indep elastic tensors | Radius {rad} km")

plt.savefig('tensor_difference_map.pdf')
plt.close()









