#!/usr/bin/env python 



import numpy as np
import healpy as hp 
import cartopy 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import glob 


path_dir_dep = '/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/_P_files/'
path_dir_indep = '/Users/earjwara/work/anisotropy_flow/model01/time_indep/_P_files/'

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
dists = []

for pathfile in glob.glob(str(path_dir_dep)+'*'):
    pathfile_indep = path_dir_indep + pathfile.split('/')[-1]
    a = np.loadtxt(pathfile, skiprows=1)
    b = np.loadtxt(pathfile_indep, skiprows=1)


    lat  = a[-1,1]
    lon = a[-1,2]
    r = a[-1,0]

    lats.append(lat)
    lons.append(lon)

    xyz_dep = a[0,3:6]
    xyz_indep = b[0,3:6]

    print(r, lon, lat)

    dist = np.sqrt(np.sum(np.power(np.subtract(xyz_dep, xyz_indep), 2)))
    dists.append(dist)
    print(dist)

lats = np.array(lats)
lons = np.array(lons)
dists = np.array(dists)

colat_rad = np.radians(90 - lats)
lon_rad = np.radians(lons)

pixel_indices = hp.ang2pix(nside, colat_rad, lon_rad)

m_dist = np.zeros(hp.nside2npix(nside))
m_dist[pixel_indices] = dists

grid_map_dist = m_dist[grid_pix]




fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))
# flip longitude to the astro convention
image = ax.pcolormesh(longitude, latitude, grid_map_dist,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), vmin=0, vmax=dists.max(),
                        cmap='gist_heat_r')

plt.colorbar(image, label = "$\delta$ distance (km)", ax=ax)

ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax.set_title(f"starting point difference between time indep and dep | Radius {rad} km")

plt.savefig('Start_point_difference.pdf')
plt.close()



