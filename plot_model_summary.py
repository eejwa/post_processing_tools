#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from terratools.terra_model import load_model_from_pickle
import earthref
import scipy.interpolate
import cartopy.crs as ccrs
import matplotlib.colors as colors
from rotODF import rotate_vector # gives lat lon rad

transform = ccrs.Robinson(central_longitude=120)

P0 = 105.7
C = 7E-3

trans_p = lambda t : C * t + P0
trans_t = lambda p : (p - P0) / C


m = load_model_from_pickle('./nc_058_adiabat.pkl')
t = m.get_field('t')
u_xyz = m.get_field('u_xyz')

t_slice = t[2]
u_slice = u_xyz[2]
u_rads = np.zeros(t_slice.shape)

# reset values for core boundary condition
t[0,:] = t[0,:].max()

radii = m.get_radii()
pmodel = earthref.EarthModel(earthref.ak135)
press = pmodel(radii)

lons, lats = m.get_lateral_points()


for i,u in enumerate(u_slice):
    lat = lats[i]
    lon = lons[i]
    u_rad = rotate_vector(u, lat, lon)[0,2] * ((60*60*365*24)*(100))
    u_rads[i] = u_rad


t_profile = np.mean(t, axis=1)
t_maxes = np.amax(t, axis=1)
t_mins = np.amin(t, axis=1)

transts = trans_t(press)

ppv_thicknesses = []
ppv_heights = []

n_points = len(lons)
for lon, lat in zip(lons, lats):

    t_profile = m.get_1d_profile(lat=lat, lon=lon, field='t')

    transps = trans_p(t_profile)

    ppv_radii = radii[np.where(press > transps)]

    if len(ppv_radii) != 0:
        ppv_thickness = ppv_radii.max() - ppv_radii.min()
        ppv_height = ppv_radii.max() - 3481

        ppv_thicknesses.append(ppv_thickness)
        ppv_heights.append(ppv_height)
    else:
        ppv_thicknesses.append(0)
        ppv_heights.append(0)

grid_lons = np.arange(-180, 180, 0.1)
grid_lats = np.arange(-90, 90, 0.1)
grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)

grid_thick = scipy.interpolate.griddata(
    (lons, lats), ppv_thicknesses, (grid_lon, grid_lat), method="nearest"
)    

grid_height = scipy.interpolate.griddata(
    (lons, lats), ppv_heights, (grid_lon, grid_lat), method="nearest"
)    

grid_t = scipy.interpolate.griddata(
    (lons, lats), t_slice, (grid_lon, grid_lat), method="nearest"
)    

grid_u_rad = scipy.interpolate.griddata(
    (lons, lats), u_rads, (grid_lon, grid_lat), method="nearest"
)    


fig = plt.figure(figsize=(8,10))

ax1 = fig.add_subplot(311, projection=transform)
ax2 = fig.add_subplot(312, projection=transform)
ax3 = fig.add_subplot(313, projection=transform)


vel = ax1.pcolormesh(grid_lons, grid_lats, grid_u_rad, transform=ccrs.PlateCarree(), rasterized=True,
                     cmap = 'RdBu_r',  norm=colors.CenteredNorm())
ax1.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=0.5)
plt.colorbar(vel, ax=ax1, label='Radial Flow Velocity (cm/year)', location='right', shrink=1)
ax1.set_title(f'Radial Flow Velocity | Radius 3530 km')

temp = ax2.pcolormesh(grid_lons, grid_lats, grid_t, transform=ccrs.PlateCarree(), rasterized=True,
                      cmap = 'gist_heat')
ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=0.5)
plt.colorbar(temp, ax=ax2, label='Temperature (K)', location='right', shrink=1)
ax2.set_title(f'Temperature | Radius 3530 km')

thickness = ax3.pcolormesh(grid_lons, grid_lats, grid_thick, transform=ccrs.PlateCarree(), rasterized=True)
ax3.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=0.5)
plt.colorbar(thickness, ax=ax3, label='ppv thickness (km)', location='right', shrink=1)
# ax3.set_title(f'P0: {P0} | Claperon {C}')
ax3.set_title(f'Post-Perovskite Thickness')

ax1.text(0.0, 0.94, 'a)', transform=ax1.transAxes,
        size=20)

ax2.text(0.0, 0.94, 'b)', transform=ax2.transAxes,
size=20)

ax3.text(0.0, 0.94, 'c)', transform=ax3.transAxes,
size=20)
plt.tight_layout(pad=0.5)

plt.savefig(f'model_summary.pdf')
plt.show()
