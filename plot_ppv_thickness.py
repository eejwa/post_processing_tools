#!/usr/bin/env python 


import numpy as np
import matplotlib.pyplot as plt
from terratools.terra_model import load_model_from_pickle
import earthref
import scipy.interpolate
import cartopy.crs as ccrs

transform = ccrs.Robinson()

P0 = 105.7
C = 7E-3

trans_p = lambda t : C * t + P0
trans_t = lambda p : (p - P0) / C


m = load_model_from_pickle('./nc_058_adiabat.pkl')
t = m.get_field('t')

# reset values for core boundary condition
t[0,:] = t[0,:].max()

radii = m.get_radii()
pmodel = earthref.EarthModel(earthref.ak135)
press = pmodel(radii)

lons, lats = m.get_lateral_points()


t_profile = np.mean(t, axis=1)
t_maxes = np.amax(t, axis=1)
t_mins = np.amin(t, axis=1)

transts = trans_t(press)

ppv_thicknesses = []
ppv_heights = []

n_points = len(lons)
for lon, lat in zip(lons, lats):
    print(str(len(ppv_thicknesses)),'/',str(n_points))
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


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection=transform)

t = ax.pcolormesh(grid_lons, grid_lats, grid_thick, transform=ccrs.PlateCarree(), rasterized=True)
ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=0.5)
plt.colorbar(t, ax=ax, label='ppv thickness (km)', location='right', shrink=1)
ax.set_title(f'P0: {P0} | Claperon {C}')

# first subplot


# plt.savefig('ppv_thickness_model01.pdf')



# ax2 = fig.add_subplot(212, projection=transform)

# t2 = ax2.pcolormesh(grid_lons, grid_lats, grid_height, transform=ccrs.PlateCarree())
# ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=0.5)
# plt.colorbar(t2, ax=ax2, label='ppv height (km)', location='right', shrink=1)

# first subplot


plt.savefig(f'ppv_thickness_model01_{P0}_{C}.pdf')
plt.show()
