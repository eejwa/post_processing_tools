#!/usr/bin/env python 

# code to plot the residual between two maps/output files

import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs 
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('-cij1', '--cijfile1', type=str, help='absolute path to cij_summary output file containing all anisotropy points of interest.')
parser.add_argument('-cij2', '--cijfile2', type=str, help='absolute path to cij_summary output file containing all anisotropy points of interest.')
parser.add_argument('-s', '--slip_system', type=str, help='slip system used.')
parser.add_argument('-nside', '--nside', type=int, required=False, help='healpix nside value for resolution')
parser.add_argument('-r', '--rad', required=False, help='Radius to plot at')
parser.add_argument('-o', '--outfile', type=str, help='outfile name. must end in pdf', required=True)

args = parser.parse_args()

slip_system=args.slip_system
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

# sample the flow field at grid points

lons = np.arange(-180, 185, 5)
lats = np.arange(-90, 95, 5)
grid_lons, grid_lats = np.meshgrid(lons, lats)


NPIX = hp.nside2npix(nside)
colats_hp, lons_hp = np.degrees(hp.pix2ang(nside=nside, ipix=np.arange(NPIX)))
lats_hp = 90 - colats_hp

# dep_file = f"/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/cij_summary_{slip_system}.txt"
# indep_file = f"/Users/earjwara/work/anisotropy_flow/model01/time_indep/cij_summary_{slip_system}.txt"

# path_file_dep = '/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/path_summary.txt'
# path_file_indep = '/Users/earjwara/work/anisotropy_flow/model01/time_indep/path_summary.txt'

# dep_file = f"/Users/earjwara/work/anisotropy_flow/model01/time_indep/topotaxy/summary_both.txt"
# indep_file = f"/Users/earjwara/work/anisotropy_flow/model01/time_indep/no_topo/summary_001_1-10.txt"

# path_file_dep = '/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/path_summary.txt'
# path_file_indep = '/Users/earjwara/work/anisotropy_flow/model01/time_indep/path_summary.txt'



data_array_timedep = np.loadtxt(args.cijfile1, skiprows=1)
data_array_timeindep = np.loadtxt(args.cijfile2, skiprows=1)

# path_array_timedep = np.loadtxt(path_file_dep)
# path_array_timeindep = np.loadtxt(path_file_indep)

data_array_timedep = data_array_timedep[np.lexsort((data_array_timedep[:, 1], data_array_timedep[:, 0], data_array_timedep[:, 2]))]
data_array_timeindep = data_array_timeindep[np.lexsort((data_array_timeindep[:, 1], data_array_timeindep[:, 0], data_array_timeindep[:, 2]))]
# path_array_timedep = path_array_timedep[np.lexsort((path_array_timedep[:, 1], path_array_timedep[:, 0], path_array_timedep[:, 2]))]
# path_array_timeindep = path_array_timeindep[np.lexsort((path_array_timeindep[:, 1], path_array_timeindep[:, 0], path_array_timeindep[:, 2]))]

print(data_array_timedep)
# print(path_array_timedep)


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

xi_r = xi[data_array_timedep[:,0] == int(rad)]
phi_r = phi_ani[data_array_timedep[:,0] == int(rad)]
eta_r = eta[data_array_timedep[:,0] == int(rad)]

xi_r_indep = xi_indep[data_array_timeindep[:,0] == int(rad)]
phi_r_indep = phi_ani_indep[data_array_timeindep[:,0] == int(rad)]
eta_r_indep = eta_indep[data_array_timeindep[:,0] == int(rad)]


xi_diff = np.subtract(xi_r, xi_r_indep)
phi_diff = np.subtract(phi_r, phi_r_indep)
eta_diff = np.subtract(eta_r, eta_r_indep)

# initialise maps
m_xi = np.zeros(hp.nside2npix(nside))
m_phi = np.zeros(hp.nside2npix(nside))
m_eta = np.zeros(hp.nside2npix(nside))

m_xi_indep = np.zeros(hp.nside2npix(nside))
m_xi_dep = np.zeros(hp.nside2npix(nside))
m_xi_indep[pixel_indices] = xi_r_indep
m_xi_dep[pixel_indices] = xi_r

m_phi_indep = np.zeros(hp.nside2npix(nside))
m_phi_dep = np.zeros(hp.nside2npix(nside))
m_phi_indep[pixel_indices] = phi_r_indep
m_phi_dep[pixel_indices] = phi_r




m_xi[pixel_indices] = xi_diff
m_phi[pixel_indices] = phi_diff
m_eta[pixel_indices] = eta_diff

grid_map_xi = m_xi[grid_pix]
grid_map_xi_indep = m_xi_indep[grid_pix]
grid_map_xi_dep = m_xi_dep[grid_pix]

grid_map_phi = m_phi[grid_pix]
grid_map_phi_indep = m_phi_indep[grid_pix]
grid_map_phi_dep = m_phi_dep[grid_pix]

grid_map_eta = m_eta[grid_pix]


# # plot all the maps 
# with PdfPages(args.outfile) as pdf:

#     for grid in [[grid_map_xi, 'xi'], [grid_map_phi, 'phi'], [grid_map_eta, 'eta']]:
#             print(grid[1])
#             fig = plt.figure(figsize=(13,6))
#             ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))
#             # flip longitude to the astro convention
#             image = ax.pcolormesh(longitude, latitude, grid[0],
#                                    rasterized=True,
#                                    transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
#                                    cmap='seismic')
            
#             if grid[1] == 'xi':
#                 plt.colorbar(image, label = "$\\xi$", ax=ax)
#             elif grid[1] == 'eta':
#                 plt.colorbar(image, label = "$\eta$", ax=ax)
#             elif grid[1] == 'phi':
#                 plt.colorbar(image, label = "$\phi$", ax=ax)

#             ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

#             gl = ax.gridlines(draw_labels=True, linewidth=0)

#             gl.top_label = True
#             gl.right_label = True
#             gl.bottom_label = True
#             gl.left_label = True

#             mean_ani = np.mean(grid[0])
#             ani_param = grid[1]

#             ax.set_title(f"$\\{ani_param}$ difference between time indep and dep | Radius {rad} km")

#             pdf.savefig()
#             plt.close()

# plot a difference map in the rows

# cmap = matplotlib.cm.get_cmap('seismic')
# cmap.set_bad(color='dimgray')

# fig = plt.figure(figsize=(10,13))
# ax1 = fig.add_subplot(311,projection=ccrs.Robinson(central_longitude=120))
# # flip longitude to the astro convention

# # grid_map_xi_indep = np.ma.masked_invalid(grid_map_xi_indep)
# # grid_map_xi_indep.set_cmap(cmap)
# # lc.set_norm(norm)
# np.ma.masked_invalid(np.atleast_2d(grid_map_xi_indep))

# print(grid_map_xi_indep)

# image = ax1.pcolormesh(longitude, latitude, grid_map_xi_indep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap=cmap)

# image = ax1.pcolormesh(longitude, latitude, grid_map_xi_indep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap='seismic')

# ax1.set_title(f"$\\xi$ time-constant flowfield")
# plt.colorbar(image, label = "$\\xi$", ax=ax1)


# ax2 = fig.add_subplot(312,projection=ccrs.Robinson(central_longitude=120))
# # flip longitude to the astro convention
# image = ax2.pcolormesh(longitude, latitude, grid_map_xi_dep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap=cmap)

# image = ax2.pcolormesh(longitude, latitude, grid_map_xi_dep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap='seismic')

# ax2.set_title(f"$\\xi$ time-varying flowfield")
# plt.colorbar(image, label = "$\\xi$", ax=ax2)


# ax3 = fig.add_subplot(313,projection=ccrs.Robinson(central_longitude=120))
# # flip longitude to the astro convention
# image = ax3.pcolormesh(longitude, latitude, grid_map_xi,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
#                       cmap=cmap)

# image = ax3.pcolormesh(longitude, latitude, grid_map_xi,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
#                       cmap='seismic')

# ax3.set_title(f"$\delta \ \\xi$")
# plt.colorbar(image, label = "$\delta \ \\xi$", ax=ax3)

# ax1.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
# gl = ax1.gridlines(draw_labels=True, linewidth=0)

# ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
# gl = ax2.gridlines(draw_labels=True, linewidth=0)

# ax3.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
# gl = ax3.gridlines(draw_labels=True, linewidth=0)


# ax1.text(0.0, 0.925, 'a)', transform=ax1.transAxes,
#         size=25)

# ax2.text(0.0, 0.925, 'b)', transform=ax2.transAxes,
# size=25)

# ax3.text(0.0, 0.925, 'c)', transform=ax3.transAxes,
# size=25)



# plt.tight_layout(pad = 0.5)
# plt.savefig(f'xi_comparison_{slip_system}.pdf')

# fig = plt.figure(figsize=(10,13))
# ax1 = fig.add_subplot(311,projection=ccrs.Robinson(central_longitude=120))
# # flip longitude to the astro convention
# image = ax1.pcolormesh(longitude, latitude, grid_map_phi_indep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap=cmap)

# image = ax1.pcolormesh(longitude, latitude, grid_map_phi_indep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap='seismic')

# ax1.set_title(f"$\phi$ time-constant flowfield")
# plt.colorbar(image, label = "$\phi$", ax=ax1)

# ax2 = fig.add_subplot(312,projection=ccrs.Robinson(central_longitude=120))
# # flip longitude to the astro convention
# image = ax2.pcolormesh(longitude, latitude, grid_map_phi_dep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap=cmap)

# image = ax2.pcolormesh(longitude, latitude, grid_map_phi_dep,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=0.8, vmax=1.2,
#                       cmap='seismic')

# ax2.set_title(f"$\phi$ time-varying flowfield")
# plt.colorbar(image, label = "$\phi$", ax=ax2)


# ax3 = fig.add_subplot(313,projection=ccrs.Robinson(central_longitude=120))
# # flip longitude to the astro convention
# image = ax3.pcolormesh(longitude, latitude, grid_map_phi,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
#                       cmap=cmap)


# image = ax3.pcolormesh(longitude, latitude, grid_map_phi,
#                       rasterized=True,
#                       transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
#                       cmap='seismic')

# ax3.set_title(f"$\delta \ \phi$")
# plt.colorbar(image, label = "$\delta \ \phi$", ax=ax3)

# ax1.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
# gl = ax1.gridlines(draw_labels=True, linewidth=0)

# ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
# gl = ax2.gridlines(draw_labels=True, linewidth=0)

# ax3.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
# gl = ax3.gridlines(draw_labels=True, linewidth=0)

# ax1.text(0.0, 0.925, 'a)', transform=ax1.transAxes,
#         size=25)

# ax2.text(0.0, 0.925, 'b)', transform=ax2.transAxes,
# size=25)

# ax3.text(0.0, 0.925, 'c)', transform=ax3.transAxes,
# size=25)
# plt.tight_layout(pad=0.5)

# plt.savefig(f'phi_comparison{slip_system}.pdf')


titles = {'001_1-3' : 'Hard', '001_1-5' : 'Medium','001_1-10' : 'Easy'}


cmap = matplotlib.cm.get_cmap('gist_heat_r')
cmap.set_bad(color='dimgray')

fig = plt.figure(figsize=(10,13))
ax1 = fig.add_subplot(311,projection=ccrs.Robinson(central_longitude=120))
ax2 = fig.add_subplot(312,projection=ccrs.Robinson(central_longitude=120))
ax3 = fig.add_subplot(313,projection=ccrs.Robinson(central_longitude=120))

axs = [ax1, ax2, ax3]

# slip system comparisons 

for s,ss in enumerate(['001_1-3', '001_1-5', '001_1-10']):

    ax = axs[s]

    sum_dep = f'summary_{ss}_time_dep.txt'
    sum_indep = f'summary_{ss}_time_indep.txt'
    
    data_array_timedep = np.loadtxt(sum_dep, skiprows=1)
    data_array_timeindep = np.loadtxt(sum_indep, skiprows=1)

    data_array_timedep = data_array_timedep[np.lexsort((data_array_timedep[:, 1], data_array_timedep[:, 0], data_array_timedep[:, 2]))]
    data_array_timeindep = data_array_timeindep[np.lexsort((data_array_timeindep[:, 1], data_array_timeindep[:, 0], data_array_timeindep[:, 2]))]

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

    xi_r = xi[data_array_timedep[:,0] == int(rad)]
    phi_r = phi_ani[data_array_timedep[:,0] == int(rad)]
    eta_r = eta[data_array_timedep[:,0] == int(rad)]

    xi_r_indep = xi_indep[data_array_timeindep[:,0] == int(rad)]
    phi_r_indep = phi_ani_indep[data_array_timeindep[:,0] == int(rad)]
    eta_r_indep = eta_indep[data_array_timeindep[:,0] == int(rad)]


    xi_diff = np.subtract(xi_r, xi_r_indep)
    phi_diff = np.subtract(phi_r, phi_r_indep)
    eta_diff = np.subtract(eta_r, eta_r_indep)

    # initialise maps
    m_xi = np.zeros(hp.nside2npix(nside))
    m_phi = np.zeros(hp.nside2npix(nside))
    m_eta = np.zeros(hp.nside2npix(nside))


    m_xi[pixel_indices] = xi_diff
    m_phi[pixel_indices] = phi_diff
    m_eta[pixel_indices] = eta_diff

    grid_map_xi = m_xi[grid_pix]
    grid_map_phi = m_phi[grid_pix]
    grid_map_eta = m_eta[grid_pix]




    image = ax.pcolormesh(longitude, latitude, grid_map_xi,
                      rasterized=True,
                      transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
                      cmap=cmap)

    image = ax.pcolormesh(longitude, latitude, grid_map_xi,
                        rasterized=True,
                        transform=ccrs.PlateCarree(), vmin=-0.15, vmax=0.15,
                        cmap='seismic')

    ax.set_title(titles[ss], fontsize=16)
    plt.colorbar(image, label = "$\delta \ \\xi$", ax=ax)



ax1.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
gl = ax1.gridlines(draw_labels=True, linewidth=0)

ax2.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
gl = ax2.gridlines(draw_labels=True, linewidth=0)

ax3.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)
gl = ax3.gridlines(draw_labels=True, linewidth=0)

ax1.text(0.0, 0.925, 'a)', transform=ax1.transAxes,
        size=25)

ax2.text(0.0, 0.925, 'b)', transform=ax2.transAxes,
size=25)

ax3.text(0.0, 0.925, 'c)', transform=ax3.transAxes,
size=25)

plt.tight_layout(pad=1)

plt.savefig(f'xi_diffs_all_ss.pdf')
plt.show()