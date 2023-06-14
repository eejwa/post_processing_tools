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
import matplotlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dir_dep', '--dir_dep', type=str, help='absolute path to cij_summary output file containing all anisotropy points of interest.')
parser.add_argument('-dir_indep', '--dir_indep', type=str, help='absolute path to cij_summary output file containing all anisotropy points of interest.')
parser.add_argument('-s', '--slip_system', type=str, help='slip system used.')
parser.add_argument('-n', '--nside', type=int, required=False, help='healpix nside value for resolution')
parser.add_argument('-r', '--rad', required=False, help='Radius to plot at')
# parser.add_argument('-o', '--outfile', type=str, help='outfile name. must end in pdf', required=True)

args = parser.parse_args()

rad = args.rad
nside = args.nside
ss = args.slip_system

path_dir_dep = args.dir_dep
path_dir_indep = args.dir_indep

# ss = '010'

# path_dir_dep = '/Users/earjwara/work/anisotropy_flow/model01/time_dep/1Ma_test/_L_files/'
# path_dir_indep = '/Users/earjwara/work/anisotropy_flow/model01/time_indep/_L_files/'

def calc_uni_ani_index(cij):


    cij = np.matrix(cij)
    if np.any(cij):

        # s = np.reciprocal(cij)
        s = cij.I
        kv = ((cij[0,0] + cij[1,1] + cij[2,2]) + 2*(cij[0,1] + cij[1,2] + cij[2,0])) / 9
        kr = 1/((s[0,0] + s[1,1] + s[2,2]) + 2*(s[0,1] + s[1,2] + s[2,0]))
        gv = ((cij[0,0] + cij[1,1] + cij[2,2]) - (cij[0,1] + cij[1,2] + cij[2,0]) + 3*(cij[3,3] + cij[4,4] + cij[5,5])) / 15
        gr = 15/(4*(s[0,0] + s[1,1] + s[2,2]) - 4*(s[0,1] + s[1,2] + s[2,0]) + 3*(s[3,3] + s[4,4] + s[5,5]))

        au = 5*(gv / gr) + (kv/kr) - 6

        return au

    else:
        return np.nan


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

# outfilename = f'l2_tensor_diff_{ss}.txt'
outfilename = f'l2_tensor_diff_{ss}.txt'

outfile = open(outfilename, 'w')
outfile.write('rad lat lon tensor_diff slip_system\n')


for pathfile in glob.glob(str(path_dir_dep)+'_L_*'+ss+'*'):
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

    print(cij_dep)
    print(cij_indep)

    # au_dep = calc_uni_ani_index(cij_dep)
    # au_indep = calc_uni_ani_index(cij_indep)

    diff = np.sqrt(np.sum((cij_no_iso_dep - cij_no_iso_indep)**2))
    # diff = au_dep - au_indep

    # print('cij')
    # print(cij_dep)
    # print('iso')
    # print(iso_dep)
    # print('cij - iso')
    # print(cij_no_iso_dep)
    

    # print('cij')
    # print(cij_indep)
    # print('iso')
    # print(iso_indep)
    # print('cij - iso')
    # print(cij_no_iso_indep)

    print(diff)
    

    diffs.append(diff)
    lons.append(lon)
    lats.append(lat)
    
    outfile.write(f'{rad} {lat} {lon} {diff} {ss}\n')




lats = np.array(lats)
lons = np.array(lons)
diffs = np.array(diffs)

# vmax = np.amax((np.absolute(diffs)))
# vmin = np.amax((np.absolute(diffs))) * -1

colat_rad = np.radians(90 - lats)
lon_rad = np.radians(lons)

pixel_indices = hp.ang2pix(nside, colat_rad, lon_rad)

m_dist = np.zeros(hp.nside2npix(nside))
m_dist[pixel_indices] = diffs

grid_map_dist = m_dist[grid_pix]


grid_map_dist[grid_map_dist == 0] = np.nan

cmap = matplotlib.cm.get_cmap('gist_heat_r')
cmap.set_bad(color='dimgray')

fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111,projection=ccrs.Robinson(central_longitude=120))

# flip longitude to the astro convention

image = ax.pcolormesh(longitude, latitude, grid_map_dist,
                        rasterized=True,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap)

image = ax.pcolormesh(longitude, latitude, grid_map_dist,
                        rasterized=True,
                        transform=ccrs.PlateCarree(),
                        cmap='gist_heat_r')

plt.colorbar(image, label = "Misfit (GPa)", ax=ax)

ax.coastlines(zorder=2, resolution='50m', color='black', linewidth=1, alpha=1)

gl = ax.gridlines(draw_labels=True, linewidth=0)

gl.top_label = True
gl.right_label = True
gl.bottom_label = True
gl.left_label = True

ax.set_title(f"Misfit between elastic tensors")
fig.tight_layout(pad=0.5)
plt.savefig(f'misfit_difference_map_{ss}.pdf')
plt.show()
plt.close()









