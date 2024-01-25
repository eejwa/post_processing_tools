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

path_names = ['_L_3530.0_-12.839588406904156_266.25', '_L_3530.0_-30.000000000000014_247.50000000000003',
              '_L_3530.0_-33.748988595888605_251.25', '_L_3530.0_-37.669886964329635_247.50000000000003',
              '_L_3530.0_-41.81031489577862_251.25']

slip_systems = ['1-10', '1-5', '1-3']

outfilename = 'memory_differences_tensor.txt'

with open(outfilename, 'w') as outfile:
    outfile.write('age tensor_diff xi_diff phi_diff slip_system _L_file \n')
    for lfile in path_names:
        for ss in slip_systems:

            avg_cij_end = read_cij_file(f'{lfile}_001_{ss}/avg_cij_')
            avg_cij_end_to_decompose = read_cij_file(f'{lfile}_001_{ss}/avg_cij_')

            iso_end = bc_decomp(avg_cij_end_to_decompose)[0]

            cij_no_iso_end = avg_cij_end - iso_end

            rad_file_end = np.loadtxt(f'{lfile}_001_{ss}/summary_file_end.txt')
            xi_end = rad_file_end[5]
            phi_end = rad_file_end[6]
            # loop over all the cij files and take the difference with the 
            # cij when all the texturising time is used. 

            # rarely get paths over 125 Ma
            # couldnt think of an easier way to do this

            fig_tensor = plt.figure(figsize=(10,4))
            ax_tensor = fig_tensor.add_subplot(111)

            fig_rad = plt.figure(figsize=(10,4))
            ax_rad = fig_rad.add_subplot(111)


            for i in range(5, 130, 5):
                if os.path.exists(f'{lfile}_001_{ss}/avg_cij_{i}'):
                    avg_cij_test = read_cij_file(f'{lfile}_001_{ss}/avg_cij_{i}')
                    avg_cij_test_to_decompose = read_cij_file(f'{lfile}_001_{ss}/avg_cij_{i}')
                    iso_test = bc_decomp(avg_cij_test_to_decompose)[0]

                    cij_no_iso_test = avg_cij_test - iso_test

                    diff = np.sqrt(np.sum((cij_no_iso_end - cij_no_iso_test)**2))

                    rad_file_test = np.loadtxt(f'{lfile}_001_{ss}/summary_file_{i}.0.txt')
                    xi_test = rad_file_test[5]
                    phi_test = rad_file_test[6]

                    xi_diff_rad = xi_end - xi_test
                    phi_diff_rad = phi_end - phi_test


                    ax_tensor.scatter(i, diff, c='C0')
                    ax_rad.scatter(i, xi_diff_rad, c='r')
                    ax_rad.scatter(i, phi_diff_rad, c='b')
                    ax_rad.legend()
                    print(i, diff, ss, lfile)

                    outfile.write(f'{i} {diff} {xi_diff_rad} {phi_diff_rad} {ss} {lfile} \n')


                else:
                    print(f'no texture at age {i}')




            # ax_rad.scatter(i, xi_diff_rad, c='r', label='xi')
            # ax_rad.scatter(i, phi_diff_rad, c='b', label='phi')


            # ax_tensor.set_title(f'{ss} | {lfile}')
            # ax_rad.set_title(f'{ss} | {lfile}')
            # plt.show()





