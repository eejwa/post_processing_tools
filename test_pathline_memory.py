#!/usr/bin/env python

# code to compare the end radial anisotropy 
# by starting from different points along the 
# line. Starts with a random texture 


import numpy as np 
from  vpsc import elastic_from_vpsc
import argparse 
import shutil 
import os 

# user input 
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--slip_system', type=str, 
                    help='slip system used.', required=True)
parser.add_argument('-l', '--gradient_file', type=str, 
                    help='file containing gradient tensor', required=True)
parser.add_argument('-ts', '--time_step', type=float, 
                    help='time step in thousands of yrs', required=True, 
                    default=25)

parser.add_argument('-t', '--test_time', type=float, 
                    help='time step to test things with in millions of years', required=True, 
                    default=5)


args = parser.parse_args()

grad_array = np.loadtxt(args.gradient_file, dtype=str, skiprows=2)


tex_file = './TEX_PH1.OUT'
vpsc_path = '/nfs/a300/eejwa/Anisotropy_Flow/vpsc/vpsc7'

steps_per_test = (args.test_time * (10**6)) / (args.time_step  * (10**3))


print(grad_array)
print(grad_array.shape)
print(steps_per_test)

vpsc_directory = str(args.gradient_file) + '_' + args.slip_system + '_vpsc'

split_filename = args.gradient_file.split('_')
rad = float(split_filename[2])
lat = float(split_filename[3])
lon = float(split_filename[4])

test_string = rad + lon + lat 
path_file = 'path_summary.txt'
path_array = np.loadtxt(path_file, skiprows=2)
path_rad_lat_lon = path_array[:,:3]

print(test_string)
print(path_rad_lat_lon)
row_index = np.where(np.sum(path_rad_lat_lon, axis=1) == test_string)[0]
path_summary_row = path_array[row_index][0]
pressure = path_summary_row[-2]
temperature = path_summary_row[-1]

restart_steps = []

steps = list(range(int(steps_per_test), int(grad_array.shape[0]), int(steps_per_test)))
# steps.append(grad_array.shape[0])


for i in steps:
    
    texture_age = (i * args.time_step)/1000
    step = int(grad_array.shape[0]) - i
    restart_steps.append(step)
    
    L_array = grad_array[step:]
    print(L_array)
    array_len = L_array.shape[0]
    with open(f"{vpsc_directory}/Lij.proc", 'w') as Wfile:
        
        Wfile.write(f"{array_len} 7 0.02 298    nsteps ictrl eqincr temp \n")
        Wfile.write("step     L11    L12    L13    L21    L22    L23    L31    L32    L33    tincr \n")
        

#    print(L_array)
    with open(f"{vpsc_directory}/Lij.proc", 'a') as Lfile:

        np.savetxt(Lfile, L_array, fmt='%s')

    # run vpsc in the directory 
    os.chdir(f"{vpsc_directory}")
    os.system(vpsc_path)

    # calculate elastic constants from the vpsc output
    elastic_from_vpsc(P=pressure, T=temperature, R=rad, lat=lat, 
                      lon=lon, phase_mode='ppv_interp', 
                      summary_file=open(f"summary_file_{texture_age}.txt", 'w'), 
                      quiet=True)


    # rename the poly_cij.out file 
    shutil.copy('poly_cij.out', f"poly_cij_{texture_age}.out")

    # rename the texture file 
    shutil.copy('./TEX_PH1.OUT', f"./TEX_PH1_{texture_age}.OUT")

    os.chdir("../")
