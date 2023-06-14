#!/usr/bin/env python 

# code to plot the residual between two maps/output files

import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

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
