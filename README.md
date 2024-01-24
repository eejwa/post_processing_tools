# Pyflowng postprocessing tools

A personal repository with codes to process the outputs of pyflowng (email Andrew Walker for more information) mainly for my publication on the memory of lowermost mantle anisotropy. 
It would be helpful to read or have the paper open when looking at some of these codes to help figure out what they do. Here are brief descriptions of what the codes do. 

* ani_percentiles.py - plots the cumulative histogram (percentile values) for the three ease-of-texturing case outputs. 
* box_plot_radial_ani.py - ignore this file. It was the beginning of new plots. 
* compare_ani_parameters.py - produces scatter plots comparing the elastic tensor differential against model parameters such as temperature and other things like path length.
* particle_path_diff_map.py - plots two particle paths from the \_P\_ files output from pyflowng. 
* plot_hists_rad_ani.py - plots the radial anisotropy distribution for the different ease of texturing cases. 
* plot_path_memory.py - plots radial anisotropy parameters with time for a path. requires output files to be labelled with time, however. 
* plot_timestep_testing.py - another plotting file, this time for showing the effect of different time steps in the pathline tracing on final anisotropy.
* residual_ani_maps.py - after collating the output files from two runs from pyflowng, this code plots a map of radial anisotropy on the HEALPix grid. The code plots three subfigures. The radial anisotropy map of the first run (in the paper this was the time-dependent flowfield case), the radial anisotropy map of the second run (in the paper this was the time-independent flowfield case) and then the difference between the two. This has command line inputs for the two cij summary output files, nside value (see healpy documentation), slip system used (for my personal use really), and radius.
* residual_path_maps.py - takes in two path summary files (output from pyflowng), plots their path lengths and the difference between the two as three subfigures. Does the same for tortuosity.  
* residual_tensor_map.py - takes in two directories which hold the \_L\_ files from two pyflowng runs. The runs must calculate the anisotropy at the same locations. The code then takes the l2 norm between the elastic tensors at the same location between the two runs (after removing the anisotropic component). Then it plots this residual on a map on the Healpix grid.
* test_all_paths.sh - shell script to automate testing texture development along a pathline. 
* test_pathline_memory.py - code to test the texture development along a pathline. Requires pyflowng to be installed and the path to vpsc needs to be changed in the file. The code takes in the slip system being tested (again this was for my personal setup), the timestep used when tracing the particle and what point along the path you want to compare the anisotropy to. The point of this test is essentially to determine the 'memory' of the anisotropy along the path or at what point along the path that previous texturing makes no difference to the final result. 
