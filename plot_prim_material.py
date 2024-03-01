#!/usr/bin/env python 



from terratools import terra_model as tm 
import numpy as np
import matplotlib.pyplot as plt
import earthref
import scipy.interpolate
import cartopy.crs as ccrs
import glob

plt.set_cmap('viridis')

# read in netcdf files 
m = tm.read_netcdf(glob.glob('nc*'))

# get primordial material data
prim_prop = m.get_field('c_hist')[:,:,-1]

# add to the fields 
m._fields['prim_props'] = prim_prop

# add to scalar fields 
tm._SCALAR_FIELDS['prim_props'] = "Proportion of primordial material [unitless]"

# plot at 2800 km depth - maybe improve this
m.plot_layer('prim_props', 2800, depth=True, delta=1)
plt.savefig('primordial_prop_map.pdf')
plt.show()


m.plot_layer('t', 2800, depth=True, delta=1)
plt.savefig('temp_map.pdf')

plt.show()