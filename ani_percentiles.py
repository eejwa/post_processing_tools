#!/usr/bin/env python 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')


rad = 3530

percentiles = [5,25,33,50,66,75,95]
percentiles = np.arange(5,100,5)


### elastic tensor comparison ###


# df_3 = pd.read_csv('l2_tensor_diff_001_1-3.txt', sep=' ', index_col=None)
# df_5 = pd.read_csv('l2_tensor_diff_001_1-5.txt', sep=' ', index_col=None)
# df_10 = pd.read_csv('l2_tensor_diff_001_1-10.txt', sep=' ', index_col=None)
# df_all = pd.read_csv('l2_tensor_diff_all.txt', sep=' ', index_col=None)

df_3 = pd.read_csv('l2_tensor_diff_001_1-3.txt', sep=' ', index_col=None)
df_5 = pd.read_csv('l2_tensor_diff_001_1-5.txt', sep=' ', index_col=None)
df_10 = pd.read_csv('l2_tensor_diff_001_1-10.txt', sep=' ', index_col=None)
df_all = pd.read_csv('l2_tensor_diff_all.txt', sep=' ', index_col=None)


df_all['slip_system'] = df_all['slip_system'].replace('001_1-3', 'Hard')
df_all['slip_system'] = df_all['slip_system'].replace('001_1-5', 'Medium')
df_all['slip_system'] = df_all['slip_system'].replace('001_1-10', 'Easy')

tensor_diffs_3 = df_3['tensor_diff'].values
tensor_diffs_5 = df_5['tensor_diff'].values
tensor_diffs_10 = df_10['tensor_diff'].values


percentile_3 = np.nanpercentile(tensor_diffs_3, percentiles)
percentile_5 = np.nanpercentile(tensor_diffs_5, percentiles)
percentile_10 = np.nanpercentile(tensor_diffs_10, percentiles)

print(percentile_3)
print(percentile_5)
print(percentile_10)


fig = plt.figure(figsize=(8,12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(percentile_10, percentiles, label = 'Easy', c='C2')
ax1.plot(percentile_5, percentiles, label = 'Medium', c='C1')
ax1.plot(percentile_3, percentiles, label = 'Hard', c='C0')

ax1.scatter(percentile_10, percentiles, s=30, c='C2')
ax1.scatter(percentile_5, percentiles, s=30 , c='C1')
ax1.scatter(percentile_3, percentiles, s=30 , c='C0')


ax1.set_ylabel('Percentile (%)')
ax1.set_xlabel('Misfit (GPa)')
# ax1.set_ylabel('$\delta$ A$^{U}$')
# ax1.legend(loc='upper right')

g = sns.histplot(data=df_all, x='tensor_diff', hue='slip_system', ax=ax2, bins=np.arange(0,455,5), 
                 fill=False, color=['C2', 'C1', 'C0'], multiple='layer', element='step', legend=False)
ax2.set_xlabel('Misfit (GPa)')
ax2.legend(title='', loc='upper right', labels=['Easy', 'Medium', 'Hard'])
# ax2.set_xlabel('$\delta$ A$^{U}$')
fig.tight_layout(pad=0.5)
ax1.text(0.001, 0.95, 'a)', transform=ax1.transAxes,
        size=20)

ax2.text(0.001, 0.95, 'b)', transform=ax2.transAxes,
size=20)
plt.savefig('misfit_tensor_percentile_hist.pdf')
plt.show()


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)

### radial anisotropy ###
for ss in ['1-10', '1-5', '1-3']:
    labels = {'1-3': 'Hard', '1-5':'Medium', '1-10':'Easy'}
    colours = {'1-3': 'C0', '1-5':'C1', '1-10':'C2'}
    data_array_timedep = np.loadtxt(f'summary_001_{ss}_time_dep.txt', skiprows=1)
    data_array_timeindep = np.loadtxt(f'summary_001_{ss}_time_indep.txt', skiprows=1)

    R, lat, lon, P, T, xi, phi_ani, eta = data_array_timedep[:,0], data_array_timedep[:,1], data_array_timedep[:,2], data_array_timedep[:,3], data_array_timedep[:,4], data_array_timedep[:,5], data_array_timedep[:,6], data_array_timedep[:,7]
    R_indep, lat_indep, lon_indep, P_indep, T_indep, xi_indep, phi_ani_indep, eta_indep = data_array_timeindep[:,0], data_array_timeindep[:,1], data_array_timeindep[:,2], data_array_timeindep[:,3], data_array_timeindep[:,4], data_array_timeindep[:,5], data_array_timeindep[:,6], data_array_timeindep[:,7]

    xi_r = xi[data_array_timedep[:,0] == int(rad)]
    phi_r = phi_ani[data_array_timedep[:,0] == int(rad)]
    eta_r = eta[data_array_timedep[:,0] == int(rad)]

    xi_r_indep = xi_indep[data_array_timeindep[:,0] == int(rad)]
    phi_r_indep = phi_ani_indep[data_array_timeindep[:,0] == int(rad)]
    eta_r_indep = eta_indep[data_array_timeindep[:,0] == int(rad)]

    

    xi_diff = np.subtract(xi_r, xi_r_indep)
    phi_diff = np.subtract(phi_r, phi_r_indep)
    eta_diff = np.subtract(eta_r, eta_r_indep)

    print(np.absolute(xi_diff))

    percentile_xi = np.nanpercentile(np.absolute(xi_diff)*100, percentiles)
    percentile_phi = np.nanpercentile(np.absolute(phi_diff)*100, percentiles)
    print(percentile_xi)
    ax1.plot(percentile_phi, percentiles, label = labels[ss], c = colours[ss])
    ax1.scatter(percentile_phi, percentiles, c = colours[ss])

ax1.set_xlabel('Percentile (%)')
ax1.set_ylabel('$\delta \phi$ (%)')
plt.legend()
fig.tight_layout(pad=0.5)
plt.savefig('phi_percentile_plot.pdf')
plt.show()

