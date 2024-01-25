#!/usr/bin/env python 



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats.mstats import spearmanr
from scipy.stats import pearsonr

import pandas as pd
plt.style.use('ggplot')


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


slip_systems = ['001_1-3', '001_1-5', '001_1-10']

path_file_indep = 'path_indep.txt'
path_file_dep = 'path_dep.txt'
temp_grad_file = 'temp_grads.txt'



path_array_timedep = np.loadtxt(path_file_dep)
path_array_timeindep = np.loadtxt(path_file_indep)
t_grad_array = np.loadtxt(temp_grad_file, skiprows=1)

# sort on lat lon rad
path_array_timedep = path_array_timedep[np.lexsort((path_array_timedep[:, 1], path_array_timedep[:, 0], path_array_timedep[:, 2]))]
path_array_timeindep = path_array_timeindep[np.lexsort((path_array_timeindep[:, 1], path_array_timeindep[:, 0], path_array_timeindep[:, 2]))]
t_grad_array = t_grad_array[np.lexsort((t_grad_array[:, 1], t_grad_array[:, 0], t_grad_array[:, 2]))]
t_grad = t_grad_array[:,3]

# pathlen
pathlen_dep = path_array_timedep[path_array_timedep[:,0] == int(3530)][:,3]
pathlen_indep = path_array_timeindep[path_array_timeindep[:,0] == int(3530)][:,3]
pathlen_diff = np.subtract(pathlen_dep, pathlen_indep)

# tort
tort_dep = path_array_timedep[path_array_timedep[:,0] == int(3530)][:,4]
tort_indep = path_array_timeindep[path_array_timeindep[:,0] == int(3530)][:,4]
tort_diff = np.subtract(tort_dep, tort_indep)

# max speed
max_dep = path_array_timedep[path_array_timedep[:,0] == int(3530)][:,5]
max_indep = path_array_timeindep[path_array_timeindep[:,0] == int(3530)][:,5]
max_diff = np.subtract(max_dep, max_indep)

# mean speed

mean_dep = path_array_timedep[path_array_timedep[:,0] == int(3530)][:,7]
mean_indep = path_array_timeindep[path_array_timeindep[:,0] == int(3530)][:,7]
mean_diff = np.subtract(mean_dep, mean_indep)


temps = path_array_timeindep[:,-1]
pathlen_diff = pathlen_diff[~np.isnan(path_array_timedep[:,5])]
pathlen_diff[pathlen_diff == 0] = 1
tort_diff = tort_diff[~np.isnan(path_array_timedep[:,5])]
tort_diff[tort_diff == 0] = 1

for ss in slip_systems:
    fig = plt.figure(figsize=(10,10))
    print(ss)
    ss_title = ss.split('_')[1]
    # ss_array = np.loadtxt(f'l2_tensor_diff_{ss}.txt', skiprows=1, usecols = (0,1,2,3))
    ss_array = np.loadtxt(f'l2_tensor_diff_{ss_title}.txt', skiprows=1, usecols = (0,1,2,3))
    au_array = np.loadtxt(f'au_info_{ss_title}.txt', skiprows=1, usecols = (0,1,2,3,5), dtype='str')

    # ss_array = ss_array[ss_array[:,0] == 3530]
    # au_array = au_array[au_array[:,0] == 3530]

    print(ss_array.shape)
    print(path_array_timedep.shape)
    print(temps.shape)

    au_indep = au_array[au_array[:,4] == 'time-constant'][:,[0,1,2,3]].astype(float)
    au_dep = au_array[au_array[:,4] == 'time-varying'][:,[0,1,2,3]].astype(float)

    # sort to be in same order as path arrays
    ss_array = ss_array[np.lexsort((ss_array[:, 1], ss_array[:, 0], ss_array[:, 2]))]
    au_dep = au_dep[np.lexsort((au_dep[:, 1], au_dep[:, 0], au_dep[:, 2]))][:,3]
    au_indep = au_indep[np.lexsort((au_indep[:, 1], au_indep[:, 0], au_indep[:, 2]))][:,3]

    print('test array orders')
    print(ss_array[:,[0,1,2]] == path_array_timedep[:,[0,1,2]])
    print(ss_array[:,[0,1,2]])
    print(path_array_timedep[:,[0,1,2]])

    tensor_diff = np.absolute(ss_array[:,3][~np.isnan(path_array_timedep[:,5])])

    print(sum(np.isnan(tensor_diff)))
    print(path_array_timedep.shape)
    print(ss_array.shape)


    log_diff_mask = np.where(np.log10(tensor_diff) > -2)

    # tensor_diff[tensor_diff == 0] = 1


    # temperature

    print(np.isnan(tensor_diff).sum())
    print(np.isnan(tensor_diff).sum())

    corr_temp = spearmanr(temps[~np.isnan(path_array_timedep[:,5])], np.log10(tensor_diff), nan_policy='omit')
    print('corr p value', corr_temp.pvalue)

    ax = fig.add_subplot(221)
    ax.scatter(temps[~np.isnan(path_array_timedep[:,5])], np.log10(tensor_diff), alpha = 0.5,
                s = 10)
    ax.set_xlabel('Temperature / K')
    ax.set_ylabel('log( Misfit )')
    ax.set_title(f'Spearmans correlation: {corr_temp.correlation:.3f}')
    ax.text(0.025, 0.925, 'a)', transform=ax.transAxes,
            size=20)
    # plt.tight_layout()
    # plt.savefig(f'scatter_au_temp_{ss_title}.pdf')
    # plt.show()


    corr_pathlen = spearmanr(np.log10(np.absolute(pathlen_diff)), np.log10(tensor_diff), nan_policy='omit')
    print('corr p value', corr_pathlen.pvalue)
    # fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(222)

    ax.scatter(np.log10(np.absolute(pathlen_diff)), np.log10(tensor_diff), alpha = 0.5,
                s = 10)
    ax.set_xlabel("log( $\delta$pathlength / km )")
    ax.set_ylabel('log( Misfit )')
    ax.set_title(f'Spearmans correlation: {corr_pathlen.correlation:.3f}')
    ax.text(0.025, 0.925, 'b)', transform=ax.transAxes,
            size=20)
    # plt.tight_layout()
    # plt.savefig(f'scatter_au_pathlen_diff_{ss_title}.pdf')
    # plt.show()


    corr_tort = spearmanr(np.log10(np.absolute(tort_diff)), np.log10(tensor_diff), nan_policy='omit')
    print('corr p value', corr_tort.pvalue)
    ax = fig.add_subplot(223)

    ax.scatter(np.log10(np.absolute(tort_diff)), np.log10(tensor_diff), alpha = 0.5,
                s = 10)
    ax.set_xlabel("log( $\delta$tortuosity )")
    ax.set_ylabel('log( Misfit)')
    ax.set_title(f'Spearmans correlation: {corr_tort.correlation:.3f}')
    ax.text(0.025, 0.925, 'c)', transform=ax.transAxes,
            size=20)
    # plt.tight_layout()
    # plt.savefig(f'scatter_au_tort_{ss_title}.pdf')
    # plt.show()

    corr_pathlen = spearmanr(np.log10(np.absolute(pathlen_indep[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), nan_policy='omit')
    # fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(224)

#     print(np.median(tensor_diff[pathlen_indep < 10]), np.percentile(tensor_diff[pathlen_indep < 10], 25), np.percentile(tensor_diff[pathlen_indep < 10], 75))
#     print(np.median(tensor_diff[(pathlen_indep > 10) * (pathlen_indep < 100)]), np.percentile(tensor_diff[(pathlen_indep > 10) * (pathlen_indep < 100)], 25), np.percentile(tensor_diff[(pathlen_indep > 10) * (pathlen_indep < 100)], 75))
#     print(np.median(tensor_diff[(pathlen_indep > 100) * (pathlen_indep < 1000)]), np.percentile(tensor_diff[(pathlen_indep > 100) * (pathlen_indep < 1000)], 25), np.percentile(tensor_diff[(pathlen_indep > 100) * (pathlen_indep < 1000)], 75))
#     print(np.median(tensor_diff[(pathlen_indep > 1000) * (pathlen_indep < 5000)]), np.percentile(tensor_diff[(pathlen_indep > 1000) * (pathlen_indep < 5000)], 25), np.percentile(tensor_diff[(pathlen_indep > 1000) * (pathlen_indep < 5000)], 75))
#     print(np.median(tensor_diff[pathlen_indep > 5000]), np.percentile(tensor_diff[pathlen_indep > 5000], 25), np.percentile(tensor_diff[pathlen_indep > 5000], 75))

    ax.scatter(np.log10(np.absolute(pathlen_indep[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), alpha = 0.5,
                s = 10)
    ax.set_xlabel('log( Steady state flowfield pathlength / km )')
    ax.set_ylabel('log( Misfit )')
    ax.set_title(f'Spearmans correlation: {corr_pathlen.correlation:.3f}')
    ax.text(0.025, 0.925, 'd)', transform=ax.transAxes,
            size=20)
    fig.tight_layout(pad=1)
    plt.savefig(f'scatter_misfit_attributes_{ss_title}.pdf')
    plt.show()


#     corr_au_indep = spearmanr(np.log10(np.absolute(au_indep[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), nan_policy='omit')

#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)

#     ax.scatter(np.log10(np.absolute(au_indep[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), alpha = 0.5,
#                 s = 10)
#     ax.set_xlabel('log( Au (time indep) )')
#     ax.set_ylabel('log( Misfit )')
#     ax.set_title(f'Spearmans correlation: {corr_au_indep.correlation:.3f}')
# #     ax.text(0.025, 0.925, 'd)', transform=ax.transAxes,
# #             size=20)
#     fig.tight_layout(pad=1)
#     plt.savefig(f'au_indep_scatter_{ss}.pdf')

#     plt.show()


#     corr_au_dep = spearmanr(np.log10(np.absolute(au_dep[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), nan_policy='omit')

#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)

#     ax.scatter(np.log10(np.absolute(au_dep[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), alpha = 0.5,
#                 s = 10)
#     ax.set_xlabel('log( Au (time dep) )')
#     ax.set_ylabel('log( Misfit )')
#     ax.set_title(f'Spearmans correlation: {corr_au_dep.correlation:.3f}')
# #     ax.text(0.025, 0.925, 'd)', transform=ax.transAxes,
# #             size=20)
#     fig.tight_layout(pad=1)
#     plt.savefig(f'au_dep_scatter_{ss}.pdf')
#     plt.show()

#     corr_tgrad_dep = spearmanr(np.log10(np.absolute(t_grad[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), nan_policy='omit')

#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)

#     ax.scatter(np.log10(np.absolute(t_grad[~np.isnan(path_array_timedep[:,5])])), np.log10(tensor_diff), alpha = 0.5,
#                 s = 10)
#     ax.set_xlabel('log( temp grad /K/deg )')
#     ax.set_ylabel('log( Misfit )')
#     ax.set_title(f'Spearmans correlation: {corr_tgrad_dep.correlation:.3f}')
# #     ax.text(0.025, 0.925, 'd)', transform=ax.transAxes,
# #             size=20)
#     fig.tight_layout(pad=1)
#     plt.savefig(f'tempgrad_scatter_{ss}.pdf')

#     plt.show()



#     au_diff = au_indep - au_dep
#     corr_au_diff = spearmanr(np.log10(np.absolute(au_diff)), np.log10(tensor_diff), nan_policy='omit')

#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)

#     ax.scatter(np.log10(np.absolute(au_diff)), np.log10(tensor_diff), alpha = 0.5,
#                 s = 10)
#     ax.set_xlabel('log( Au (time dep) )')
#     ax.set_ylabel('log( Misfit )')
#     ax.set_title(f'Spearmans correlation: {corr_au_diff.correlation:.3f}')
#     ax.text(0.025, 0.925, 'd)', transform=ax.transAxes,
#             size=20)
#     fig.tight_layout(pad=1)
#     plt.show()




# nans_max_speed = np.isnan(max_diff)

# corr_max_speed = spearmanr(np.log10(np.absolute(max_diff[~nans_max_speed])), np.log10(tensor_diff[~nans_max_speed]), nan_policy='omit')
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)

# ax.scatter(np.log10(np.absolute(max_diff[~nans_max_speed])), np.log10(tensor_diff[~nans_max_speed]), alpha = 0.5,
#             s = 10)
# ax.set_xlabel("log( $\delta$max_velocity )")
# ax.set_ylabel('log( L2 norm )')
# ax.set_title(f'Spearmans correlation: {corr_max_speed.correlation:.3f} | slip system {ss_title}')
# plt.tight_layout()
# plt.savefig(f'scatter_au_maxvel_{ss_title}.pdf')
# plt.show()

# nans_mean_speed = np.isnan(mean_diff)

# corr_mean_speed = spearmanr(np.log10(np.absolute(mean_diff[~nans_mean_speed])), np.log10(tensor_diff[~nans_mean_speed]), nan_policy='omit')
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)

# ax.scatter(np.log10(np.absolute(mean_diff[~nans_mean_speed])), np.log10(tensor_diff[~nans_mean_speed]), alpha = 0.5,
#             s = 10)
# ax.set_xlabel("log( $\delta$mean_velocity )")
# ax.set_ylabel('log( L2 norm )')
# ax.set_title(f'Spearmans correlation: {corr_mean_speed.correlation:.3f} | slip system {ss_title}')
# plt.tight_layout()
# plt.savefig(f'scatter_au_meanvel_{ss_title}.pdf')
# plt.show()
