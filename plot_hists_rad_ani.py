#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 


df_all = pd.read_csv('summary_ani_stats_all.txt', sep = ' ', index_col = None)

df_no_eta = df_all[df_all['ani_param'] != 'eta']

# g = sns.displot(data=df_no_eta[df_no_eta['slip_system'] == '1-3'], 
#                 x = 'ani_value', row = 'ani_param', hue='flow_type', 
#                 kde=True, legend=False)


for ss in ['1-3','1-5', '1-10']:

    fig  = plt.figure(figsize=(6,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    sns.histplot(data=df_no_eta[(df_no_eta['slip_system'] == ss) & (df_no_eta['ani_param'] == 'xi')], 
                x = 'ani_value', hue='flow_type', ax = ax1, kde=True, legend=False, bins = np.arange(0.7,1.32,0.02))

    sns.histplot(data=df_no_eta[(df_no_eta['slip_system'] == ss) & (df_no_eta['ani_param'] == 'phi')], 
                x = 'ani_value', hue='flow_type', ax = ax2, kde=True, legend=False, bins = np.arange(0.7,1.32,0.02))


    ax1.set_xlabel("$\\xi$")
    ax2.set_xlabel("$\phi$")
    ax1.set_title('')
    ax2.set_title('')
    ax1.legend(labels = ['Time constant', 'Time varying'])

    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.text(0.025, 0.94, 'a)', transform=ax1.transAxes,
            size=20)

    ax2.text(0.025, 0.94, 'b)', transform=ax2.transAxes,
    size=20)


    # axes[1].legend(loc='best', labels = ['Time constant', 'Time varying'])
    plt.tight_layout(pad=1)
    plt.savefig(f'histplots_rad_ani_{ss}.pdf')
    plt.show()