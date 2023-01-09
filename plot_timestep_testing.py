#!/usr/bin/env python 

import matplotlib.pyplot as plt 
import numpy as np 

time_steps = np.arange(5,30,5)
time_steps = np.append(time_steps, [50, 75, 100, 150, 200, 250, 500, 1000])

slip_systems = ['001', '010', '100']
print(time_steps)



for slip in slip_systems:
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    xis = []
    phis = []
    etas = []
    for t in time_steps:
            
        out_dir=f"output_{t}"

        results = np.loadtxt(f"{out_dir}/cij_summary_{slip}_vpsc")

        xis.append(results[:,5])
        phis.append(results[:,6])
        etas.append(results[:,7])

    xis = np.array(xis)
    phis = np.array(phis)
    etas = np.array(etas)


    # plot for path 1
    ax1.plot(time_steps, xis[:,0])
    ax1.scatter(time_steps, xis[:,0], s=20)

    ax1.plot(time_steps, phis[:,0])
    ax1.scatter(time_steps, phis[:,0], s=20)

    ax1.plot(time_steps, etas[:,0])
    ax1.scatter(time_steps, etas[:,0], s=20)

    ax1.set_title('Path 1')

    # plot for path 2
    ax2.plot(time_steps, xis[:,1])
    ax2.scatter(time_steps, xis[:,1], s=20)

    ax2.plot(time_steps, phis[:,1])
    ax2.scatter(time_steps, phis[:,1], s=20)

    ax2.plot(time_steps, etas[:,1])
    ax2.scatter(time_steps, etas[:,1], s=20)

    ax2.set_title('Path 2')


    # plot for path 3
    ax3.plot(time_steps, xis[:,2], label='xi')
    ax3.scatter(time_steps, xis[:,2], s=20)

    ax3.plot(time_steps, phis[:,2], label='phi')
    ax3.scatter(time_steps, phis[:,2], s=20)

    ax3.plot(time_steps, etas[:,2], label='eta')
    ax3.scatter(time_steps, etas[:,2], s=20)

    ax3.set_title('Path 3')

    fig.legend()
    fig.suptitle(f"Slip System {slip}")
    fig.supxlabel("Time step used (Kyr)")
    fig.supylabel("Anisotropy Strength")
    plt.tight_layout()
    plt.savefig(f"time_step_test_plot_{slip}.pdf")
    plt.show()