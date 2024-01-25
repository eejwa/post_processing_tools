#!/usr/bin/env python

from tex2elas import mat2tens, tens2mat, rotT, read_cij_file
from rotODF import rotmat_from_euler
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
cij_single = read_cij_file("ppv.cij")

angles = np.arange(0,95,5)
diffs = []
for angle in angles:

    g = rotmat_from_euler(angle, 0, 0)
    tense_single = mat2tens(cij_single)


    rot_crystal = rotT(tense_single, g)
    rot_cij = tens2mat(rot_crystal)


    rot_cij = np.triu(rot_cij)
    cij_single = np.triu(cij_single)


    diff = np.sqrt(np.sum((rot_cij - cij_single) ** 2))
    diffs.append(diff)
    print(angle, diff)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(angles, diffs,'-o')
ax.set_xlabel('Rotation angle ($^{\circ}$)')
ax.set_ylabel('Misfit (GPa)')
ax.set_title('Misfit between original and rotated ppv crystal')
plt.savefig('ppv_rotation_misfit.pdf')
plt.show()

