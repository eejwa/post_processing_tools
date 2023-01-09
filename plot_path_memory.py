#!/usr/bin/env python 


import matplotlib.pyplot as plt 
import glob
import numpy as np 

plt.style.use('seaborn')

xi = []
phi = [] 
eta = []
ages = []
for s in glob.glob('summary_file*'):
    age = s.split('_')[2].split('.txt')[0]
    s_array = np.loadtxt(s)
    xi.append(s_array[5])
    phi.append(s_array[6])
    eta.append(s_array[7])
    ages.append(age)


print(xi)
print(phi)
print(eta)

fig = plt.figure(figsize=(10,7), tight_layout=True)
ax = fig.add_subplot(111)

# xi
ax.plot(ages, xi, label="$\\xi$")
ax.scatter(ages, xi)

# phi
ax.plot(ages, phi, label="$\phi$")
ax.scatter(ages, phi)

# eta
ax.plot(ages, eta, label="$\eta$")
ax.scatter(ages, eta)


ax.set_xlabel(f"Time being deformed by flowfield (Ma)")
ax.set_ylabel(f"Radial anisotropy strength")
ax.set_title(f"Mantle Memory Test")

plt.legend(loc='best')
plt.savefig(f"texture_memory_test.pdf")
plt.show()
