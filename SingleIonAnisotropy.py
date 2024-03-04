# EXAMPLE: 1D S = 1 AFM Chain with single-ion anisotropy
import numpy as np
from Preliminary_Functions import *

# Number of sites in the unit cell
nsites = 2;

# Coordinates of the sites in the unit cell as fractional coordinates
coords = np.array([[0, 0], [0, 0.5]])

# Local Hamiltonian --> as a list for each site

sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
sy = -1j * np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]) / np.sqrt(2)
sz = np.array([[1, 0, 0], [0, 0, 0,], [0, 0, -1]])
J = 1

Di = -0.1
siteham = np.array([
    Di * sz @ sz,
    Di * sz @ sz
])

# Intersite Hamiltonians
# list of {site1, site2, vector, operator, coupling constant}

bondop = J * (np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
intersiteham = [
    (1, 2, [0, 0], bondop),
    (1, 2, [0, -1])
]

# Starting Wavefunctions for each site as vectors
psis = [np.array([1., 0., 0]), np.array([1., 0., 0])]

# Run the mean field
# to determine the semi-classical groundstate of the system.
psis = RunMeanField(siteham, intersiteham, psis, 100, 100)
print(np.array(psis))

# Run linear flavor wave calculation
# Use ground state wavefunctions to compute the linearized Hamiltonian describing generic
# bosonic excitations above the ground state.

out = ConstructFlavorWave(coords, siteham, intersiteham, psis, kx, ky)
linearham = out[1]
print(linearham)

# Diagonalize Hamiltonian --> @DiagChol
