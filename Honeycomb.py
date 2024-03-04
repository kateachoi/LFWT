# EXAMPLE: 2D S = 1/2 AFM Honeycomb
import matplotlib.pyplot as plt
from Preliminary_Functions import *

# number of sites in the unit cell
nsites = 2
coords = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])

# Local Hamiltonian --> as a list for each site
# S = 1/2 (spin up or spin down)
siteham = [np.zeros((2, 2), dtype=complex) for _ in range(nsites)]

# Intersite Hamiltonians
# list of {site1, site2, vector, operator, coupling constant}

# Pauli matrices
sx = np.matrix([[0, 1], [1, 0]]) / 2
sy = np.matrix([[0, -1j], [1j, 0]]) / 2
sz = np.matrix([[1, 0], [0, -1]]) / 2
J = 1

bondop = J * (np.kron(sx, sx) + np.kron(sy, sy) + 1.01 * np.kron(sz, sz))
intersiteham = [
    (1, 2, [0, 0], bondop),
    (1, 2, [0, -1], bondop),
    (1, 2, [1, 0], bondop)
]

# Starting Wavefunctions for each site as vectors
psis = [np.array([1., 0.]), np.array([1., 2.]) / np.sqrt(5)]

# Run the mean field
# to determine the semi-classical groundstate of the system.
psis = RunMeanField(siteham, intersiteham, psis, 100, 0)
print(np.array(psis))

# Run the linear flavor wave calculation
# Use the ground state wavefunctions describing the generic bosonic excitations above the ground state.
kx = np.arange(-2 * np.pi, 2 * np.pi, np.pi / 100)
ky = np.arange(-2 * np.pi, 2 * np.pi, np.pi / 100)

out = ConstructFlavorWave(coords, siteham, intersiteham, psis, kx, ky)
linearham = out[1]
print(linearham)

# Diagonalize Hamiltonian --> internal Bogoliubov transformation (func DiagChol)
eigenvalues = []
for ky_val in ky:
    temp = []
    for kx_val in kx:
        eigenvalues_kx_ky, _ = DiagChol(linearham.subs({kx: kx_val, ky: ky_val}))
        temp.append(eigenvalues_kx_ky)
    eigenvalues.apped(temp)

eigenvalues = np.array(eigenvalues)
plt.plot(ky, eigenvalues[:, :, 2:4])
plt.xlabel('ky')
plt.ylabel('Eigenvalues')
plt.show()

# Compute a spectrum of a local particular operator --> func Intensities[]
op = [[1, sx], [2, sx]]
tab = Intensities(out[0], DiagChol(linearham.subs({kx: 0, ky: 0})), op, [0, 3, 100, 0.05])
plt.imshow(tab, extent=(0, 3, -2 * np.pi, 2 * np.pi), aspect='auto', origin='lower')
plt.xlabel('Frequency')
plt.ylabel('ky')
plt.show()