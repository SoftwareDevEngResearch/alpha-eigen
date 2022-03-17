from alpha_eigen.group import OneGroup
import numpy as np


class TimeUnit:
    q_initial = 1

    def __init__(self, slab, group, num_steps, dt):
        self.num_steps = num_steps
        self.slab = slab
        self.group = group
        self.source = np.zeros((slab.num_zones, slab.num_angles))
        self.phi = np.zeros((slab.num_zones, num_steps))
        self.psi = np.zeros((slab.num_zones, slab.num_angles, num_steps))
        self.dt = dt

    def calculate_time_dependent_flux(self):
        for n in range(self.slab.num_angles):
            self.source[:,n] = self.group.source
        for step in range(self.num_steps):
            k = self.group.perform_power_iteration()
            self.phi[:,step] = np.reshape(self.group.phi.new,100)
            for n in range(self.slab.num_angles):
                self.psi[:,n,step] = self.group.angle[n].psi_midpoint

    def compute_alpha_eigenvalues(self, skip=2):
        it = self.num_steps-1
        # Reshape matrix from zones x angles x steps to (zones*angles) x steps
        eigenvalue_size = self.slab.num_zones*self.slab.num_angles
        reshaped_psi = np.zeros((eigenvalue_size, self.num_steps))
        for step in range(self.num_steps):
            reshaped_psi[:,step] = np.reshape(self.psi[:,:,step],eigenvalue_size)
        # Want svd of psi_(n-1), so want up to penultimate step
        [U,S,V] = np.linalg.svd(reshaped_psi[:,skip:it], full_matrices=False)

        # Make diagonal matrix of Sigmas, removing small values to avoid a divide-by-zero
        importance = 1-np.cumsum(S)/np.sum(S)
        S_reduced = S[importance > 1e-13]
        mat_size = len(S_reduced)
        S_square = np.zeros((mat_size,mat_size))
        diagonal_indices = np.diag_indices(mat_size)
        S_square[diagonal_indices] = 1/S_reduced
        # Match u and v sizes
        U = U[:,0:mat_size]
        V = V[0:mat_size,:]

        U_star = U.conj().T
        V_star = V.conj().T
        psi_star = np.dot(U_star, reshaped_psi[:,skip+1:it+1])
        psi_star_star = np.dot(psi_star, V_star)
        Atilde = np.dot(psi_star_star, S_square)
        [eigsN,vsN] = np.linalg.eig(Atilde)
        eigsN = (1-1.0/eigsN)/self.dt
        return eigsN
