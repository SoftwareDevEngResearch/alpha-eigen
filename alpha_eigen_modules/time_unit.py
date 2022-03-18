from alpha_eigen_modules.group import OneGroup
import numpy as np


class TimeUnit:
    q_initial = 1

    def __init__(self, slab, group, num_steps, dt):
        self.num_steps = num_steps
        self.slab = slab
        self.group = group
        self.source = np.zeros((slab.num_zones, slab.num_angles, 1))
        self.phi = np.zeros((slab.num_zones, 1, num_steps))
        self.psi = np.zeros((slab.num_zones, slab.num_angles, 1, num_steps))
        self.dt = dt

    def calculate_time_dependent_flux(self):
        for n in range(self.slab.num_angles):
            self.source[:,n] = self.group.source[:]
        source = self.source.copy()
        for step in range(self.num_steps):
            for n in range(self.slab.num_angles):
                source[:,n,0] = self.source[:,n,0] + self.psi[:,n,0,step]*self.group.inv_speed[:,0]/self.dt
            self.calculate_one_step_flux(source, sigT=self.group.total_xs + 1/self.dt*self.group.inv_speed)
            self.phi[:,0,step] = np.reshape(self.group.phi.new, self.slab.num_zones)    # Fix broadcasting error
            for n in range(self.slab.num_angles):
                self.psi[:,n,:,step] = self.group.angle[n].psi_midpoint

    def calculate_one_step_flux(self, source, sigT, tolerance=1e-10):
        phi = np.zeros((self.slab.num_zones, self.slab.num_groups))
        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            assert iteration < 100, "Warning: time-dependent flux iteration did not converge"
            phi_old = phi.copy()
            Q = source.copy()
            for angle in range(self.slab.num_angles):
                Q[:, angle, 0] += 0.5*(self.group.chi[:,0]*phi[:,0]*self.group.nu_sigmaf[:,0])
            self.time_dependent_source_iteration(sigT, Q)
            change = np.linalg.norm(np.reshape(phi-phi_old, (self.slab.num_zones*self.slab.num_groups,1)))/np.linalg.norm(np.reshape(phi, (self.slab.num_zones*self.slab.num_groups,1)))
            converged = change < tolerance

    def time_dependent_source_iteration(self, sigT, source, tolerance=1e-10):
        phi_old = np.zeros((self.slab.num_zones,1))
        converged = False
        iteration = 0
        while not converged:
            phi = np.zeros((self.slab.num_zones,1))
            iteration += 1
            assert iteration < 100, "Warning: source iteration did not converge"
            for n in range(self.slab.num_angles):
                psi_mid = self.group.sweep1D(self.group.angle[n], source[:,n] + phi_old*self.group.scatter_xs*0.5, sigT)
                psi_mid = np.reshape(psi_mid, [self.slab.num_zones,1])
                phi += psi_mid[:]*self.group.angle[n].weight
            change = np.linalg.norm(phi - phi_old)/np.linalg.norm(phi)
            converged = change < tolerance
            phi_old = phi.copy()
        self.phi.new = phi_old

    def compute_alpha_eigenvalues(self):
        skip = int(self.num_steps/5)
        num_included_steps = int(self.num_steps - skip - self.num_steps/50)
        it = num_included_steps-1
        psi_to_include = self.psi[:,:,:,skip:(skip+num_included_steps+1)]
        # Reshape matrix from zones x angles x steps to (zones*angles) x steps
        eigenvalue_size = self.slab.num_zones*self.slab.num_angles*self.slab.num_groups
        reshaped_psi = np.zeros((eigenvalue_size, num_included_steps))
        for step in range(num_included_steps):
            reshaped_psi[:,step] = np.reshape(psi_to_include[:,:,0,step],eigenvalue_size)
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
