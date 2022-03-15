from classes.one_direction import OneDirection
from classes.phi import Phi, ZoneFlux
from classes.zone import Zone
from classes.eigenvalue import Eigenvalue
import numpy as np


class OneGroup:
    def __init__(self, slab, material):
        self.slab = slab
        self.material = material
        self.total_xs = np.zeros(self.slab.num_zones)
        self.scatter_xs = np.zeros(self.slab.num_zones)
        self.source = np.zeros(self.slab.num_zones)
        self.phi = Phi(self.slab.num_zones)
        self.angle = []
        self.zone = []
        for n in range(self.slab.num_angles):
            self.angle.append(OneDirection(number=n, slab=self.slab, group=self))

    def create_zones(self):
        zone_midpoints = np.linspace(self.slab.hx / 2, self.slab.length - self.slab.hx / 2, self.slab.num_zones)
        for i in range(self.slab.num_zones):
            self.zone.append(Zone(self.slab))
            self.zone[i].material = self.material
#            self.zone[i].assign_homog_material(midpoint=zone_midpoints[i])

    def sweep1D(self, angle, source):
        assert(np.abs(angle.mu) > 1e-10)
        psi = np.zeros(self.slab.num_zones)
        ihx = 1/self.slab.hx
        if angle.mu > 0:
            psi_left = 0
            for i in range(self.slab.num_zones):
                psi_right = (source[i] + (angle.mu*ihx - 0.5*self.zone[i].material.total_xs)*psi_left)/(0.5*self.zone[i].material.total_xs + angle.mu*ihx)
                psi[i] = 0.5*(psi_left + psi_right)
                psi_left = psi_right
        return psi

    def source_iteration(self, tolerance=1.0e-8, max_iterations=100):
        self.phi.reset_scalar_flux()
        psi_mid = np.zeros((self.slab.num_zones, self.slab.num_angles))
        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            assert iteration < max_iterations, "Warning: source iteration did not converge"
            source = self.source + self.phi.old*self.material.scatter_xs*0.5
            for n in range(self.slab.num_angles):
                psi_mid[:,n] = self.sweep1D(self.angle[n], source)
                self.phi.new += psi_mid[:,n]*self.angle[n].weight
            change = np.linalg.norm(self.phi.new - self.phi.old)/np.linalg.norm(self.phi.new)
            converged = change < tolerance

    def perform_power_iteration(self, tolerance=1.0e-8, max_iterations=100):
        k = Eigenvalue()
        iteration = 0
        while k.converged is False:
            iteration += 1
            assert iteration < max_iterations, "Warning: source iteration did not converge"
            self.source = np.zeros((self.slab.num_zones, self.slab.num_groups))
            # compute fission source
            for i in range(self.slab.num_zones):
                self.source += 0.5*self.zone[i].material.chi*self.zone[i].material.nusigma_f*self.phi.old[i]
            self.source_iteration()
            k.new = np.linalg.norm(self.phi.new) / np.linalg.norm(self.phi.old)
            k.converged = np.abs(k.new - k.old) < tolerance
            k.old = k.new
            self.phi.old = self.phi.new / k.new
        return k

    def calculate_scalar_flux(self):
        # Sweep over each direction to calculate angular flux
        for n in range(self.slab.num_angles):
            self.angle[n].calculate_midpoint_flux()
        # Sum scalar flux in each zone
        for n in range(self.slab.num_angles):
            self.phi.new += self.angle[n].psi_midpoint

    def one_group_source_iteration(self, tolerance=1.0e-8, max_iterations=100):
        self.phi.reset_scalar_flux()
        iteration = 0
        while self.phi.converged is False:
            iteration += 1
            assert iteration < max_iterations, "Warning: source iteration did not converge"

            self.phi.reset_scalar_flux()
            self.calculate_scalar_flux()
            change_in_flux = np.linalg.norm(self.phi.new - self.phi.old) / np.linalg.norm(self.phi.new)
            self.phi.converged = (change_in_flux < tolerance)

    def calculate_source_term(self, phi_old):
        for i in range(self.slab.num_zones):
            for g in range(self.slab.num_groups):
                for gprime in range(self.slab.num_groups):
                    self.Q[i, g] += 0.5 * (
                            self.zone[i].material.chi[g] * self.zone[i].material.nusigma_f[gprime] * phi_old[i, gprime])

    def calculate_steady_state_scalar_flux(self, nusigma_f, chi, tolerance=1.0e-8*0.001, maxits=100):
        phi = np.zeros((self.slab.num_zones, self.slab.num_groups))
        phi_old = phi.copy()
        converged = False
        iteration = 0
        while not converged:
            for g in range(self.slab.num_groups):
                Q = self.Q[:,g].copy()
        for i in range(self.slab.num_zones):
            # Reset source to just fission contribution so you can re-calculate with updated flux
            self.zone[i].reset_scattering_source()
            self.zone[i].compute_scattering_source()
        self.perform_source_iteration()

        for i in range(self.slab.num_zones):
            for g in range(self.slab.num_groups):
                self.phi.new[i, g] = self.group[g].phi.new[i]

    def calculate_source_term(self, phi_old):
        for i in range(self.slab.num_zones):
            for g in range(self.slab.num_groups):
                for gprime in range(self.slab.num_groups):
                    self.Q[i, g] += 0.5 * (
                            self.zone[i].material.chi[g] * self.zone[i].material.nusigma_f[gprime] * phi_old[i, gprime])

class MultiGroup:
    def __init__(self, slab):
        self.slab = slab
        self.phi = Phi(self.slab.num_zones, self.slab.num_groups)
        self.zone = []
        self.group = [OneGroup(self.slab)] * self.slab.num_groups
        self.Q = np.zeros((self.slab.num_zones, self.slab.num_groups))

    def create_zones(self):
        zone_midpoints = np.linspace(self.slab.hx / 2, self.slab.length - self.slab.hx / 2, self.slab.num_zones)
        for i in range(self.slab.num_zones):
            self.zone.append(Zone(self.slab))
            self.zone[i].assign_material(midpoint=zone_midpoints[i])
            for g in range(self.slab.num_groups):
                self.group[g].total_xs[i] = self.zone[i].material.total_xs[g]
                self.group[g].scatter_xs[i] = self.zone[i].material.scatter_xs[g, g]

    def perform_source_iteration(self):
        for g in range(self.slab.num_groups):
            for i in range(self.slab.num_zones):
                self.group[g].source[i] = self.zone[i].source[g]
            self.group[g].one_group_source_iteration()

    def return_energy_based_flux(self):
        phi = ZoneFlux(self.slab.num_zones)
        for i in range(self.slab.num_zones):
            phi.thermal[i] = np.sum(self.phi.old[i, self.zone[i].material.group_edges[0:70] <= 0.55e-6])
            phi.fast[i] = np.sum(self.phi.old[i, self.zone[i].material.group_edges[1:71] >= 1])
            phi.epithermal[i] = np.sum(self.phi.old[i, :]) - phi.thermal[i] - phi.fast[i]
        return phi

    def calculate_steady_state_scalar_flux(self, nusigma_f, chi, tolerance=1.0e-8*0.001, maxits=100):
        phi = np.zeros((self.slab.num_zones, self.slab.num_groups))
        phi_old = phi.copy()
        converged = False
        iteration = 0
#        while not converged:
#            for g in range(self.slab.num_groups):
#                Q = self.Q[:,g].copy()
#                for i in range(self.slab.num_zones):
#                    for gprime in range(self.slab.num_groups):
#
#                                                                                                               :,
#                                                                                                               gprime])
#        for i in range(self.slab.num_zones):
#            # Reset source to just fission contribution so you can re-calculate with updated flux
#            self.zone[i].reset_scattering_source()
#            self.zone[i].compute_scattering_source()
#        self.perform_source_iteration()
#
#        for i in range(self.slab.num_zones):
#            for g in range(self.slab.num_groups):
#                self.phi.new[i, g] = self.group[g].phi.new[i]
#
#    def calculate_source_term(self, phi_old):
#        for i in range(self.slab.num_zones):
#            for g in range(self.slab.num_groups):
#                for gprime in range(self.slab.num_groups):
#                    self.Q[i, g] += 0.5 * (
#                            self.zone[i].material.chi[g] * self.zone[i].material.nusigma_f[gprime] * phi_old[i, gprime])

    def perform_power_iteration(self, tolerance=1.0e-8, max_iterations=100):
        phi = np.random.rand(self.slab.num_zones, self.slab.num_groups)
        phi = phi + np.flip(phi, 0)
        phi_old = phi.copy()
        k = Eigenvalue()
        iteration = 0
        while k.converged is False:
            iteration += 1
            assert iteration < max_iterations, "Warning: source iteration did not converge"
            self.Q = np.zeros((self.slab.num_zones, self.slab.num_groups))
            self.calculate_source_term(phi_old)
            zeros = np.zeros((self.slab.num_zones, self.slab.num_groups))
            phi = self.calculate_steady_state_scalar_flux(nusigma_f=zeros, chi=zeros)
            k.new = np.linalg.norm(self.phi.new) / np.linalg.norm(self.phi.old)
            k.converged = np.abs(k.new - k.old) < tolerance
        return k

    def compute_time_dependent_flux(self):
        beep = 1
        return beep
