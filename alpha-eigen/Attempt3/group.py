from Attempt3.one_direction import OneDirection
from Attempt3.phi import Phi, ZoneFlux
from Attempt3.zone import Zone
from Attempt3.eigenvalue import Eigenvalue
import numpy as np


class OneGroup:
    def __init__(self, slab):
        self.slab = slab
        self.total_xs = np.zeros(self.slab.num_zones)
        self.scatter_xs = np.zeros(self.slab.num_zones)
        self.source = np.zeros(self.slab.num_zones)
        self.phi = Phi(self.slab.num_zones)
        self.angle = []
        for n in range(self.slab.num_angles):
            self.angle.append(OneDirection(number=n, slab=self.slab, group=self))

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
            change_in_flux = np.linalg.norm(self.phi.new - self.phi.old)/np.linalg.norm(self.phi.new)
            self.phi.converged = (change_in_flux < tolerance)


class MultiGroup:
    def __init__(self, slab):
        self.slab = slab
        self.phi = Phi(self.slab.num_zones, self.slab.num_groups)
        self.zone = []
        self.group = [OneGroup(self.slab)] * self.slab.num_groups

    def create_zones(self):
        zone_midpoints = np.linspace(self.slab.hx / 2, self.slab.length - self.slab.hx / 2, self.slab.num_zones)
        for i in range(self.slab.num_zones):
            self.zone.append(Zone(self.slab))
            self.zone[i].assign_material(midpoint=zone_midpoints[i])
            for g in range(self.slab.num_groups):
                self.group[g].total_xs[i] = self.zone[i].material.total_xs[g]
                self.group[g].scatter_xs[i] = self.zone[i].material.scatter_xs[g,g]

    def calculate_steady_state_scalar_flux(self):
        for i in range(self.slab.num_zones):
            # Reset source to just fission contribution so you can re-calculate with updated flux
            self.zone[i].reset_scattering_source()
            self.zone[i].compute_scattering_source()
        self.perform_source_iteration()
        for i in range(self.slab.num_zones):
            for g in range(self.slab.num_groups):
                self.phi.new[i, g] = self.group[g].phi.new[i]

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

    def perform_power_iteration(self, tolerance=1.0e-8, max_iterations=100):
        k = Eigenvalue()
        iteration = 0
        while k.converged is False:
            iteration += 1
            assert iteration < max_iterations, "Warning: source iteration did not converge"
            for i in range(self.slab.num_zones):
                self.zone[i].compute_fission_source()
            self.calculate_steady_state_scalar_flux()
            k.new = np.linalg.norm(self.phi.new) / np.linalg.norm(self.phi.old)
            k.converged = np.abs(k.new - k.old) < tolerance
        return k
