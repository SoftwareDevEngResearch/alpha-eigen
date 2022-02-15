import numpy as np

from Attempt2.classes import Eigenvalue
from Attempt2.classes import Zone


class Slab:
    def __init__(self, length, reflector_thick, inner_thick, num_zones, num_angles, num_groups):
        self.length = length
        self.outer_ref_thick = reflector_thick
        self.inner_mod_thick = inner_thick
        self.fuel_thick = None  # dependent on slab geometry
        self.num_zones = num_zones
        self.hx = self.length / self.num_zones
        self.num_angles = num_angles
        self.mu, self.weights = np.polynomial.legendre.leggauss(num_angles)
        self.num_groups = num_groups
        self.tolerance = 1.0e-8
        self.max_iterations = 100
        self.boundary_condition = 0
        self.zone = []

    def create_zones(self, metal, polyethylene):
        zone_midpoints = np.linspace(self.hx / 2, self.length - self.hx / 2, self.num_zones)
        for i in range(self.num_zones):
            self.zone.append(Zone(number=i, midpoint=zone_midpoints[i], num_groups=self.num_groups))
            self.zone[i].define_material(self, reflector_material=polyethylene, fuel_material=metal, moderator_material=polyethylene)
            self.zone[i].create_groups(self.num_angles)

    def perform_power_iteration(self, tolerance=None, max_iterations=None):
        def calculate_source_term():
            for i in range(self.num_zones):
                self.zone[i].reset_source_term()
                self.zone[i].calculate_power_iteration_source()

        if tolerance is None:
            tolerance = self.tolerance
        if max_iterations is None:
            max_iterations = self.max_iterations
        k = Eigenvalue()
        iteration = 1
        while k.converged is False:
            calculate_source_term()
            self.calculate_multigroup_scalar_flux()
            k.new = self.new_generation_rate() / self.old_generation_rate()
            k.converged = (np.abs(k.new - k.old) < tolerance) or (iteration > max_iterations)
            iteration += 1
            k.update_eigenvalue()
            self.power_iteration_flux_update(k.new)
        if iteration > max_iterations:
            print('Warning: Power iteration did not converge!')
        return k.new

    def calculate_multigroup_scalar_flux(self):
        self.reset_scalar_flux()
        iteration = 1
        converged = False
        while not converged:
            for g in range(self.num_groups):
                for i in range(self.num_zones):
                    self.zone[i].calculate_one_group_source_term(g)
                self.perform_source_iteration(g)
            change_in_flux = self.calculate_change_in_flux()
            converged = (change_in_flux < self.tolerance) or (iteration > self.max_iterations)
        if iteration > self.max_iterations:
            print("Warning: Group Iterations did not converge on scalar flux")
        iteration += 1
        self.group_iteration_flux_update()
        return

    def reset_scalar_flux(self, g=None):
        for i in range(self.num_zones):
            if g is None:
                for g in range(self.num_groups):
                    self.zone[i].group[g].phi.reset_scalar_flux()
            else:
                self.zone[i].group[g].phi.reset_scalar_flux()

    def calculate_change_in_flux(self):
        old = np.zeros((self.num_zones, self.num_groups))
        new = old.copy()
        for i in range(self.num_zones):
            for g in range(self.num_groups):
                old[i, g] = self.zone[i].group[g].phi.old
                new[i, g] = self.zone[i].group[g].phi.new
        change = np.linalg.norm(new - old) / np.linalg.norm(new)
        return change

    def perform_source_iteration(self, g):
        iteration = 1
        converged = False
        while not converged:
            self.reset_scalar_flux(g)
            self.calculate_one_group_scalar_flux(g)

    def calculate_one_group_scalar_flux(self, g):
        for n in range(self.num_angles):
            self.zone[0].group[g].angle[n].calculate_midpoint_flux()
            self.add_weighted_angular_flux()


#    def perform_source_iteration(self, g):
#        # This is for one-group
#        converged = False
#        self.reset_angular_flux(g)
#        iteration = 1
#        while not converged:
#            self.reset_scalar_flux(g)
#            for n in range(self.num_angles):
#                self.calculate_midpoint_angular_flux(g, n)
#                self.add_weighted_angular_flux(g, n)
#
#    def calculate_midpoint_angular_flux(self, g, n):
#        self.calculate_boundary_fluxes(g, n)
#        for i in range(self.num_zones):
#            self.zone[i].group[g].angle[n].psi_midpoint = np.mean(self.zone[i].group[g].angle[n].psi_left,
#                                                                  self.zone[i].group[g].angle[n].psi_right)
#
#    def calculate_boundary_fluxes(self, g, n):
#        right_side_vector = np.zeros(self.num_zones)
#        for i in range(self.num_zones):
#            right_side_vector[i] = self.zone[i].group[g].source
#        if self.mu[n] > 0:
#            right_side_vector = np.append([self.boundary_condition], right_side_vector)
#        else:
#            right_side_vector = np.append(right_side_vector, [self.boundary_condition])
#        psi_zone_boundaries = np.linalg.solve(self.coefficient_matrix, right_side_vector)
#
#    def add_weighted_angular_flux(self, g, n):
#        for i in range(self.num_zones):
#            self.zone[i].group[g].phi.new += self.zone[i].group[g].angle[n].psi_midpoint*self.zone[i].mu[n]

    def power_iteration_flux_update(self, k):
        for i in range(self.num_zones):
            self.zone[i].power_iteration_flux_update(k)

    def group_iteration_flux_update(self):
        for i in range(self.num_zones):
            for g in range(self.num_groups):
                self.zone[i].group[g].phi.old = self.zone[i].group[g].phi.new

    def new_generation_rate(self):
        generation_rate = np.zeros((self.num_zones, self.num_groups))
        for i in range(self.num_zones):
            for g in range(self.num_groups):
                generation_rate[i, g] = self.zone[i].group[g].nu_sigmaf * self.zone[i].group[g].phi.new
        new_generation_rate = np.linalg.norm(generation_rate)
        return new_generation_rate

    def old_generation_rate(self):
        generation_rate = np.zeros((self.num_zones, self.num_groups))
        for i in range(self.num_zones):
            for g in range(self.num_groups):
                generation_rate[i, g] = self.zone[i].group[g].nu_sigmaf * self.zone[i].group[g].phi.old
        old_generation_rate = np.linalg.norm(generation_rate)
        return old_generation_rate

    def reset_angular_flux(self, g):
        for i in range(self.num_zones):
            for a in range(self.num_angles):
                self.zone[i].group[g].angle[a].psi_midpoint = 0.0

#    def calculate_multigroup_steady_state_flux(self):
#        iteration = 1
#        for g in range(self.num_groups):
#            for gprime in range(self.num_groups):
#                self.group[g].source += 0.5 * (
#                            self.group[gprime].phi.updated * self.group[gprime].sigma_s[g] * (gprime != g) +
#                            self.group[g].chi * self.group[gprime].phi.updated * self.group[gprime].nu_sigmaf)
#            self.group[g].perform_source_iteration
#            change_in_flux = np.linalg.norm()


class MultiRegionSlab(Slab):
    def __init__(self, length, reflector_thick, inner_thick, num_zones, num_angles, num_groups):
        super().__init__(length, reflector_thick, inner_thick, num_zones, num_angles, num_groups)
        self.num_regions = 5
        self.geometry = ['reflector', 'fuel', 'moderator', 'fuel', 'reflector']
        self.fuel_thick = (self.length - 2 * self.outer_ref_thick - self.inner_mod_thick) / 2
        self.reflector1_right = 0
        self.fuel1_right = 0
        self.moderator_right = 0
        self.fuel2_right = 0
        self.reflector2_right = 0
        self.define_region_boundaries()

    def define_region_boundaries(self):
        self.reflector1_right = self.outer_ref_thick
        self.fuel1_right = self.reflector1_right + self.fuel_thick
        self.moderator_right = self.fuel1_right + self.inner_mod_thick
        self.fuel2_right = self.moderator_right + self.fuel_thick
        self.reflector2_right = self.fuel2_right + self.outer_ref_thick
        assert self.reflector2_right == self.length
