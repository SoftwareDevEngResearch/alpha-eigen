import numpy as np


class Angle:
    def __init__(self, mu, zone):
        self.boundary_condition = 0
        self.mu = mu
        self.psi_left = 0
        self.psi_right = 0
        self.psi_midpoint = 0
        self.zone = zone

    def calculate_midpoint_flux(self):
        self.calculate_zone_boundary_fluxes()
        for i in range(self.zone.num_zones):
            self.psi_midpoint = np.mean(self.psi_zone_boundaries[i], self.psi_zone_boundaries[i + 1])

    def calculate_zone_boundary_fluxes(self):
        right_side_vector = self.create_rhs_vector
        self.solve_linear_system_for_boundary_flux(right_side_vector)

#class Angle:
#    def __init__(self):
#        self.group = None
#        self.zone = None
#        self.mu = 0.0
#        self.weight = 0.0
#        self.boundary_condition = 0
#        self.psi_zone_boundaries = np.zeros(self.group.num_zones + 1)
#        self.psi_midpoint = np.zeros(self.group.num_zones)
#        if self.mu > 0:
#            self.create_rhs_vector = self.set_positive_boundary_condition()
#        else:
#            self.create_rhs_vector = self.set_negative_boundary_condition()
#        i_plus_one_coefficient = self.mu / self.hx + 0.5 * self.group.total_xs
#        i_coefficient = self.mu / self.hx - 0.5 * self.group.total_xs
#        self.coefficient_matrix = None
#        self.create_coefficient_matrix(i_plus_one_coefficient, i_coefficient)
#
#    def create_coefficient_matrix(self, i_plus_one_coefficient, i_coefficient):
#        if self.mu > 0:
#            i_plus_one_coefficient = np.append([1], i_plus_one_coefficient)
#        else:
#            i_plus_one_coefficient = np.append(i_plus_one_coefficient, [1])
#        self.coefficient_matrix = np.diag(i_plus_one_coefficient, k=0) + np.diag(i_coefficient, k=-1)
#
#    def calculate_midpoint_flux(self):
#        self.calculate_zone_boundary_fluxes()
#        for i in range(self.group.num_zones):
#            self.psi_midpoint = np.mean(self.psi_zone_boundaries[i], self.psi_zone_boundaries[i + 1])
#
#    def calculate_zone_boundary_fluxes(self):
#        right_side_vector = self.create_rhs_vector
#        self.solve_linear_system_for_boundary_flux(right_side_vector)
#
#    def solve_linear_system_for_boundary_flux(self, right_side_vector):
#        self.psi_zone_boundaries = np.linalg.solve(self.coefficient_matrix, right_side_vector)
#
#    def set_positive_boundary_condition(self):
#        right_side_vector = np.append([self.boundary_condition], self.group.source)
#        return right_side_vector
#
#    def set_negative_boundary_condition(self):
#        right_side_vector = np.append(self.group.source, [self.boundary_condition])
#        return right_side_vector
