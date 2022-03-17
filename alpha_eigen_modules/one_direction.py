import numpy as np


class OneDirection(object):
    """Holds per-angle information and performs per-angle actions.
    Knows which group it's in, and stores per-angle data across all zones. """
    def __init__(self, number, slab, group):
        self.slab = slab
        self.group = group
        self.mu = self.slab.mu[number]
        self.weight = self.slab.weight[number]
        self.boundary_condition = 0
        self.psi_zone_boundaries = np.zeros(self.slab.num_zones + 1)
        self.psi_midpoint = np.zeros((self.slab.num_zones,1))
        if self.mu > 0:
            self.create_rhs_vector = self.set_positive_boundary_condition()
        else:
            self.create_rhs_vector = self.set_negative_boundary_condition()
        i_plus_one_coefficient = self.mu / self.slab.hx + 0.5 * self.group.total_xs
        i_coefficient = self.mu / self.slab.hx - 0.5 * self.group.total_xs
        self.coefficient_matrix = None
        self.create_coefficient_matrix(i_plus_one_coefficient, i_coefficient)

    def set_positive_boundary_condition(self):
        source = self.group.source + self.group.phi.old * self.group.scatter_xs * 0.5
        right_side_vector = np.append([self.boundary_condition], [source])
        return right_side_vector

    def set_negative_boundary_condition(self):
        source = self.group.source + self.group.phi.old * self.group.scatter_xs * 0.5
        right_side_vector = np.append([source], [self.boundary_condition])
        return right_side_vector

    def create_coefficient_matrix(self, i_plus_one_coefficient, i_coefficient):
        if self.mu > 0:
            i_plus_one_coefficient = np.append([1], i_plus_one_coefficient)
        else:
            i_plus_one_coefficient = np.append(i_plus_one_coefficient, [1])
        self.coefficient_matrix = np.diag(i_plus_one_coefficient, k=0) + np.diag(i_coefficient, k=-1)

    def calculate_midpoint_flux(self):
        # This is sweep_1D in Ryan's code, which returns psi at all the midpoints
        self.calculate_zone_boundary_fluxes()
        for i in range(self.slab.num_zones):
            self.psi_midpoint = (self.psi_zone_boundaries[i] + self.psi_zone_boundaries[i + 1])/2

    def calculate_zone_boundary_fluxes(self):
        # This is the loop over zones in Ryan's sweep_1D function
        right_side_vector = self.create_rhs_vector
        self.solve_linear_system_for_boundary_flux(right_side_vector)

    def solve_linear_system_for_boundary_flux(self, right_side_vector):
        self.psi_zone_boundaries = np.linalg.solve(self.coefficient_matrix, right_side_vector)

