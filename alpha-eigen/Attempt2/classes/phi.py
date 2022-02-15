import numpy as np


class Phi:
    def __init__(self):
        self.new = 0
        self.old = 0
        self.converged = False

    def reset_scalar_flux(self):
        self.old = self.new
        self.new = 0
        self.converged = False

    def power_iteration_flux_update(self, k):
        self.new = self.new / k

# class Phi:
#     def __init__(self, num_groups):
#         self.updated = np.zeros(num_zones)
#         self.previous = None
#         self.converged = False
#         self.reset_scalar_flux()
#
#     def reset_scalar_flux(self):
#         self.previous = self.updated
#         self.updated[:] = 0.0
#         self.converged = False
