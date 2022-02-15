import numpy as np


class Phi:
    def __init__(self, num_to_create, num2=None):
        if num2 is None:
            self.new = np.random.rand(num_to_create)*2
        else:
            self.new = np.random.rand(num_to_create, num2)*2
        self.old = self.new.copy()
        self.converged = False

    def reset_scalar_flux(self):
        self.old = self.new
        self.new[:] = 0.0 + 1e-12
        self.converged = False

    def power_iteration_flux_update(self, k):
        self.new = self.new / k


class ZoneFlux:
    def __init__(self, num_to_create):
        self.thermal = np.zeros(num_to_create)
        self.epithermal = np.zeros(num_to_create)
        self.fast = np.zeros(num_to_create)


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
