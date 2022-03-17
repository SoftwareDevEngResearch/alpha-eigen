import numpy as np
np.random.seed(2021)


class Phi:
    def __init__(self, num_to_create, num2=None):
        if num2 is None:
            self.new = np.random.rand(num_to_create)*2
        else:
            phi = np.random.rand(num_to_create, num2)
            self.new = phi + np.flip(phi,0)
        self.old = self.new.copy()
        self.converged = False

    def reset_scalar_flux(self):
        self.old = self.new.copy()
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
