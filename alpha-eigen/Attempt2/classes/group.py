from Attempt2.classes.phi import Phi
from Attempt2.classes.direction import Angle


class Group:
    def __init__(self, number, num_angles, mu, zone):
        self.number = number
        self.scalar_flux = Phi()
        self.direction = []
        self.source = 0.0

        for n in range(num_angles):
            self.direction.append(Angle(mu[n], zone))

# class Group:
#     def __init__(self, number, max_iterations, scatter_xs, total_xs, chi, nu_sigmaf):
#         self.number = number
#         self.max_iterations = max_iterations
#         self.num_angles = 1
#         self.num_zones = 0
#         self.phi = Phi(self.num_zones)
#         self.source = np.zeros(self.num_zones)
#         self.tolerance = 1.0e-8
#         self.angle = [self.Angle(self)] * self.num_angles
#         self.scatter_xs = scatter_xs
#         self.total_xs = total_xs
#         self.chi = chi
#         self.nu_sigmaf = nu_sigmaf
#
#     def perform_source_iteration(self):
#         iteration = 1
#         while not self.phi.converged:
#             # Start source iteration with zero scalar flux, and set phi_old to phi_new
#             self.phi.reset_scalar_flux()
#             self.calculate_scalar_flux()
#             change_in_flux = np.linalg.norm(self.phi.updated - self.phi.previous) / np.linalg.norm(self.phi.updated)
#
#             # Break the loop if we've converged to tolerance or if we've hit the maximum number of iterations
#             self.phi.converged = (change_in_flux < self.tolerance) or (iteration > self.max_iterations)
#             iteration += 1
#
#     def calculate_scalar_flux(self):
#         # Calculate scalar flux by summing angular flux from all directions
#         for n in range(self.num_angles):
#             self.angle[n].calculate_midpoint_flux()
#             self.phi.updated += self.angle[n].psi_midpoint * self.angle[n].weight
