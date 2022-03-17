import numpy as np
from alpha_eigen.phi import Phi


class Zone:
    def __init__(self, slab, midpoint):
        self.slab = slab
        self.material = None
        self.midpoint = midpoint
        self.region_name = 'undefined'
        self.phi = Phi(self.slab.num_groups)
        self.source = np.zeros(self.slab.num_groups)
        self.fission_source = self.source.copy()
        self.out_group_scatter_xs = None

    def assign_material(self, midpoint):
        if midpoint <= self.slab.reflector1_right:
            self.material = self.slab.reflector_material
            self.region_name = 'reflector'
        elif self.slab.reflector1_right < midpoint < self.slab.fuel1_right:
            self.material = self.slab.fuel_material
            self.region_name = 'fuel'
        elif self.slab.fuel1_right <= midpoint <= self.slab.moderator_right:
            self.material = self.slab.moderator_material
            self.region_name = 'moderator'
        elif self.slab.moderator_right < midpoint < self.slab.fuel2_right:
            self.material = self.slab.fuel_material
            self.region_name = 'fuel'
        elif self.slab.fuel2_right <= midpoint <= self.slab.reflector2_right:
            self.material = self.slab.reflector_material
            self.region_name = 'reflector'

        # Exclude in-group scattering by zero-ing the scatter x-sec along the diagonal
        self.out_group_scatter_xs = self.material.scatter_xs.copy()
        for g in range(self.slab.num_groups):
            self.out_group_scatter_xs[g,g] = 0

    def compute_fission_source(self):
        self.fission_source[:] = 0.0
        fission_from_all_groups = self.material.nu_sigmaf * self.phi.old
        fission_from_all_groups = np.sum(fission_from_all_groups)
        self.fission_source = 0.5 * self.material.chi * fission_from_all_groups

    def reset_scattering_source(self):
        # I don't know what this needs to do yet
        self.source = self.fission_source

    def compute_scattering_source(self):
        for g_prime in range(self.slab.num_groups):
            self.source += 0.5*(self.phi.new[g_prime] * self.out_group_scatter_xs[g_prime, :]
                                + self.material.chi * self.phi.new[g_prime] * self.material.nu_sigmaf[g_prime])
