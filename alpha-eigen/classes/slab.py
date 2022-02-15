import numpy as np


class Slab:
    def __init__(self, length, reflector_thick, inner_thick, num_zones, num_angles, num_groups):
        self.length = length
        self.outer_ref_thick = reflector_thick
        self.inner_mod_thick = inner_thick
        self.fuel_thick = None  # dependent on slab geometry
        self.num_zones = num_zones
        self.hx = self.length / self.num_zones
        self.num_angles = num_angles
        self.mu, self.weight = np.polynomial.legendre.leggauss(num_angles)
        self.num_groups = num_groups


class MultiRegionSlab(Slab):
    def __init__(self, length, reflector_thick, inner_thick, num_zones, num_angles, num_groups, material1, material2):
        super().__init__(length, reflector_thick, inner_thick, num_zones, num_angles, num_groups)
        self.num_regions = 5
        self.reflector_material = material1
        self.fuel_material = material2
        self.moderator_material = material1
        self.geometry = ['reflector', 'fuel', 'moderator', 'fuel', 'reflector']
        self.fuel_thick = (self.length - 2 * self.outer_ref_thick - self.inner_mod_thick) / 2
        self.reflector1_right = 0
        self.fuel1_right = 0
        self.moderator_right = 0
        self.fuel2_right = 0
        self.reflector2_right = 0
        self.num_time_steps = 100
        self.define_region_boundaries()

    def define_region_boundaries(self):
        self.reflector1_right = self.outer_ref_thick
        self.fuel1_right = self.reflector1_right + self.fuel_thick
        self.moderator_right = self.fuel1_right + self.inner_mod_thick
        self.fuel2_right = self.moderator_right + self.fuel_thick
        self.reflector2_right = self.fuel2_right + self.outer_ref_thick
        assert self.reflector2_right == self.length

