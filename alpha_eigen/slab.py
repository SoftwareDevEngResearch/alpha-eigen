import numpy as np


# there are various problem definitions within slab. The simplest form is 1D, one-speed, homogeneous slab
class Slab:
    def __init__(self, length, cells_per_mfp, num_angles):
        self.length = length
        self.num_zones = self.length*cells_per_mfp
        self.hx = self.length / self.num_zones
        self.num_angles = num_angles
        self.mu, self.weight = np.polynomial.legendre.leggauss(num_angles)
        self.num_groups = 1


class SymmetricHeterogeneousSlab(Slab):
    def __init__(self, length, cells_per_mfp, num_angles, mod_mat, fuel_mat, abs_mat):
        super().__init__(length, cells_per_mfp, num_angles)
        self.fuel_thick = 1
        self.mod_thick = 1
        self.absorber_material = 5
        self.num_regions = 5
        self.fuel_material = fuel_mat
        self.moderator_material = mod_mat
        self.absorber_material = abs_mat
        self.reflector1_right = 0
        self.fuel1_right = 0
        self.mod1_right = 0
        self.abs_right = 0
        self.mod2_right = 0
        self.fuel2_right = 0
        self.define_region_boundaries()

    def define_region_boundaries(self):
        self.fuel1_right = 1
        self.mod1_right = 2
        self.abs_right = 7
        self.mod2_right = 8
        self.fuel2_right = 9
        assert self.fuel2_right == self.length

    def assign_zone_material(self, zone):
        if zone.midpoint <= self.fuel1_right:
            zone.material = self.fuel_material
            zone.region_name = 'fuel'
        elif self.fuel1_right < zone.midpoint < self.mod1_right:
            zone.material = self.moderator_material
            zone.region_name = 'moderator'
        elif self.mod1_right <= zone.midpoint <= self.abs_right:
            zone.material = self.absorber_material
            zone.region_name = 'absorber'
        elif self.abs_right < zone.midpoint < self.mod2_right:
            zone.material = self.moderator_material
            zone.region_name = 'moderator'
        elif self.mod2_right <= zone.midpoint <= self.fuel2_right:
            zone.material = self.fuel_material
            zone.region_name = 'fuel'


class MultiRegionSlab(Slab):
    def __init__(self, length, cells_per_mfp, num_angles, reflector_thick, inner_thick, num_zones,num_groups, material1, material2):
        super.__init__(length, cells_per_mfp, num_angles)
        self.reflector_thick = reflector_thick
        self.inner_thick = inner_thick
        self.fuel_thick = (length - 2*reflector_thick - inner_thick) / 2
        self.num_zones = num_zones
        self.num_groups = num_groups
        self.metal = material1
        self.polyethylene = material2
