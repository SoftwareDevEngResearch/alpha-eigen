import numpy as np


# there are various problem definitions within slab. The simplest form is 1D, one-speed, homogeneous slab
class Slab:
    def __init__(self, length, num_zones, num_angles):
        self.length = length
        self.num_zones = num_zones
        self.hx = self.length / self.num_zones
        self.num_angles = num_angles
        self.mu, self.weight = np.polynomial.legendre.leggauss(num_angles)

# no matter what the material type is, the same info will be needed each time
