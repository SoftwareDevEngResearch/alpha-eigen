from Attempt2.classes import Material
from Attempt2.classes import MultiRegionSlab
from Attempt2.classes import Zone


# In a zone, each group has a scalar flux and 4 angles, each of which has an angular flux
def main():
    # Load in data
    metal = Material('metal')
    polyethylene = Material('polyethylene')

    # Slab contains problem definition
    slab = MultiRegionSlab(length=25.25, reflector_thick=1.5, inner_thick=20, num_zones=400, num_angles=4, num_groups=70)
    Zone.width = slab.hx
    Zone.mu = slab.mu
    Zone.weights = slab.weights

    # Slab is broken down into zones
    slab.create_zones(metal, polyethylene)

    # Computing k-eigenvalue using power iteration
    k = slab.perform_power_iteration()
