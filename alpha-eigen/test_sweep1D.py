from Attempt3.one_direction import OneDirection
from Attempt3.material import Material
from Attempt3.slab import MultiRegionSlab

# Load in data
metal = Material('metal')
polyethylene = Material('polyethylene')

# Slab contains problem definition
slab = MultiRegionSlab(length=25.25, reflector_thick=1.5, inner_thick=20, num_zones=400, num_angles=4,
                       num_groups=70, material1=polyethylene, material2=metal)

angle = []
for n in range(4):
    angle.append(OneDirection(number=n, slab=slab, group=self)