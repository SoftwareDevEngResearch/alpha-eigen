import classes.material as mat
import classes.slab as sl
import numpy as np


length = 1
num_zones = 10
num_angles = 4
slab = sl.Slab(length, num_zones, num_angles)
assert isinstance(slab,sl.Slab)
assert slab.length == length
assert slab.hx == length/num_zones

num_groups = 70
metal = mat.Material('metal', num_groups)
assert isinstance(metal,mat.Material)
assert np.size(metal.total_xs) == num_groups
assert np.size(metal.group_edges) == num_groups + 1

poly = mat.Material('polyethylene', num_groups)
assert isinstance(poly,mat.Material)
assert np.size(poly.total_xs) == num_groups
assert np.size(poly.group_edges) == num_groups + 1

num_groups = 1
dahl = mat.Material('dahl_problem', num_groups)
assert isinstance(dahl,mat.Material)
assert np.size(dahl.total_xs) == num_groups
assert np.size(dahl.group_edges) == num_groups + 1
