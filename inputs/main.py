from classes.material import Material
from classes.slab import Slab
from classes.group import OneGroup


def instantiate_material_data(material1, material2):
    metal = Material(material1,70)
    polyethylene = Material(material2,70)
    return metal, polyethylene


def main():
    dahl = Material('dahl_problem', num_groups=1)

    # Slab contains problem definition
    slab = Slab(length=9, num_zones=1, num_angles=2)

    # Create one-group problem container
    one_group = OneGroup(slab=slab, material=dahl)
    one_group.create_zones()

    # Computing k-eigenvalue using steady-state power iteration
    k = one_group.perform_power_iteration()

#    angular_flux = one_group.compute_time_dependent_flux()

    print(k.new)
    print()


main()
