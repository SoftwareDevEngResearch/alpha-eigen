from classes.material import Material
from classes.slab import MultiRegionSlab
from classes.group import MultiGroup


def instantiate_material_data(material1, material2):
    metal = Material(material1)
    polyethylene = Material(material2)
    return metal, polyethylene


def main():
    metal, polyethylene = instantiate_material_data('metal','polyethylene')

    # Slab contains problem definition
    slab = MultiRegionSlab(length=25.25, reflector_thick=1.5, inner_thick=20, num_zones=5, num_angles=2,
                           num_groups=70, material1=polyethylene, material2=metal)

    # Create multigroup problem container
    multigroup = MultiGroup(slab=slab)
    multigroup.create_zones()

    # Computing k-eigenvalue using steady-state power iteration
    k = multigroup.perform_power_iteration()

    angular_flux = multigroup.compute_time_dependent_flux()

    print(k)
    print(angular_flux)


main()
