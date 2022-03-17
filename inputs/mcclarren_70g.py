from alpha_eigen.slab import MultiRegionSlab
from alpha_eigen.group import MultiGroup
from alpha_eigen.material import Material
from alpha_eigen.time_unit import TimeUnit


def instantiate_material_data(material1, material2):
    metal = Material(material1, 70)
    polyethylene = Material(material2, 70)
    return metal, polyethylene


def main():
    metal, polyethylene = instantiate_material_data('metal','polyethylene')

    # Slab contains problem definition
    slab = MultiRegionSlab(length=25.25, reflector_thick=1.5, inner_thick=20, num_zones=5, num_angles=2,
                           num_groups=70, material1=metal, material2=polyethylene)

    # Create one-group problem container
    multigroup = MultiGroup(slab=slab)
    multigroup.create_zones()

    num_steps = 1
    dt = 0.10
    if num_steps == 1:
        # Computing k-eigenvalue using steady-state power iteration
        k = multigroup.perform_power_iteration()
        print(k.new)
    else:
        # Run as time-dependent problem
        time_dependent = TimeUnit(slab=slab,group=multigroup,num_steps=num_steps,dt=dt)
        time_dependent.calculate_time_dependent_flux()
        alpha_eigs = time_dependent.compute_alpha_eigenvalues()
        print("alphas = ", alpha_eigs)

    angular_flux = multigroup.compute_time_dependent_flux()
    print(angular_flux)


main()
