from classes.material import Material
<<<<<<< HEAD
from classes.slab import SymmetricHeterogeneousSlab
from classes.time_unit import TimeUnit
from classes.group import OneGroup
import numpy as np


def instantiate_material_data(material1, material2):
    metal = Material(material1,70)
    polyethylene = Material(material2,70)
=======
from classes.slab import MultiRegionSlab
from classes.group import MultiGroup


def instantiate_material_data(material1, material2):
    metal = Material(material1)
    polyethylene = Material(material2)
>>>>>>> a335c77... Added time-dependent kornreich-parsons input
    return metal, polyethylene


def main():
<<<<<<< HEAD
    fuel = Material('dahl_problem', num_groups=1)
    moderator = Material('dahl_problem', num_groups=1)
    absorber = Material('dahl_problem', num_groups=1)

    fuel.scatter_xs = 0.8
    fuel.nu_sigmaf = 0.7
    fuel.chi = 1.0
    moderator.scatter_xs = 0.8
    moderator.nu_sigmaf = 0.0
    moderator.chi = 1.0
    absorber.scatter_xs = 0.1
    absorber.nu_sigmaf = 0.0
    absorber.chi = 1.0
    # Slab contains problem definition
    slab = SymmetricHeterogeneousSlab(length=9, num_zones=4950, num_angles=16, mod_mat=moderator, fuel_mat=fuel, abs_mat=absorber)

    # Create one-group problem container
    one_group = OneGroup(slab=slab)
    one_group.create_zones()

    num_steps = 1
    dt = 0.10
    if num_steps == 1:
        # Computing k-eigenvalue using steady-state power iteration
        k = one_group.perform_power_iteration()
        print(k.new)
    else:
        # Run as time-dependent problem
        time_dependent = TimeUnit(slab=slab,group=one_group,num_steps=num_steps,dt=dt)
        time_dependent.calculate_time_dependent_flux()
        alpha_eigs = time_dependent.compute_alpha_eigenvalues()
        print("alphas = ", alpha_eigs)
=======
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
>>>>>>> a335c77... Added time-dependent kornreich-parsons input


main()
