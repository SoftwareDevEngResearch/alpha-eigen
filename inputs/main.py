from classes.material import Material
from classes.slab import SymmetricHeterogeneousSlab
from classes.time_unit import TimeUnit
from classes.group import OneGroup
import numpy as np


def instantiate_material_data(material1, material2):
    metal = Material(material1,70)
    polyethylene = Material(material2,70)
    return metal, polyethylene


def main():
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


main()
