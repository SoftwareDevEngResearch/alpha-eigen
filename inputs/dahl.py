from classes.material import Material
from classes.slab import Slab
from classes.time_unit import TimeUnit
from classes.group import OneGroup


def instantiate_material_data(material1, material2):
    metal = Material(material1,70)
    polyethylene = Material(material2,70)
    return metal, polyethylene


def main():
    dahl = Material('dahl_problem', num_groups=1)

    # Slab contains problem definition
    slab = Slab(length=1, num_zones=100, num_angles=16)

    # Create one-group problem container
    one_group = OneGroup(slab=slab, material=dahl)
    one_group.create_zones()

    num_steps = 500
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
