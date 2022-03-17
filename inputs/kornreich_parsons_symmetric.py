from classes.material import SimpleMaterial
from classes.slab import SymmetricHeterogeneousSlab
from classes.time_unit import TimeUnit
from classes.group import OneGroup


def main():
    fuel = SimpleMaterial('fuel', num_groups=1)
    moderator = SimpleMaterial('moderator', num_groups=1)
    absorber = SimpleMaterial('absorber', num_groups=1)

    # Slab contains problem definition
    slab = SymmetricHeterogeneousSlab(length=9, cells_per_mfp=50, num_angles=4, mod_mat=moderator, fuel_mat=fuel, abs_mat=absorber)

    # Create one-group problem container
    one_group = OneGroup(slab=slab)
    one_group.create_zones()

    num_steps = 1
    dt = 0.10
    if num_steps == 1:
        print("*************************====================")
        print("Initiating k-eigenvalue solver")
        print("*************************====================")
        # Compute k-eigenvalue using steady-state power iteration
        k = one_group.perform_power_iteration()
        print(k.new)
    else:
        # Run as time-dependent problem
        time_dependent = TimeUnit(slab=slab,group=one_group,num_steps=num_steps,dt=dt)
        time_dependent.calculate_time_dependent_flux()
        alpha_eigs = time_dependent.compute_alpha_eigenvalues()
        print("alphas = ", alpha_eigs)

main()
