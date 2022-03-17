from alpha_eigen_modules.material import SimpleMaterial
from alpha_eigen_modules.slab import SymmetricHeterogeneousSlab
from alpha_eigen_modules.time_unit import TimeUnit
from alpha_eigen_modules.group import OneGroup


def kornreich_parsons_symmetric(length, cells_per_mfp, num_angles, time_steps=1):
    """Calculates the steady-state k-eigenvalue or time-dependent alpha-eigenvalue of the 9 mfp
        heterogeneous one-speed slab in Kornreich Parsons (2005).

        This input runs a steady-state version of the heterogeneous slab problem in
        Kornreich, Drew & Parsons, D.. (2005). Time–eigenvalue calculations in multi-region
        Cartesian geometry using Green’s functions. Annals of Nuclear Energy. 32. 964-985.
        10.1016/j.anucene.2005.02.004.
        using methods described in
        Ryan G. McClarren (2019) Calculating Time Eigenvalues of the Neutron Transport
        Equation with Dynamic Mode Decomposition, Nuclear Science and Engineering, 193:8,
        854-867, DOI: 10.1080/00295639.2018.1565014
    -------------------------------------------------------------------------------------------
    @param length: int
        Length of slab in mfp
    @param cells_per_mfp: int
        Number of cells per mfp. Total zones in problem = length*cells_per_mfp
    @param num_angles: int
        Number of angles over which to perform SN sweeps for angular flux
    @param time_steps: int
        Number of time steps over which to run. If == 1, runs steady-state.
    """
    fuel = SimpleMaterial('fuel', num_groups=1)
    moderator = SimpleMaterial('moderator', num_groups=1)
    absorber = SimpleMaterial('absorber', num_groups=1)
    # Slab contains problem definition
    slab = SymmetricHeterogeneousSlab(length=length, cells_per_mfp=cells_per_mfp, num_angles=num_angles,
                                      mod_mat=moderator, fuel_mat=fuel, abs_mat=absorber)
    # Create one-group problem container
    one_group = OneGroup(slab=slab)
    one_group.create_zones()
    num_steps = time_steps
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

if __name__ == "__main__":
    kornreich_parsons_symmetric(length=9, cells_per_mfp=50, num_angles=16, time_steps=4)
