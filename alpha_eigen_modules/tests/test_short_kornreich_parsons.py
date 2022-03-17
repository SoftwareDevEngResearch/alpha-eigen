from .context import alpha_eigen_modules
from alpha_eigen_modules.slab import SymmetricHeterogeneousSlab
from alpha_eigen_modules.group import OneGroup
from alpha_eigen_modules.material import SimpleMaterial
""" 
This input runs a steady-state version of the heterogeneous slab problem in 
Kornreich, Drew & Parsons, D.. (2005). Time–eigenvalue calculations in multi-region 
Cartesian geometry using Green’s functions. Annals of Nuclear Energy. 32. 964-985. 
10.1016/j.anucene.2005.02.004.  
using methods described in 
Ryan G. McClarren (2019) Calculating Time Eigenvalues of the Neutron Transport 
Equation with Dynamic Mode Decomposition, Nuclear Science and Engineering, 193:8, 
854-867, DOI: 10.1080/00295639.2018.1565014
"""


def calculate_k_eigenvalue(length, cells_per_mfp, num_angles):
    fuel = SimpleMaterial('fuel', num_groups=1)
    moderator = SimpleMaterial('moderator', num_groups=1)
    absorber = SimpleMaterial('absorber', num_groups=1)

    # Slab contains problem definition
    slab = SymmetricHeterogeneousSlab(length, cells_per_mfp, num_angles, moderator, fuel_mat=fuel, abs_mat=absorber)

    # Create one-group problem container
    one_group = OneGroup(slab=slab)
    one_group.create_zones()

    num_steps = 1
    # Compute k-eigenvalue using steady-state power iteration
    k = one_group.perform_power_iteration(quiet=True)
    return k


def test_calculate_k_eigenvalue_1():
    print("*************************====================")
    print("Testing steady-state Kornreich-Parsons problem")
    print("*************************====================")
    k = calculate_k_eigenvalue(length=9, cells_per_mfp=50, num_angles=4)
    assert abs(k.old - 0.9358157950764868) < 1e-8
    k = calculate_k_eigenvalue(length=9, cells_per_mfp=100, num_angles=16)
    assert abs(k.old - 0.9885264464200719) < 1e-8

