#!/usr/bin/python3
"""
This code produces the results in section V.B of McClarren, 'Calculating Time Eigenvalues of the Neutron Transport Equation with Dynamic Mode Decomposition', NSE 2018
It can take a long time to run and it is expecting to be run interactively as it is set up
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import multigroup_sn as sn
from Attempt2.classes import EigenvalueK

from Attempt2.classes import MultiRegionSlab


def compute_alpha(psi_input, skip, nsteps, I, G, N, dt):
    it = nsteps - 1

    # need to reshape matrix
    phi_mat = np.zeros((I * G * N, nsteps))
    for i in range(nsteps):
        phi_mat[:, i] = np.reshape(psi_input[:, :, :, i], I * G * N)
    [u, s, v] = np.linalg.svd(phi_mat[:, skip:it], full_matrices=False)
    print(u.shape, s.shape, v.shape)

    # make diagonal matrix
    # print("Cumulative e-val sum:", (1-np.cumsum(s)/np.sum(s)).tolist())
    spos = s[(1 - np.cumsum(s) / np.sum(s)) > 1e-13]  # [ np.abs(s) > 1.e-5]
    mat_size = np.min([I * G * N, len(spos)])
    S = np.zeros((mat_size, mat_size))

    unew = 1.0 * u[:, 0:mat_size]
    vnew = 1.0 * v[0:mat_size, :]

    S[np.diag_indices(mat_size)] = 1.0 / spos
    print(s)
    Atilde = np.dot(np.dot(np.dot(np.matrix(unew).getH(), phi_mat[:, (skip + 1):(it + 1)]), np.matrix(vnew).getH()), S)
    print("Atilde size =", Atilde.shape)
    # xnew = np.dot(Atilde,phi_mat[:,0:it])
    # print("Xnew********",xnew[:,1],"phi_mat********",phi_mat[:,1])
    [eigsN, vsN] = np.linalg.eig(Atilde)
    eigsN = (1 - 1.0 / eigsN) / dt
    return eigsN, vsN, u


class Material(object):
    def __init__(self, material_type):
        self.name = material_type
        self.filename = './data/' + material_type + '.hdf5'
        self.scat_xs = None
        self.total_xs = None
        self.chi = None
        self.nu_sigmaf = None
        self.inv_speed = None
        self.group_edges = None
        self.group_centers = None
        self.num_energy_groups = None
        self.parse_file()

    def parse_file(self):
        with h5py.File(self.filename, 'r') as hf:
            self.scat_xs = hf['scat_xs'][:].transpose()
            self.total_xs = np.array(hf['total_xs'])
            self.chi = hf['chi'][:]
            self.nu_sigmaf = hf['nusigma_f'][:]
            self.inv_speed = hf['inverse_speed'][:]
            self.group_edges = hf['group_edges'][:]
        self.group_centers = (self.group_edges[0:-1] + self.group_edges[1:]) * 0.5
        self.num_energy_groups = np.size(self.group_centers)


class Slab(object):
    def __init__(self, length, reflector_thick, inner_thick, num_zones, num_angles, num_groups):
        self.length = length
        self.outer_ref_thick = reflector_thick
        self.inner_mod_thick = inner_thick
        self.fuel_thick = None  # dependent on slab geometry
        self.num_zones = num_zones
        self.hx = self.length / self.num_zones
        self.zone_midpoints = np.linspace(self.hx / 2, self.length - self.hx / 2, self.num_zones)
        self.num_angles = num_angles
        self.mu, self.weights = np.polynomial.legendre.leggauss(num_angles)
        self.num_groups = num_groups

    def solve_k_eigenvalue_problem(self, tolerance, max_iterations, LOUD=False):
        iteration = 1
        k = EigenvalueK()
        while k.converged is False:
            self.zones[:].compute_source_term()

    def sweep_1D(self):
        psi = np.zeros(self.num_zones)
        ihx = 1 / self.hx
        if mu > 0:
            psi_left = boundary_psi
            for i in range(self.num_zones):
                psi_right = source[i] + (mu*ihx - 0.5*sigma_t[i]) * psi_left) / (0.5 * sigma_t[i] + mu * ihx)
                psi[i] = 0.5 * (psi_right + psi_left)
                psi_left = psi_right
        else:
            psi_right = boundary_psi
            for i in reversed(range(num_zones)):
                psi_left = (source[i] + (-mu * ihx - 0.5 * sigma_t[i]) * psi_right) / (0.5 * sigma_t[i] - mu * ihx)
                psi[i] = 0.5 * (psi_right + psi_left)
                psi_right = psi_left
        return psi


class Zone(object):
    def __init__(self, slab):
        self.region_name = 'undefined'
        self.material = None
        self.length = slab.hx
        self.midpoint = None
        self.group = None
        self.num_energy_groups = slab.num_groups

    def define_material(self, slab, reflector_material, fuel_material, moderator_material):
        if self.midpoint <= slab.reflector1_right:
            self.material = reflector_material
            self.region_name = 'reflector'
        elif slab.reflector1_right < self.midpoint < slab.fuel1_right:
            self.material = fuel_material
            self.region_name = 'fuel'
        elif slab.fuel1_right <= self.midpoint <= slab.moderator_right:
            self.material = moderator_material
            self.region_name = 'moderator'
        elif slab.moderator_right < self.midpoint < slab.fuel2_right:
            self.material = fuel_material
            self.region_name = 'fuel'
        elif slab.fuel2_right <= self.midpoint <= slab.reflector2_right:
            self.material = reflector_material
            self.region_name = 'reflector'

    def create_groups(self, slab):
        self.group = []
        for i in range(self.material.num_energy_groups):
            self.group.append(Group(i, self.material, slab.num_angles, slab.mu, slab.weights))

    def compute_source_term(self):
        self.group[:].source = 0
        for to_g in range(self.num_energy_groups):
            self.group[to_g].source += 0.5 * self.group[to_g].chi * np.sum(self.group[:].nu_sigmaf * self.group[:].previous_flux)


class Group:
    def __init__(self, number, material, num_angles, mu, weights):
        self.number = number
        self.scat_xs = material.scat_xs[number, :]
        self.total_xs = material.total_xs[number]
        self.chi = material.chi[number]
        self.nu_sigmaf = material.nu_sigmaf[number]
        self.inv_speed = material.inv_speed[number]
        self.edges = np.array([material.group_edges[number], material.group_edges[number+1]])
        self.center = material.group_centers[number]
        self.source = 0
        self.scalar_flux = np.random.rand()
        self.previous_flux = np.random.rand()
        self.angle = []
        for i in range(num_angles):
            self.angle.append(Angle(mu[i], weights[i]))


class Angle:
    def __init__(self, mu, weight):
        self.mu = mu
        self.weight = weight
        self.psi = 0
        self.boundary_condition = 0


def main():
    # Load in data
    metal = Material('metal')
    polyethylene = Material('polyethylene')

    # Define problem geometry
    slab = MultiRegionSlab(length=25.25, reflector_thick=1.5, inner_thick=20, num_zones=400, num_angles=4, num_groups=70)

    # Slab is broken down into zones
    zone = [Zone(slab)] * slab.num_zones
    for i in range(slab.num_zones):
        zone[i].midpoint = slab.zone_midpoints[i]
        zone[i].define_material(slab, reflector_material=polyethylene, fuel_material=metal, moderator_material=polyethylene)
        zone[i].create_groups(slab)

    x, k, phi = slab.solve_k_eigenvalue_problem(tolerance=1.0e-8, max_iterations=400)
    print("k =", k)

    plt.plot(x, phi_sol[:, 0], 'o', label="Group 1")
    plt.plot(x, phi_sol[:, 1], 'o', label="Group 2")
    plt.plot(x, phi_sol[:, -2], 'o', label="Group 11")
    plt.plot(x, phi_sol[:, -1], 'o', label="Group 12")
    plt.legend()
    plt.show()
    plt.plot(x, phi_thermal, label="Thermal")
    plt.plot(x, phi_epithermal, label="Epithermal")
    plt.plot(x, phi_fast, label="Fast")
    plt.legend()
    plt.show()
    plt.semilogx(group_centers, phi_sol[num_zones // 2, :])
    plt.show()

    source = np.ones((num_zones, num_energy_groups)) * 0
    psi0 = np.zeros((num_zones, num_sn_angles, num_energy_groups)) + 1e-12
    group_mids = group_centers
    dt_group = np.argmin(np.abs(group_mids - .025e-6))  # 14.1))
    print("14.1 MeV is in group", dt_group)
    psi0[0, MU > 0, dt_group] = 1
    psi0[-1, MU < 0, dt_group] = 1
    numsteps = 100
    dt = 5.0e-1
    x, phi, psi = sn.multigroup_td(num_zones, hx, num_energy_groups, sigma_t, (sigma_s), nusigma_f, chi, inv_speed,
                                   num_sn_angles, BCs, psi0, source, numsteps, dt, group_edges, tolerance=1.0e-8, maxits=200,
                                   LOUD=1)
    plt.plot(x, phi[:, 0, -1])
    plt.plot(x, phi[:, num_energy_groups - 1, -1])
    plt.show()

    eigs, vsN, u = compute_alpha(psi, 2, numsteps, num_zones, num_energy_groups, num_sn_angles, dt)
    upsi = np.reshape(u[:, 0], (num_zones, num_sn_angles, num_energy_groups))
    uphi = np.zeros((num_zones, num_energy_groups))
    for g in range(num_energy_groups):
        for n in range(num_sn_angles):
            uphi[:, g] += W[n] * upsi[:, n, g]
    totneut = np.zeros(num_zones)
    for i in range(num_zones):
        totneut[i] = np.sum(uphi[i, :] * inv_speed)

    eigs, vsN, u = compute_alpha(psi, 2, numsteps, num_zones, num_energy_groups, num_sn_angles, dt)
    u1psi = np.reshape(u[:, 1], (num_zones, num_sn_angles, num_energy_groups))
    u1phi = np.zeros((num_zones, num_energy_groups))
    for g in range(num_energy_groups):
        for n in range(num_sn_angles):
            u1phi[:, g] += W[n] * u1psi[:, n, g]
    totneut1 = np.zeros(num_zones)
    for i in range(num_zones):
        totneut1[i] = np.sum(u1phi[i, :] * inv_speed)
    print("alphas = ", eigs)
    x, alpha, phi_mode = sn.multigroup_alpha(num_zones, hx, num_energy_groups, sigma_t, sigma_s, nusigma_f, chi, num_sn_angles, BCs,
                                             inv_speed,
                                             np.max(np.real(eigs)), tolerance=1.0e-8, maxits=100, LOUD=2)


main()
