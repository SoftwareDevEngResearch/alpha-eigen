#!/usr/bin/python3
"""
This code is a modification of PolyPu-70g.py, which produces the results in section V.B of McClarren, 'Calculating Time Eigenvalues of the Neutron Transport Equation with Dynamic Mode Decomposition', NSE 2018
It can take a long time to run and it is expecting to be run interactively as it is set up

Modifications:
    Scalar alpha calculation added to compute_alpha
    Group collapse implemented
"""
import numpy as np
import matplotlib.pyplot as plt
from . import multigroup_sn as sn
from . import linear_alg_ksolve as la

# set graphs = 0 to do all plots, graphs = 1 to do none
# set collapsed = 0 to perform 12-group collapse using infinite flux
graphs = 1
collapsed = 0


def compute_alpha(mat_input, skip, nsteps, I, G, N, dt, type):
    # Type 1 is angular, type 2 is scalar

    it = nsteps - 1

    if type == 1:
        shape = I * G * N
    elif type == 2:
        shape = I * G

    # need to reshape matrix
    phi_mat = np.zeros((shape, nsteps))
    for i in range(nsteps):
        if type == 1:
            phi_mat[:, i] = np.reshape(mat_input[:, :, :, i], shape)
        else:
            phi_mat[:, i] = np.reshape(mat_input[:, :, i], shape)
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
    Atilde = np.dot(np.dot(np.dot(np.matrix(unew).getH(), phi_mat[:, (skip + 1):(it + 1)]), np.matrix(vnew).getH()), S)
    print("Atilde size =", Atilde.shape)
    # xnew = np.dot(Atilde,phi_mat[:,0:it])
    # print("Xnew********",xnew[:,1],"phi_mat********",phi_mat[:,1])
    [eigsN, vsN] = np.linalg.eig(Atilde)
    eigsN = (1 - 1.0 / eigsN) / dt
    return eigsN, vsN, u


# load in 70g data
metal_tmp = np.load("metal_data.npz")
metal_scat = metal_tmp['metal_scat']
metal_sig_t = metal_tmp['metal_sig_t']
metal_chi = metal_tmp['metal_chi']
metal_nu_sig_f = metal_tmp['metal_nu_sig_f']

poly_tmp = np.load("poly_data.npz")
poly_scat = poly_tmp['poly_scat']
poly_sig_t = poly_tmp['poly_sig_t']
poly_chi = poly_tmp['poly_chi']
poly_nu_sig_f = poly_tmp['poly_nu_sig_f']

inv_speed = metal_tmp['metal_inv_speed']
group_edges = metal_tmp['metal_group_edges']

group_des = -np.diff(group_edges)
group_centers = (group_edges[0:-1] + group_edges[1:]) * 0.5
G = group_centers.size

# coarse group data
pu12_tmp = np.load("Pu12_data.npz")
coarse_edges = pu12_tmp['Pu_group_edges']
n = 12

# collapse with infinite flux
if collapsed == 0:
    print("***** Collapsing into", n, "groups *****")
    # calculate infinite homogeneous flux using 70 groups and collapse metal
    phi_inf = matrixSolve(metal_scat, metal_sig_t, metal_chi, metal_nu_sig_f, G)
    metal_scat, metal_sig_t, metal_chi, metal_nu_sig_f, inv_speed = collapse(phi_inf,
                                                                             n, group_edges, coarse_edges, G, metal_chi,
                                                                             metal_sig_t, metal_scat, metal_nu_sig_f,
                                                                             inv_speed, opt=1)

    # calculate infinite homogeneous flux using 70 groups and collapse poly
    phi_inf = matrixSolve(poly_scat, poly_sig_t, poly_chi, poly_nu_sig_f, G)
    poly_scat, poly_sig_t, poly_chi, poly_nu_sig_f = collapse(phi_inf, n, group_edges, coarse_edges, G,
                                                              poly_chi, poly_sig_t, poly_scat, poly_nu_sig_f,
                                                              np.zeros(G), opt=0)

    # re-set group variables
    G = n
    group_edges = coarse_edges
    group_des = -np.diff(group_edges)
    group_centers = (group_edges[0:-1] + group_edges[1:]) * 0.5

# set necessary parameters
inner_thick = 20
I = 400  # number of cells
L = 25.25  # slab length
Lx = L
hx = L / I
ref_thick = 3  # reflector thickness
# Ryan - k = 0.995432853326 with ref_thick = 8.5 and L = 11 and I = 200
q = np.ones((I, G)) * 0
Xs = np.linspace(hx / 2, L - hx / 2, I)
sigma_t = np.zeros((I, G))
nusigma_f = np.zeros((I, G))
chi = np.zeros((I, G))
sigma_s = np.zeros((I, G, G))
sigma_t[:, 0:G] = metal_sig_t
sigma_s[:, 0:G, 0:G] = metal_scat.transpose()
chi[:, 0:G] = metal_chi
nusigma_f[:, 0:G] = metal_nu_sig_f
for i in range(I):

    if ((Xs[i] + hx / 2) <= Lx / 2 + inner_thick / 2) and (Xs[i] - hx / 2 >= Lx / 2 - inner_thick / 2):
        sigma_t[i, 0:G] = poly_sig_t
        sigma_s[i, 0:G, 0:G] = poly_scat.transpose()
        nusigma_f[i, 0:G] = poly_nu_sig_f
        chi[i, 0:G] = poly_chi

    elif (np.abs(Xs[i] + hx / 2) <= (ref_thick) / 2) or (Xs[i] - hx / 2 >= Lx - (ref_thick) / 2):
        sigma_t[i, 0:G] = poly_sig_t
        sigma_s[i, 0:G, 0:G] = poly_scat.transpose()
        chi[i, 0:G] = poly_chi
        nusigma_f[i, 0:G] = poly_nu_sig_f
N = 4  # weighting of LeGendre polynomial
MU, W = np.polynomial.legendre.leggauss(N)  # 1D array containing sample points, 1D array containing weights
BCs = np.zeros((N, G))
BCs[(N // 2):N, :] = 0.0

x, k, phi_sol, phi_thermal, phi_epithermal, phi_fast = multigroup_k(I, hx, G, sigma_t, sigma_s, nusigma_f, chi, N, BCs,
                                                                    group_edges,
                                                                    tolerance=1.0e-8, maxits=400, LOUD=1)
standard = np.load('standard.npz')

if collapsed == 0:
    print("collapsed k =", k)
else:
    print("k =", k)

if graphs == 0:
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
    plt.semilogx(group_centers, phi_sol[I // 2, :])
    plt.show()

# ***** Time Dependent Section *****

q = np.ones((I, G)) * 0
psi0 = np.zeros((I, N, G)) + 1e-12
group_mids = group_centers
dt_group = np.argmin(np.abs(group_mids - .025e-6))  # 14.1))
print("14.1 MeV is in group", dt_group)
psi0[0, MU > 0, dt_group] = 1
psi0[-1, MU < 0, dt_group] = 1
numsteps = 100
dt = 5.0e-1
x, phi, psi = multigroup_td(I, hx, G, sigma_t, sigma_s, nusigma_f, chi, inv_speed,
                            N, BCs, psi0, q, numsteps, dt, group_edges, tolerance=1.0e-8, maxits=200, LOUD=1)

plt.plot(x, phi[:, 0, -1])
plt.plot(x, phi[:, G - 1, -1])
plt.show()

# compute alpha using angular
eigs, vsN, u = compute_alpha(psi, 2, numsteps, I, G, N, dt, 1)
upsi = np.reshape(u[:, 0], (I, N, G))
u1psi = np.reshape(u[:, 1], (I, N, G))
uphi = np.zeros((I, G))
u1phi = np.zeros((I, G))
for g in range(G):
    for n in range(N):
        uphi[:, g] += W[n] * upsi[:, n, g]
        u1phi[:, g] += W[n] * u1psi[:, n, g]
totneut = np.zeros(I)
totneut1 = np.zeros(I)
for i in range(I):
    totneut[i] = np.sum(uphi[i, :] * inv_speed)
    totneut1[i] = np.sum(u1phi[i, :] * inv_speed)

# compute alpha using scalar
seigs, svsN, su = compute_alpha(phi, 2, numsteps, I, G, N, dt, 2)
suphi = np.reshape(su[:, 0], (I, G))
su1phi = np.reshape(su[:, 1], (I, G))
stotneut = np.zeros(I)
stotneut1 = np.zeros(I)
for i in range(I):
    stotneut[i] = np.sum(suphi[i, :] * inv_speed)
    stotneut1[i] = np.sum(su1phi[i, :] * inv_speed)

print("max angular alpha = ", np.max(np.real(eigs)))
print("max scalar alpha = ", np.max(np.real(seigs)))
# x,alpha,phi_mode = multigroup_alpha(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,inv_speed,np.max(np.real(eigs)), tolerance = 1.0e-8,maxits = 100, LOUD=2 )
