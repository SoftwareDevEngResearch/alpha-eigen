#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 08:59:13 2021

@author: KaylaClements
"""

import numpy as np
import matplotlib.pyplot as plt


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

def multigroup_alpha(I, hx, G, sigma_t, sigma_s, nusigma_f, chi, N, BCs, inv_speed, min_alpha, max_alpha,
                     group_edges=None, phi=np.zeros(1), tolerance=1.0e-8, maxits=100, LOUD=False, atol=1e-5):
    """Solve alpha eigenvalue problem
    Inputs:
        I:               number of zones
        hx:              size of each zone
        G:               number of groups
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        nusigma_f:       array of nu times fission cross-sections format [i,g]
        chi:             energy distribution of fission neutrons
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    if (phi.size == 1) and (I != 1):
        phi = np.random.rand(I, G)
    sigstar = sigma_t.copy()
    sigma_sstar = sigma_s.copy()
    print("min_alpha =", min_alpha)
    print("max_alpha =", max_alpha)
    # check signs
    alpha = min_alpha
    for i in range(I):
        if (alpha >= -10000):
            sigstar[i, :] = sigma_t[i, :] + alpha * inv_speed
        else:
            for g in range(G):
                sigma_sstar[i, g, g] = sigma_sstar[i, g, g] - alpha * inv_speed[g]

    x, k, phi_old, phi_thermal, phi_epithermal, phi_fast = multigroup_k(I, hx, G, sigstar, sigma_sstar, nusigma_f, chi,
                                                                        N, BCs, group_edges, phi,
                                                                        tolerance=tolerance * 100, maxits=100,
                                                                        LOUD=LOUD - 1)
    print("k at min alpha =", k)
    assert (k < 1)

    alpha = max_alpha
    for i in range(I):
        if (alpha >= -10000):
            sigstar[i, :] = sigma_t[i, :] + alpha * inv_speed
        else:
            for g in range(G):
                sigma_sstar[i, g, g] = sigma_sstar[i, g, g] - alpha * inv_speed[g]

    x, k, phi_old, phi_thermal, phi_epithermal, phi_fast = multigroup_k(I, hx, G, sigstar, sigma_sstar, nusigma_f, chi,
                                                                        N, BCs, group_edges, phi,
                                                                        tolerance=tolerance * 100, maxits=100,
                                                                        LOUD=LOUD - 1)
    print("k at max alpha =", k)
    assert (k > 1)
    converged = 0
    step = 0
    while not (converged):
        for i in range(I):
            if (alpha >= -10000):
                sigstar[i, :] = sigma_t[i, :] + alpha * inv_speed
            else:
                for g in range(G):
                    sigma_sstar[i, g, g] = sigma_sstar[i, g, g] - alpha * inv_speed[g]

        x, k, phi_old, phi_thermal, phi_epithermal, phi_fast = multigroup_k(I, hx, G, sigstar, sigma_sstar, nusigma_f,
                                                                            chi, N, BCs, group_edges, phi,
                                                                            tolerance=tolerance, maxits=100,
                                                                            LOUD=LOUD - 1)
        if (k < 1):
            min_alpha = alpha
        else:
            max_alpha = alpha
        alpha = 0.5 * (max_alpha + min_alpha)
        step += 1
        converged = math.fabs(max_alpha - min_alpha) < atol
        print("Step", step, ": alpha =", alpha, "k =", k, "[", min_alpha, ",", max_alpha, "]")
    return x, k, phi_old, alpha


generate = 'no'
# load in 70g data
if generate == 'yes':
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

    # set necessary parameters
    inner_thick = 20
    L = 25.25  # slab length
    Lx = L
    hx = L / I
    ref_thick = 3  # reflector thickness
    tol = 1e-8
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

I = 400
tol = 1e-8
# noise = np.random.uniform(-1,1,np.shape(psi))*tol
# noisyPsi = psi + noise
# noisyEig, noisyV, noisyU = compute_alpha(noisyPsi, 2, 100, 400, 70, 4, .5, 1)
# noisyEig = noisyEig[:49]
#
# plt.scatter(np.real(eigs),np.imag(eigs))
# plt.figure(2)
# plt.scatter(np.real(scalar_eigs),np.imag(scalar_eigs))
#
# plt.show()

# comparing fine and coarse group fluxes
## sPhiSol = standard['phi_sol']
## cPhiSol = collapse['phi_sol']
## sPsi = standard['psi']
## cPsi = collapse['psi']
##
## sPhiTot = np.sum(sPhiSol, axis=1)
## cPhiTot = np.sum(cPhiSol, axis=1)
## sPsiTot = np.sum(sPsi, axis=2)
## cPsiTot = np.sum(cPsi, axis=2)
##
## plt.figure(1)
## plt.plot(range(I), sPhiTot, 'b')
## plt.title('Total scalar flux summed across all 70 groups, fine run')
## plt.figure(2)
## plt.plot(range(I), cPhiTot, 'r')
## plt.title('Total scalar flux summed across all 12 groups, coarse run')
##

standard = np.load('standard.npz')
psi = standard['psi']
phi = standard['phi']
eigs = np.array([ 6.50506913e+00+2.28541234j,  6.50506913e+00-2.28541234j,
        8.21975712e-01+2.07231943j,  8.21975712e-01-2.07231943j,
       -6.03334302e+01+0.j        , -2.02798109e+00+9.00562091j,
       -2.02798109e+00-9.00562091j, -2.12700642e+00+5.39793556j,
       -2.12700642e+00-5.39793556j, -8.02852742e-01+4.22132852j,
       -8.02852742e-01-4.22132852j, -9.84054258e-01+3.02015615j,
       -9.84054258e-01-3.02015615j, -8.52473866e-01+2.46918795j,
       -8.52473866e-01-2.46918795j, -6.75398653e-01+1.94447578j,
       -6.75398653e-01-1.94447578j, -2.54419963e+00+0.j        ,
       -5.20005718e-01+1.6024207j , -5.20005718e-01-1.6024207j ,
       -3.90096298e-01+1.34509873j, -3.90096298e-01-1.34509873j,
       -2.85442969e-01+1.14978247j, -2.85442969e-01-1.14978247j,
       -2.06937839e-01+0.99740762j, -2.06937839e-01-0.99740762j,
       -1.48626106e-01+0.87769392j, -1.48626106e-01-0.87769392j,
       -9.91458428e-02+0.7774847j , -9.91458428e-02-0.7774847j ,
       -6.79066166e-02+0.68456953j, -6.79066166e-02-0.68456953j,
       -5.28175031e-02+0.6174585j , -5.28175031e-02-0.6174585j ,
       -4.49927922e-02+0.52353621j, -4.49927922e-02-0.52353621j,
       -4.12668836e-02+0.43064507j, -4.12668836e-02-0.43064507j,
       -2.00188801e-02+0.j        , -4.61472213e-02+0.05278772j,
       -4.61472213e-02-0.05278772j, -5.15486444e-02+0.11310533j,
       -5.15486444e-02-0.11310533j, -6.07259349e-02+0.19928326j,
       -6.07259349e-02-0.19928326j, -1.79365264e-01+0.28554738j,
       -1.79365264e-01-0.28554738j, -5.96392544e-02+0.33908533j,
       -5.96392544e-02-0.33908533j])
seigs = np.array([ 6.60808595e+00+1.26371423j,  6.60808595e+00-1.26371423j,
        1.01107741e+00+2.17469636j,  1.01107741e+00-2.17469636j,
       -1.89126960e+03+0.j        , -9.39666149e-01+9.82462206j,
       -9.39666149e-01-9.82462206j, -6.40807350e-01+6.84698956j,
       -6.40807350e-01-6.84698956j, -7.03924070e-01+3.50359625j,
       -7.03924070e-01-3.50359625j, -1.22442633e+00+3.94230549j,
       -1.22442633e+00-3.94230549j, -4.36368596e-01+2.64692808j,
       -4.36368596e-01-2.64692808j, -5.23321024e+00+0.j        ,
       -3.19838495e-01+1.94263115j, -3.19838495e-01-1.94263115j,
       -8.83874091e-01+1.78188064j, -8.83874091e-01-1.78188064j,
       -2.28045038e-01+1.52775665j, -2.28045038e-01-1.52775665j,
       -1.58293587e-01+1.25094699j, -1.58293587e-01-1.25094699j,
       -1.63137874e-01+1.0596423j , -1.63137874e-01-1.0596423j ,
       -9.28605576e-02+0.97837161j, -9.28605576e-02-0.97837161j,
       -7.81564719e-01+0.j        , -3.95197928e-02+0.80609294j,
       -3.95197928e-02-0.80609294j, -5.19972836e-02+0.75262746j,
       -5.19972836e-02-0.75262746j,  6.91116259e-03+0.61436724j,
        6.91116259e-03-0.61436724j, -9.21888568e-03+0.50336714j,
       -9.21888568e-03-0.50336714j, -6.43151160e-03+0.4156306j ,
       -6.43151160e-03-0.4156306j , -1.30529801e-02+0.35113684j,
       -1.30529801e-02-0.35113684j, -2.26173931e-04+0.23111015j,
       -2.26173931e-04-0.23111015j,  3.07456233e-03+0.01925242j,
        3.07456233e-03-0.01925242j,  2.84645355e-03+0.10090259j,
        2.84645355e-03-0.10090259j, -2.23392920e-03+0.15047945j,
       -2.23392920e-03-0.15047945j])

# eigs, v, u = compute_alpha(psi, 2, 100, 400, 70, 4, .5, 1)
# seigs, sv, su = compute_alpha(phi, 2, 100, 400, 70, 4, .5, 2)


# plt.figure(3)
# plt.plot(eigs, c='b', marker='*', label='angular, min = {a:.2f}, max = {b:.4f}'.format(a=np.min(eigs), b=np.max(eigs)))
# plt.plot(seigs, c='r', marker='*', label='scalar, min = {a:.2f}, max = {b:.4f}'.format(a=np.min(seigs), b=np.max(seigs)))
# plt.legend()
# plt.ylim([-10,10])
# plt.title(['Real alphas calculated from angular and scalar fluxes'])

plt.figure(1)
plt.scatter(np.real(eigs[2:]), np.imag(eigs[2:]), c='b',label=r'Angular $\alpha$')
plt.scatter(np.real(seigs[2:]),np.imag(seigs[2:]),c='r',label=r'Scalar $\alpha$')
plt.scatter(np.real(eigs[0:2]), np.imag(eigs[0:2]), c='b',marker='*')
plt.scatter(np.real(seigs[0:2]),np.imag(seigs[0:2]),c='r',marker='*')
plt.xlim([-70,10])
plt.legend()
plt.ylabel(r'Im($\alpha$)')
plt.xlabel(r'Re($\alpha$)')
plt.savefig('AlphaEig.eps')
##
## # comparing scalar and angular alphas for the coarse group
## eigs = np.real(collapse['eigs'])
## seigs = np.real(collapse['scalar_eigs'])
##
## plt.figure(4)
## plt.plot(eigs, 'g', label='angular, min = {a:.2f}, max = {b:.4f}'.format(a=np.min(eigs), b=np.max(eigs)))
## plt.plot(seigs, 'k', label='scalar, min = {a:.2f}, max = {b:.4f}'.format(a=np.min(seigs), b=np.max(seigs)))
## plt.legend()
## plt.title('Real alpha-eigenvalues calculated from angular and scalar fluxes, coarse group')

plt.show()
##
