#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:39:52 2021

@author: KaylaClements

Part one assumes infinite medium and solves transport problem w/ matrix inversion using 70 groups

Part two uses the calculated 70 group infinite flux to perform a group collapse of size n 
"""
import numpy as np
import matplotlib.pyplot as plt
import math



def matrixSolve(sigma_s, sig_t_in, chi, nusigma_f, G):
    """Compute homogeneous k_inf and flux for given energy data
    Data from Ryan McClarren
    Inputs:
        G:               number of groups 
    Outputs:
        phi:             value of scalar flux in each group
    """
    sigma_t = np.zeros((G,G))
    phi = np.zeros(G)
    
    for i in range(G):
        sigma_t[i,i] = sig_t_in[i]
    
    Diff = sigma_t - sigma_s
    iDiff = np.linalg.inv(Diff)
    phi = np.matmul(iDiff,chi)

    k = 0
    for i in range(G):
        k += nusigma_f[i]*phi[i]
    
    print(k)
    
    return phi

# # Group collapse
def collapse(phi, n, fine_edges, coarse_edges, G, chi, sigma_t, sigma_s, nusigma_f, inv_speed, opt):
    """Perform group collapse
    Data from Ryan McClarren
    Inputs:
        phi:             fine group infinite homogeneous flux 
        n:               coarse group size
        fine_edges:      fine group edges
        coarse_edges:    coarse group edges
        G:               fine group size
        opt:             whether to calculate inv_speed; only metal inv_speed is used
    Outputs:
        psi:             value of angular flux in each zone
    """
    coarse_phi = np.zeros(n)
    coarse_scat = np.zeros((n,n))
    coarse_sigt = np.zeros(n)
    coarse_chi = np.zeros(n)
    coarse_nusigf = np.zeros(n)
    coarse_invspeed = np.zeros(n)
    

    # Get corresponding fine group number for coarse group edges
    index = np.zeros(n+1,np.int)
    i = 0
    for j in range(G):
        if i > n:
            break
        if ( abs(fine_edges[j] - coarse_edges[i]) < 0.0001 ):
            index[i] = j
            i += 1
    
    # Calculate coarse group flux, chi, sigt, nusigf, and invspeed
    i = 1
    for j in range(index[n]):
        if j < index[i]:                    # If we're in the current coarse group
            coarse_phi[i-1] += phi[j]
            coarse_chi[i-1] += chi[j]
            
            # These will be divided by coarse group flux once calculated, 
            # So essentially this is all phi_g * sigma_g
            coarse_sigt[i-1] += sigma_t[j]*phi[j]
            coarse_nusigf[i-1] += nusigma_f[j]*phi[j]
            coarse_invspeed[i-1] += inv_speed[j]*phi[j]
            
            
            if (j == index[i] - 1):         # If the next fine group is in the next coarse group, divide by coarse flux and move to next coarse group
                coarse_sigt[i-1] = coarse_sigt[i-1]/coarse_phi[i-1]
                coarse_nusigf[i-1] = coarse_nusigf[i-1]/coarse_phi[i-1]
                coarse_invspeed[i-1] = coarse_invspeed[i-1]/coarse_phi[i-1]
                i += 1

    # Calculate coarse scattering cross sections        
    for coarse_from in range(n):
        fromgroups = (index[coarse_from], index[coarse_from + 1])
        coarseflux = coarse_phi[coarse_from]
        for coarse_to in range(n):
            togroups = (index[coarse_to], index[coarse_to + 1])
            
            for fine_from in range(fromgroups[0], fromgroups[1]):
                fineflux = phi[fine_from]
                for fine_to in range(togroups[0], togroups[1]):
                    coarse_scat[coarse_to, coarse_from] += sigma_s[fine_to, fine_from]*fineflux
                    
            coarse_scat[coarse_to, coarse_from] = coarse_scat[coarse_to, coarse_from]/coarseflux
            
    # Once all group constants calculated, compute k_inf again to compare
    matrix_solve(coarse_scat, coarse_sigt, coarse_chi, coarse_nusigf, n)    
    if opt == 1:
        return coarse_scat, coarse_sigt, coarse_chi, coarse_nusigf, coarse_invspeed
    else:
        return coarse_scat, coarse_sigt, coarse_chi, coarse_nusigf
