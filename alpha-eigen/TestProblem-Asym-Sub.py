"""
This file uses DMD to solve the Asymmetric Kornreich and Parsons alpha eigenvalue problem, sub critical version
D. E. KORNREICH and D. KENT PARSONS, “Time– Eigenvalue Calculations in Multi-Region Cartesian Geometry Using Green’s Functions,” 
Ann. Nucl. Energy, 32, 9, 964 (June 2005); https://doi.org/10.1016/j.anucene. 2005.02.004.

It estimates the eigenvalue using Shanks acceleration to check that the code gets the right answer for k = 0.4556758 before doing the alpha calc
"""

from . import multigroup_sn as sn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick

import numpy as np

from scipy import interpolate

import mpmath as mpm


def compute_alpha(psi_input, skip, num_time_steps, num_zones, num_energy_groups, num_sn_angles, dt):
    it = num_time_steps - 1

    # need to reshape matrix
    phi_mat = np.zeros((num_zones * num_energy_groups * num_sn_angles, num_time_steps))
    for i in range(num_time_steps):
        phi_mat[:, i] = np.reshape(psi_input[:, :, :, i], num_zones * num_energy_groups * num_sn_angles)
    [u, s, v] = np.linalg.svd(phi_mat[:, skip:it], full_matrices=False)
    print(u.shape, s.shape, v.shape)

    # make diagonal matrix
    # print("Cumulative e-val sum:", (1-np.cumsum(s)/np.sum(s)).tolist())
    spos = s[(1 - np.cumsum(s) / np.sum(s)) > 1e-13]  # [ np.abs(s) > 1.e-5]
    mat_size = np.min([num_zones * num_energy_groups * num_sn_angles, len(spos)])
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


# In[2]:


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def hide_spines(intx=False, inty=False):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))


def show(nm, a=0, b=0, show=1):
    hide_spines(a, b)
    # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    # plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
    # ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    if (len(nm) > 0):
        plt.savefig(nm, bbox_inches='tight');
    if show:
        plt.show()
    else:
        plt.close()


# In[3]:


def run_slab(cells=100, num_sn_angles=16):
    num_energy_groups = 1
    slab_length = 9.1
    num_zones = int(np.round(cells * slab_length))  # 540
    zone_width = slab_length / num_zones
    source = np.ones((num_zones, num_energy_groups)) * 0
    zone_midpoints = np.linspace(zone_width / 2, slab_length - zone_width / 2, num_zones)
    sigma_t = np.ones((num_zones, num_energy_groups))
    nusigma_f = np.zeros((num_zones, num_energy_groups))
    chi = np.ones((num_zones, num_energy_groups))
    sigma_s = np.zeros((num_zones, num_energy_groups, num_energy_groups))
    current_zone = 0
    for x in zone_midpoints:
        # first region
        if x < 1.0:
            sigma_s[current_zone, 0:num_energy_groups, 0:num_energy_groups] = 0.8
            nusigma_f[current_zone, 0:num_energy_groups] = 0.3
        # second region
        elif x < 2.0:
            sigma_s[current_zone, 0:num_energy_groups, 0:num_energy_groups] = 0.8
            nusigma_f[current_zone, 0:num_energy_groups] = 0.0
        # third region
        elif x < 7.0:
            sigma_s[current_zone, 0:num_energy_groups, 0:num_energy_groups] = 0.1
            nusigma_f[current_zone, 0:num_energy_groups] = 0.0
        # fourth region
        elif x < 8.0:
            sigma_s[current_zone, 0:num_energy_groups, 0:num_energy_groups] = 0.8
            nusigma_f[current_zone, 0:num_energy_groups] = 0.0
        # fourth region
        else:
            sigma_s[current_zone, 0:num_energy_groups, 0:num_energy_groups] = 0.8
            nusigma_f[current_zone:num_zones, 0:num_energy_groups] = 0.3
        current_zone += 1

    plt.plot(zone_midpoints, chi)  # -np.flip(chi,0))
    plt.plot(zone_midpoints, nusigma_f)  # -np.flip(nusigma_f,0))
    plt.plot(zone_midpoints, sigma_s[:, 0])  # -np.flip(sigma_s[:,0],0))
    plt.show()
    inv_speed = 1.0

    # N = 196
    MU, W = np.polynomial.legendre.leggauss(num_sn_angles)
    boundary_conditions = np.zeros((num_sn_angles, num_energy_groups))

    x, k, phi_sol = sn.multigroup_k(num_zones, zone_width, num_energy_groups, sigma_t, sigma_s, nusigma_f, chi, num_sn_angles, boundary_conditions,
                                    tolerance=1.0e-8, maxits=400000, LOUD=1)
    return x, k, phi_sol


# In[4]:


num_k_eigenvalues = 20
k_eigenvalues = np.zeros(num_k_eigenvalues)
for i in range(num_k_eigenvalues):
    num_cells = 10 * (i + 1)
    num_sn_angles = 8 * (i + 1)
    x, k, phi_sol = run_slab(num_cells, num_sn_angles)
    k_eigenvalues[i] = k
    if i > 0:
        T = mpm.shanks(k_eigenvalues[:(i + 1)])
        for row in T:
            mpm.nprint(row)

# In[5]:


T = mpm.shanks(k_eigenvalues)
for row in T:
    mpm.nprint(row)

# In[ ]:


# In[6]:


num_energy_groups = 1
slab_length = 9.1
num_cells = 200
num_sn_angles = 196
num_zones = int(np.round(num_cells * slab_length))  # 540
zone_length = slab_length / num_zones
source = np.ones((num_zones, num_energy_groups)) * 0
zone_midpoints = np.linspace(zone_length / 2, slab_length - zone_length / 2, num_zones)
sigma_t = np.ones((num_zones, num_energy_groups))
nusigma_f = np.zeros((num_zones, num_energy_groups))
chi = np.ones((num_zones, num_energy_groups))
sigma_s = np.zeros((num_zones, num_energy_groups, num_energy_groups))
inv_speed = 1
count = 0
nusf = 0.3
for x in zone_midpoints:
    # first region
    if x < 1.0:
        sigma_s[count, 0:num_energy_groups, 0:num_energy_groups] = 0.8
        nusigma_f[count, 0:num_energy_groups] = nusf
    # second region
    elif x < 2.0:
        sigma_s[count, 0:num_energy_groups, 0:num_energy_groups] = 0.8
        nusigma_f[count, 0:num_energy_groups] = 0.0
    # third region
    elif x < 7.0:
        sigma_s[count, 0:num_energy_groups, 0:num_energy_groups] = 0.1
        nusigma_f[count, 0:num_energy_groups] = 0.0
    # fourth region
    elif x < 8.0:
        sigma_s[count, 0:num_energy_groups, 0:num_energy_groups] = 0.8
        nusigma_f[count, 0:num_energy_groups] = 0.0
    # fourth region
    else:
        sigma_s[count, 0:num_energy_groups, 0:num_energy_groups] = 0.8
        nusigma_f[count:num_zones, 0:num_energy_groups] = nusf
    count += 1
MU, W = np.polynomial.legendre.leggauss(num_sn_angles)
BCs = np.zeros((num_sn_angles, num_energy_groups))
psi0 = np.zeros((num_zones, num_sn_angles, num_energy_groups)) + 1e-12
# psi0[0,MU>0,0] = 1
psi0[-1, MU < 0, 0] = 1
numsteps = 500
dt = 1.0e-1
x, phi, psi = sn.multigroup_td(num_zones, zone_length, num_energy_groups, sigma_t, (sigma_s), nusigma_f, chi, inv_speed,
                               num_sn_angles, BCs, psi0, source, numsteps, dt, tolerance=1.0e-8, maxits=400000, LOUD=0)
plt.plot(x, phi[:, 0, -1])
plt.show()

# In[7]:


print(phi.shape)
plt.plot(x, phi[:, :, -1])
plt.show()

# In[8]:


psi.shape
step = 100
included = 375
eigsN, vsN, u = compute_alpha(psi[:, :, :, step:(step + included + 1)], 0, included, num_zones, num_energy_groups, num_sn_angles, dt)

# In[9]:


print(vsN.shape, u.shape)
print(eigsN[np.abs(np.imag(eigsN)) < 1e+1])

# In[10]:


MU, W = np.polynomial.legendre.leggauss(num_sn_angles)
psi0 = np.random.uniform(high=1, low=0, size=(num_zones, num_sn_angles, num_energy_groups)) + 1e-12
numsteps = 500
dt = 1.0e-1
# psi0[0,MU>0,0] = 1
# psi0[-1,MU<0,0] = 1
x, phi2, psi2 = sn.multigroup_td(num_zones, zone_length, num_energy_groups, sigma_t, (sigma_s), nusigma_f, chi, inv_speed,
                                 num_sn_angles, BCs, psi0, source, numsteps, dt, tolerance=1.0e-8, maxits=400000, LOUD=0)
plt.plot(x, phi2[:, 0, -1])
plt.show()

# In[11]:


print(phi.shape)
plt.plot(x, phi2[:, :, -1])
plt.show()

# In[12]:


psi.shape
step = 100
included = 400
eigsN, vsN, u = compute_alpha(psi2[:, :, :, step:(step + included + 1)], 0, included, num_zones, num_energy_groups, num_sn_angles, dt)

# In[ ]:


print(vsN.shape, u.shape)
print(eigsN[np.abs(np.imag(eigsN)) < 1e+1])

# In[ ]:


print(vsN.shape, u.shape)
print(eigsN[np.abs(np.imag(eigsN)) < 1e+0])

# In[16]:


print(u.shape, vsN.shape)
evect = np.reshape(np.dot(u[:, 0:vsN.shape[0]], vsN[:, np.argmin(np.abs(-0.2932468 - eigsN))]), (num_zones, num_sn_angles, num_energy_groups))
phi_mat = evect[:, 0] * 0
print(evect.shape, phi_mat.shape)
for angle in range(num_sn_angles):
    phi_mat += evect[:, angle] * W[angle]

evect = np.reshape(np.dot(u[:, 0:vsN.shape[0]], vsN[:, np.argmin(np.abs(-.32 - eigsN))]), (num_zones, num_sn_angles, num_energy_groups))
phi_mat2 = evect[:, 0] * 0
print(evect.shape, phi_mat.shape)
for angle in range(num_sn_angles):
    phi_mat2 += evect[:, angle] * W[angle]

# In[ ]:


fund = np.loadtxt("/Users/ryanmcclarren/Downloads/Brezler1_asym.csv", delimiter=",")
fund_sort = np.sort(fund[:, 0])
fund_new = fund * 0
for i in range(fund_sort.size):
    fund_new[i, :] = fund[np.argmin(np.abs(fund[:, 0] - fund_sort[i])), :]

sec = np.loadtxt("/Users/ryanmcclarren/Downloads/Brezler2_asym.csv", delimiter=",")
fund_sort = np.sort(sec[:, 0])
sec_new = sec * 0
for i in range(fund_sort.size):
    sec_new[i, :] = sec[np.argmin(np.abs(sec[:, 0] - fund_sort[i])), :]

# In[17]:


print(phi_mat.shape)
plt.plot(x, np.real(phi_mat) / np.max(np.abs(phi_mat)), label="Fundamental DMD")
# plt.plot(fund_new[:,0],fund_new[:,1]/np.max(np.abs(fund[:,1])),"--")
plt.plot(x, -np.real(phi_mat2) / np.max(np.abs(phi_mat2)), "--", label="Second DMD")
# plt.plot(sec_new[:,0],sec_new[:,1]/np.max(np.abs(sec_new[:,1])),"-.")
plt.legend(loc="best")
show("asymmetric_sub.pdf")

# In[ ]:
