import tables as tb
import numpy as np


class Material(tb.IsDescription):
    total_xs = tb.Float64Col()
    scatter_xs = tb.Float64Col()
    chi = tb.Float64Col()
    nu_sigmaf = tb.Float64Col()
    inv_speed = tb.Float64Col()


class EnergyGroup(tb.IsDescription):
    edge = tb.Float64Col()
    center = tb.Float64Col()

# import data from Ryan Mcclarren's format
npz_path = '../DMD_SN_MCCLARREN/'
metal_tmp = np.load(npz_path + 'metal_data.npz')
metal_scat = metal_tmp['metal_scat']
metal_sig_t = metal_tmp['metal_sig_t']
metal_chi = metal_tmp['metal_chi']
metal_nu_sig_f = metal_tmp['metal_nu_sig_f']
inv_speed = metal_tmp['metal_inv_speed']
group_edges = metal_tmp['metal_group_edges']
group_centers = (group_edges[0:-1] + group_edges[1:]) * 0.5
poly_tmp = np.load(npz_path + "poly_data.npz")
poly_scat = poly_tmp['poly_scat']
poly_sig_t = poly_tmp['poly_sig_t']
poly_chi = poly_tmp['poly_chi']
poly_nu_sig_f = poly_tmp['poly_nu_sig_f']

# create, format, and fill h5file
h5file = tb.open_file('70_group_data', mode='w', title='Data for 70-group Hetergeneous Slab Problem')

material_group = h5file.create_group('/', 'material', 'Material Data')
energy_group = h5file.create_group('/', 'energy', 'Energy Group Data')

metal_table = h5file.create_table(material_group, 'metal', Material, '70-group Metal Data')
poly_table = h5file.create_table(material_group, 'polyethylene', Material, '70-Group Polyethylene Data')
energy_table = h5file.create_table(energy_group, 'energy', EnergyGroup, 'Eenrgy group edges and centers')

# fill in rows
metal = metal_table.row
poly = poly_table.row
energy = energy_table.row
energy['edge'] = group_edges[0]; energy.append()
for i in range(70):
    metal['total_xs'] = metal_sig_t[i]
    metal['scatter_xs'] = metal_scat[i]
    metal['chi'] = metal_chi[i]
    metal['nu_sigmaf'] = metal_nu_sig_f[i]
    metal['inv_speed'] = inv_speed[i]
    poly['total_xs'] = poly_sig_t[i]
    poly['scatter_xs'] = poly_scat[i]
    poly['chi'] = poly_chi[i]
    poly['nu_sigmaf'] = poly_nu_sig_f[i]
    energy['edge'] = group_edges[i+1]
    energy['center'] = group_centers[i]

    metal.append()
    poly.append()
    energy.append()

metal.flush()
poly.flush()
energy.flush()
