import numpy as np
import h5py


class Material(object):
    def __init__(self, material_type):
        self.name = material_type
        self.filename = './data/' + material_type + '.hdf5'
        self.scatter_xs = None
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
            self.scatter_xs = hf['scat_xs'][:].transpose()
            self.total_xs = hf['total_xs'][:]
            self.chi = hf['chi'][:]
            self.nu_sigmaf = hf['nusigma_f'][:]
            self.inv_speed = hf['inverse_speed'][:]
            self.group_edges = hf['group_edges'][:]
        self.group_centers = (self.group_edges[0:-1] + self.group_edges[1:]) * 0.5
        self.num_energy_groups = np.size(self.group_centers)
