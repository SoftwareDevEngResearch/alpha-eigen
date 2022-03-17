import numpy as np
import h5py


class SimpleMaterial(object):
    def __init__(self, material_type, num_groups):
        self.name = material_type
        self.num_groups = num_groups
        self.total_xs = 1.0
        self.chi = 1.0
        self.inv_speed = 1.0
        if self.name == 'fuel':
            self.scatter_xs = 0.8
            self.nu_sigmaf = 0.7
        elif self.name == 'moderator':
            self.scatter_xs = 0.8
            self.nu_sigmaf = 0.0
        elif self.name == 'absorber':
            self.scatter_xs = 0.1
            self.nu_sigmaf = 0.0
        else:
            self.scatter_xs = None
            self.nu_sigmaf = None


class Material(SimpleMaterial):
    def __init__(self, material_type, num_groups):
        super().__init__(material_type, num_groups)
        self.name = material_type
        self.filename = './data/' + str(num_groups) + '_group/' + str(material_type) + '.hdf5'
        self.group_edges = None
        self.group_centers = None
        self.num_energy_groups = num_groups
        self.parse_file()

    def parse_file(self):
        with h5py.File(self.filename, 'r') as hf:
            self.scatter_xs = hf['scat_xs'][:].transpose()
            self.total_xs = hf['total_xs'][:]
            self.nu_sigmaf = hf['nusigma_f'][:]
            self.inv_speed = hf['inverse_speed'][:]
            self.group_edges = hf['group_edges'][:]
            self.chi = np.zeros(self.num_energy_groups)
            self.nusigma_f = self.chi.copy()
        self.group_centers = (self.group_edges[0:-1] + self.group_edges[1:]) * 0.5
        self.num_energy_groups = np.size(self.group_centers)
