from Attempt2.classes.group import Group


class Zone:
    width = 0
    mu = 0
    weights = 0

    def __init__(self, number, midpoint, num_groups):
        self.number = number
        self.region_name = 'undefined'
        self.material = None
        self.midpoint = midpoint
        self.num_groups = num_groups
        self.group = []

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

    def create_groups(self, num_angles):
        for i in range(self.num_groups):
            self.group.append(Group(i, num_angles, self.mu, self))

    def reset_source_term(self):
        for i in range(self.num_groups):
            self.group[i].source = 0.0

    def calculate_power_iteration_source(self):
        for g in range(self.num_groups):
            for g_prime in range(self.num_groups):
                self.group[g].source = 0.5 * self.fission_source(g, g_prime)

    def fission_source(self, g, g_prime):
        fission_source = self.material.chi[g] * self.material.nusigma_f[g_prime] * self.group[g_prime].phi.old
        return fission_source

    def scattering_source(self, g, g_prime):
        # Exclude in-group scattering by multiplying by (g_prime != g)
        scattering_source = self.group[g_prime].phi.new * self.material.scatter_xs[g_prime, g] * (g_prime != g)
        return scattering_source

    def calculate_one_group_source_term(self, g):
        for g_prime in range(self.num_groups):
            self.group[g].source = 0.5 * (self.fission_source(g, g_prime) + self.scattering_source(g, g_prime))

    def power_iteration_flux_update(self, k):
        for i in range(self.num_groups):
            self.group[i].phi.power_iteration_flux_update(k)
