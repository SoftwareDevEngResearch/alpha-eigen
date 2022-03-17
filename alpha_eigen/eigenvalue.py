class Eigenvalue:
    def __init__(self):
        self.new = 1
        self.old = 1.1
        self.converged = False

    def update_eigenvalue(self):
        self.old = self.new
        self.new = 1
