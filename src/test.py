from Bio.PDB import PDBParser

class CarboneAlpha:
    """Classe pour représenter un carbone alpha d'une protéine."""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def compute_distance(self, other):
        dist = ((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + (other.z - self.z) ** 2) ** 0.5
        return dist

class Template:
    """Classe pour représenter le template utilisé."""
    
    def __init__(self, file):
        self.structure = self.get_c_alpha(file)
    
    def get_c_alpha(file):
        