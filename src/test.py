import numpy as np

class CarboneAlpha:
    """Classe pour représenter un carbone alpha d'une protéine."""
    
    def __init__(self, number, x, y, z):
        self.number = number
        self.x = x
        self.y = y
        self.z = z
    
    def compute_distance(self, other):
        dist = ((other.x - self.x) ** 2 + (other.y - self.y) ** 2 + (other.z - self.z) ** 2) ** 0.5
        return dist

class Template:
    """Classe pour représenter le template utilisé."""
    
    def __init__(self, file):
        self.structure = self.build_template_from_pdb(file)
        self.length = len(self.structure)

    def build_template_from_pdb(self, filename):
        list_calpha = []
        with open(filename, "r") as pdb :
            for ligne in pdb:
                if ligne.startswith("ATOM") and (ligne[12:16].strip() == "CA"):
                    number = ligne[6:11].strip()
                    x = float(ligne[30:38].strip())
                    y = float(ligne[38:46].strip())
                    z = float(ligne[46:54].strip())
                                       
                    list_calpha.append(CarboneAlpha(number, x, y, z))
        return list_calpha
        
    def build_dist_matrix(self):
        dist_list = []
        
        for i, atom in enumerate(self.structure):
            dist_ligne = []
            for other in (self.structure):
                dist_ligne.append(atom.compute_distance(other))
            dist_list.append(dist_ligne)
            
        dist_matrix = np.array(dist_list)
        return dist_matrix
    
    def __str__(self):
        string = ""
        for i, ca in enumerate(self.structure):
            string += f"position {i}-{ca.number}, coor( {ca.x}, {ca.y}, {ca.z})\n"
        return string

def clean_DOPE_matrix(filename):
    with open(filename, "r") as dope :
        for ligne in dope:
            if ligne[3:7].strip() == "CA" and ligne[11:14].strip() == "CA":
                print(ligne)
    return 8

if __name__ == "__main__":
    #test avec 5AWL
    PDB_FILE = "5awl.pdb"
    TEMPLATE = Template(PDB_FILE)
    print(TEMPLATE)
    print(TEMPLATE.build_dist_matrix())
    
    # matrice DOPE
    DOPE_FILE = "dope.par"
    DOPE_MATRIX = clean_DOPE_matrix(DOPE_FILE)
    print(DOPE_MATRIX)
    
    # import Bio.PDB as pdb

    # structure = pdb.PDBParser().get_structure("id","path")
    
    # for model in structure:
        # for chain in model:
            # for residue in chain:
                # for atom in residue:
                
# faire une matrice de confusion à la fin.