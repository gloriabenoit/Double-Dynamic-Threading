import pandas as pd
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

def clean_DOPE_data(filename):
    ca_matrix = []
    
    with open(filename, "r") as dope :
        for ligne in dope:
            if ligne[3:7].strip() == "CA" and ligne[11:14].strip() == "CA":
                ca_matrix.append(ligne.split())
    
    columns = ['res1', 'temp1', 'res2', 'temp2'] + list(np.arange(0.25, 15, 0.5))
    dope_score = pd.DataFrame(ca_matrix, columns = columns) 
    dope_score = dope_score.drop(['temp1', 'temp2'], axis=1)
    
    return dope_score

def get_fasta_sequence(filename):
    sequence = ""
    with open(filename, "r") as fasta:
        for ligne in fasta:
            if ligne.startswith(">"):
                continue
            sequence += ligne.strip()
    return sequence

class DynamicMatrix:    
    def __init__(self, lines, columns, gap):
        self.matrix = np.zeros((lines, columns))
        self.lines = lines
        self.columns = columns
        self.gap = gap

    def initialize_matrix(self, first_val, start, end, get_score):
        if (start[0] < 0) or (start[1] < 0):
            raise ValueError("Start of initialization out of matrix.")
        if (end[0] >= self.lines) or (end[1] >= self.columns):
            raise ValueError("End of initialization out of matrix.")
        
        # Première case
        self.matrix[start[0], start[1]] = first_val
        
        # Remplissage de la première colonne jusqu'à la limite
        for i in range(start[0] + 1, end[0] + 1):
            self.matrix[i, start[1]] = self.matrix[i - 1, start[1]] + self.gap + get_score(i, start[1])

        # Remplissage de la première ligne jusqu'à la limite
        for j in range(start[1] + 1, end[1] + 1):
            self.matrix[start[0], j] = self.matrix[start[0], j - 1] + self.gap + get_score(start[0], j)

class LowLevelMatrix(DynamicMatrix):
    aa_codes = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    def __init__(self, gap, frozen, distance, dope, sequence):
        lines = len(sequence)
        columns = len(distance)
        
        DynamicMatrix.__init__(self, lines, columns, gap)

        # Vérification du blocage de la case
        if (frozen['seq_id'] >= lines) or (frozen['seq_id'] < 0):
            raise ValueError("Frozen line index out of matrix.")
        if (frozen['pos_id'] >= columns) or (frozen['pos_id'] < 0):
            raise ValueError("Frozen column index out of matrix")

        # Récupération du résidu fixé
        frozen['seq_res'] = sequence[frozen['seq_id']]
        
        self.frozen = frozen
        self.distance = distance
        self.dope = dope
        self.sequence = sequence

    def round_distance(self, dist):
        # arrondi au quart le plus proche
        rounded_value = round(dist * 4) / 4
        
        # ne garde que 0.25 ou 0.75
        decimal = rounded_value % 1
        if decimal == 0.0:
            return rounded_value + 0.25
        elif decimal == 0.5:
            return rounded_value + 0.25
        else:
            return rounded_value
    
    def get_score(self, i, j):
        dist = self.distance[self.frozen["pos_id"], j]

        # Cas du résidu bloqué avec sa propre position
        if (dist == 0):
            return 0
        
        closest_dist = self.round_distance(dist)

        score = self.dope.loc[(self.dope['res1'] == self.aa_codes[self.frozen['seq_res']]) & 
                              (self.dope['res2'] == self.aa_codes[self.sequence[i]]), 
                              closest_dist]
        
        return float(score.values[0])
    
    def fill_matrix(self):
        # Partie supérieure gauche
        self.initialize_matrix(self.get_score(0, 0), [0, 0], 
                               [self.frozen['seq_id'], self.frozen['pos_id']],
                               self.get_score)

        for i in range(1, self.frozen['seq_id'] + 1):
            for j in range(1, self.frozen['pos_id'] + 1):
                score = self.get_score(i, j)
                self.matrix[i, j] = score + min(self.matrix[i - 1, j - 1],
                                                self.matrix[i - 1, j] + self.gap,
                                                self.matrix[i, j - 1] + self.gap
                                               )
        # Partie inférieure droite
        self.initialize_matrix(self.matrix[self.frozen['seq_id'], self.frozen['pos_id']],
                               [self.frozen['seq_id'], self.frozen['pos_id']],
                               [self.lines - 1, self.columns - 1],
                               self.get_score)

        for i in range(self.frozen['seq_id'] + 1, self.lines):
            for j in range(self.frozen['pos_id'] + 1, self.columns):
                score = self.get_score(i, j)
                self.matrix[i, j] = score + min(self.matrix[i - 1, j - 1],
                                                self.matrix[i - 1, j] + self.gap,
                                                self.matrix[i, j - 1] + self.gap
                                               )

        max_score = self.matrix[self.lines - 1, self.columns - 1]
        return max_score

class HighLevelMatrix(DynamicMatrix):
    def __init__(self, gap, query, template, dope):
        distance = template.build_dist_matrix()
        lines = len(query)
        columns = len(distance)

        DynamicMatrix.__init__(self, lines, columns, gap)

        self.sequence = query
        self.distance = distance
        self.dope = dope

        self.get_score_matrix()

    def get_score_matrix(self):
        self.score_matrix = np.zeros((self.lines, self.columns))
        for i in range(self.lines):
            for j in range(self.columns):
                frozen = {'seq_id': i, 'pos_id': j}
                low_level = LowLevelMatrix(self.gap, frozen, self.distance, self.dope, self.sequence)
                self.score_matrix[i, j] =  low_level.fill_matrix()

    def get_score(self, i, j):
        score = self.score_matrix[i, j]
        return score

    def fill_matrix(self):
        # Initialisation
        self.initialize_matrix(self.get_score(0, 0), [0, 0], 
                               [self.lines - 1, self.columns - 1],
                               self.get_score)
        
        # Remplissage
        for i in range(1, self.lines):
            for j in range(1, self.columns):
                score = self.get_score(i, j)
                self.matrix[i, j] = score + min(self.matrix[i - 1, j - 1],
                                                self.matrix[i - 1, j] + self.gap,
                                                self.matrix[i, j - 1] + self.gap
                                               )
        max_score = self.matrix[self.lines - 1, self.columns - 1]

        return max_score

    def get_alignment(self):
        structure_align = []
        sequence_align = []
        
        i = self.lines - 1
        j = self.columns - 1
        while not ((i == 0) and (j == 0)):
            print(i, j)
            square = self.matrix[i, j]
            score = self.score_matrix[i, j]
            # Match
            if (square == self.matrix[i - 1, j - 1] + score):
                print("match")
                structure_align.insert(0, j + 1)
                sequence_align.insert(0, self.sequence[i])
                i = i - 1
                j = j - 1
            # Gap
            else:
                if (square == self.matrix[i - 1, j] + score + self.gap):
                    print("gap structure")
                    structure_align.insert(0, '-')
                    sequence_align.insert(0, self.sequence[i])
                    i = i - 1
                elif (square == self.matrix[i, j - 1] + score + self.gap):
                    print("gap sequence")
                    structure_align.insert(0, j + 1)
                    sequence_align.insert(0, '-')
                    j = j - 1

        return ''.join(sequence_align), ''.join(str(x) for x in structure_align)

if __name__ == "__main__":
    # Séquence 'query'
    FASTA_FILE = "../data/5AWL.fasta"
    QUERY = get_fasta_sequence(FASTA_FILE)

    # Structure 'template'
    PDB_FILE = "../data/5awl.pdb"
    TEMPLATE = Template(PDB_FILE)
    
    # Matrice de distance
    #DIST_MATRIX = TEMPLATE.build_dist_matrix()
    
    # Matrice DOPE
    DOPE_FILE = "../data/dope.par"
    DOPE_MATRIX = clean_DOPE_data(DOPE_FILE)

    # Information(s) supplémentaire(s)
    GAP = 0

    # Algorithme principal
    HIGH_LEVEL = HighLevelMatrix(GAP, QUERY, TEMPLATE, DOPE_MATRIX)
    MAX_SCORE = HIGH_LEVEL.fill_matrix()
    print(MAX_SCORE)
    ALIGN_SEQ, ALIGN_STRUCT = HIGH_LEVEL.get_alignment()
    print(ALIGN_SEQ)
    print(ALIGN_STRUCT)