"""
Double Dynamic Programming algorithm to thread sequence onto template.

Author :
    Gloria BENOIT
Date :
    2024-09-12
"""
from multiprocessing import Pool

import pandas as pd
import numpy as np


class AlphaCarbon:
    """
    Class used to represent a protein's alpha carbon.

    Instance Attributes
    -------------------
    number : int
        Position number in protein.
    x : float
        X position.
    y : float
        Y position.
    z : float
        Z position.

    Methods
    -------
    compute_distance(other)
        Compute distance between himself and another alpha carbon.
    """

    def __init__(self, number, x, y, z):
        """
        Construct an alpha carbon.

        Parameters
        ----------
        number : int
            Position number in protein.
        x : float
            X position.
        y : float
            Y position.
        z : float
            Z position.
        """
        self.number = number
        self.x = x
        self.y = y
        self.z = z

    def compute_distance(self, other):
        """
        Compute distance between himself and another alpha carbon.

        Parameters
        ----------
        other : AlphaCarbon
            Another alpha carbon.

        Returns
        -------
        float
            Distance between both alpha carbon.
        """
        dist = (
            (other.x - self.x) ** 2
            + (other.y - self.y) ** 2
            + (other.z - self.z) ** 2
        ) ** 0.5
        return dist


class Template:
    """
    Class used to represent a structural template.

    Instance Attributes
    -------------------
    structure : list of AlphaCarbon
        All alpha carbon found in structure.
    length : int
        Number of alpha carbon found.

    Methods
    -------
    build_from_pdb(filename)
        Retrieve structural template from PDB file.
    build_dist_matrix()
        Compute a distance matrix between all alpha carbon
        in the template.
    """

    def __init__(self, file):
        """
        Construct a structural template.

        Parameters
        ----------
        file : str
            PDB file to use.
        """
        self.structure = self.build_from_pdb(file)
        self.length = len(self.structure)

    def __str__(self):
        """
        Return information on the template's alpha carbons.

        Returns
        -------
        str
            Number, position and coordinates of each alpha carbon
            in the template.
        """
        string = ""
        for i, ca in enumerate(self.structure):
            string += (
                f"position {i}-{ca.number}, coor( {ca.x}, {ca.y}, {ca.z})\n"
            )
        return string

    def build_from_pdb(self, filename):
        """
        Retrieve structural template from PDB file.

        Parameters
        ----------
        filename : str
            PDB file.

        Returns
        -------
        list of AlphaCarbon
            List of all alpha carbon found in the template.
        """
        list_calpha = []
        with open(filename, "r", encoding='UTF-8') as pdb:
            for ligne in pdb:
                if ligne.startswith("ATOM") and (ligne[12:16].strip() == "CA"):
                    number = ligne[6:11].strip()
                    x = float(ligne[30:38].strip())
                    y = float(ligne[38:46].strip())
                    z = float(ligne[46:54].strip())

                    list_calpha.append(AlphaCarbon(number, x, y, z))
        return list_calpha

    def build_dist_matrix(self):
        """
        Compute a distance matrix between all alpha carbon in the template.

        Returns
        -------
        numpy.ndarray
            2D array of float, representing the distance between
            all pairs of alpha carbon in template.
        """
        dist_list = []

        for atom in self.structure:
            dist_ligne = []
            for other in self.structure:
                dist_ligne.append(atom.compute_distance(other))
            dist_list.append(dist_ligne)

        dist_matrix = np.array(dist_list)
        return dist_matrix


class DynamicMatrix:
    """
    Class used to represent a dynamic programming matrix.

    Instance Attributes
    -------------------
    matrix : numpy.ndarray
        2D array of float, representing the scores obtained.
    lines : int
        Number of lines in matrix.
    columns : int
        Number of columns in matrix.
    gap : int
        Gap penalty.

    Methods
    -------
    initialize_matrix(first_val, start, end)
        Initialize the first line and column of part of the matrix.
    """

    def __init__(self, lines, columns, gap):
        """
        Construct a dynamic programming matrix.

        Parameters
        ----------
        lines : int
            Number of lines in matrix.
        columns : int
            Number of columns in matrix.
        gap : int
            Gap penalty.
        """
        self.matrix = np.zeros((lines, columns))
        self.lines = lines
        self.columns = columns
        self.gap = gap

    def initialize_matrix(self, first_val, start, end):
        """
        Initialize the first line and column of part of the matrix.

        Parameters
        ----------
        first_val : float
            Value of the first square.
        start : list
            Line and column index of the first position to initialize.
        end : list
            Line anc column index of last position to initialize.

        Raises
        ------
        ValueError
            If the part to initialize is out of the matrix.
        """
        if (start[0] < 0) or (start[1] < 0):
            raise ValueError("Start of initialization out of matrix.")
        if (end[0] >= self.lines) or (end[1] >= self.columns):
            raise ValueError("End of initialization out of matrix.")

        # Première case
        self.matrix[start[0], start[1]] = first_val

        # Remplissage de la première colonne jusqu'à la limite
        for i in range(start[0] + 1, end[0] + 1):
            self.matrix[i, start[1]] = self.matrix[i - 1, start[1]] + self.gap

        # Remplissage de la première ligne jusqu'à la limite
        for j in range(start[1] + 1, end[1] + 1):
            self.matrix[start[0], j] = self.matrix[start[0], j - 1] + self.gap


class LowLevelMatrix(DynamicMatrix):
    """
    Class used to represent a low level matrix.

    Lines are the sequence and columns the structure.
    Inherits all attributes and methods from DynamicMatrix.

    Class Attributes
    ----------------
    aa_codes : dict
        Dictionnary of amino acides codes, with one letter code
        as keys and three letter code as values.

    Instance Attributes
    -------------------
    frozen : dict
        Dictionary containing information on the frozen square.
            - 'seq_id' : int
                Sequence index of frozen square.
            - 'pos_id' : int
                Structure index of frozen square.
            - 'seq_res' : str
                Amino acid of the frozen square.
    distance : numpy.ndarray
        2D array of float, representing the distance between
        all pairs of alpha carbon in template.
    dope : pandas.DataFrame
        Dataframe containing DOPE potentials based on amino acids
        and distance.
            - 'res1' : str
                Three letter code of the first amino acid.
            - 'res2' : str
                Three letter code of the second amino acid.
            - All other columns : float
                Column names are distance from 0.25 to 14.75 with 0.5
                incrementation. Values are the DOPE potentials for said
                distance between first and second amino acids.
    sequence : str
        Sequence to thread on template.

    Methods
    -------
    round_distance(dist)
        Round value to closest 0.25 or 0.75 decimal.
    get_score(i, j)
        Compute score for i residue and j position.
    fill_matrix()
        Compute low level matrix by dynamic programming.
    """

    aa_codes = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    }

    def __init__(self, gap, frozen, distance, dope, sequence):
        """
        Construct a low level matrix.

        Parameters
        ----------
        gap : int
            Gap penalty.
        frozen : dict
            Dictionary containing information on the frozen square.
                - 'seq_id' : int
                    Sequence index of frozen square.
                - 'pos_id' : int
                    Structure index of frozen square.
        distance : numpy.ndarray
            2D array of float, representing the distance between
            all pairs of alpha carbon in template.
        dope : pandas.DataFrame
            Dataframe containing DOPE potentials based on amino acids
            and distance.
                - 'res1' : str
                    Three letter code of the first amino acid.
                - 'res2' : str
                    Three letter code of the second amino acid.
                - All other columns : float
                    Column names are distance from 0.25 to 14.75 with 0.5
                    incrementation. Values are the DOPE potentials for said
                    distance between first and second amino acids.
        sequence : str
            Sequence to thread on template.

        Raises
        ------
        ValueError
            If frozen square is out of matrix.
        """
        lines = len(sequence)
        columns = np.shape(distance)[1]

        DynamicMatrix.__init__(self, lines, columns, gap)

        # Vérification du blocage de la case
        if (frozen["seq_id"] >= lines) or (frozen["seq_id"] < 0):
            raise ValueError("Frozen line index out of matrix.")
        if (frozen["pos_id"] >= columns) or (frozen["pos_id"] < 0):
            raise ValueError("Frozen column index out of matrix")

        # Récupération du résidu fixé
        frozen["seq_res"] = sequence[frozen["seq_id"]]

        self.frozen = frozen
        self.distance = distance
        self.dope = dope
        self.sequence = sequence

    def round_distance(self, dist):
        """
        Round value to closest 0.25 or 0.75 decimal.

        If value is equally closer to 0.25 and 0.75,
        it is rounded to the upper.

        Parameters
        ----------
        dist : float
            Distance between two alpha carbon.

        Returns
        -------
        float
            Rounded distance.
        """
        # arrondi au quart le plus proche
        rounded_value = round(dist * 4) / 4

        # ne garde que 0.25 ou 0.75
        decimal = rounded_value % 1
        if decimal == 0.0:
            return rounded_value + 0.25
        if decimal == 0.5:
            return rounded_value + 0.25

        return rounded_value

    def get_score(self, i, j):
        """
        Compute score for i residue and j position.

        Parameters
        ----------
        i : int
            Residue index.
        j : int
            Position index.

        Returns
        -------
        float
            DOPE potential for said residue at said position,
            with distance to frozen square.
        """
        dist = self.distance[self.frozen["pos_id"], j]
        closest_dist = self.round_distance(dist)
        score = self.dope.loc[
            (self.dope["res1"] == self.aa_codes[self.frozen["seq_res"]])
            & (self.dope["res2"] == self.aa_codes[self.sequence[i]]),
            closest_dist,
        ]

        return float(score.values[0])

    def fill_matrix(self):
        """
        Compute low level matrix by dynamic programming.

        Returns
        -------
        float
            Final score of low level matrix.
        """
        # Partie supérieure gauche
        self.initialize_matrix(
            self.get_score(0, 0),
            [0, 0],
            [self.frozen["seq_id"] - 1, self.frozen["pos_id"] - 1],
        )

        for i in range(1, self.frozen["seq_id"]):
            for j in range(1, self.frozen["pos_id"]):
                score = self.get_score(i, j)
                self.matrix[i, j] = min(
                    self.matrix[i - 1, j - 1] + score,
                    self.matrix[i - 1, j] + self.gap,
                    self.matrix[i, j - 1] + self.gap,
                )

        # Case fixée
        self.matrix[self.frozen["seq_id"], self.frozen["pos_id"]] = (
            self.matrix[self.frozen["seq_id"] - 1, self.frozen["pos_id"] - 1]
        )

        # Partie inférieure droite (si elle existe)
        if (
            self.frozen["pos_id"] == self.columns - 1
            or self.frozen["seq_id"] == self.lines - 1
        ):
            max_score = self.matrix[
                self.frozen["seq_id"], self.frozen["pos_id"]
            ]

        else:
            self.initialize_matrix(
                self.matrix[self.frozen["seq_id"], self.frozen["pos_id"]],
                [self.frozen["seq_id"] + 1, self.frozen["pos_id"] + 1],
                [self.lines - 1, self.columns - 1],
            )

            for i in range(self.frozen["seq_id"] + 1, self.lines):
                for j in range(self.frozen["pos_id"] + 1, self.columns):
                    score = self.get_score(i, j)
                    self.matrix[i, j] = min(
                        self.matrix[i - 1, j - 1] + score,
                        self.matrix[i - 1, j] + self.gap,
                        self.matrix[i, j - 1] + self.gap,
                    )

            max_score = self.matrix[self.lines - 1, self.columns - 1]
        return max_score


class HighLevelMatrix(DynamicMatrix):
    """
    Class used to represent a high level matrix.

    Lines are the sequence and columns the structure.
    Inherits all attributes and methods from DynamicMatrix.

    Instance Attributes
    -------------------
    sequence : str
        Sequence to thread on template.
    distance : numpy.ndarray
        2D array of float, representing the distance between
        all pairs of alpha carbon in template.
    dope : pandas.DataFrame
        Dataframe containing DOPE potentials based on amino acids
        and distance.
            - 'res1' : str
                Three letter code of the first amino acid.
            - 'res2' : str
                Three letter code of the second amino acid.
            - All other columns : float
                Column names are distance from 0.25 to 14.75 with 0.5
                incrementation. Values are the DOPE potentials for said
                distance between first and second amino acids.
    score_matrix : numpy.ndarray
        2D array of float, representing the maximum scores obtained in all
        low level matrixes.

    Methods
    -------
    compute_low_level(args)
        Construct a low level matrix.
    get_score_matrix()
        Construct score matrix based on maximum scores obtained in
        low level matrixes.
    get_score(i, j)
        Compute score for i residue and j position.
    fill_matrix()
        Compute high level matrix by dynamic programming.
    get_alignment()
        Find optimal alignement.
    """

    def __init__(self, gap, query, template, dope):
        """
        Construct a low level matrix.

        Parameters
        ----------
        gap : int
            Gap penalty.
        query : str
            Sequence to thread on template.
        template : Template
            Structural template.
        dope : pandas.DataFrame
            Dataframe containing DOPE potentials based on amino acids
            and distance.
                - 'res1' : str
                    Three letter code of the first amino acid.
                - 'res2' : str
                    Three letter code of the second amino acid.
                - All other columns : float
                    Column names are distance from 0.25 to 14.75 with 0.5
                    incrementation. Values are the DOPE potentials for said
                    distance between first and second amino acids.
        """
        distance = template.build_dist_matrix()
        lines = len(query)
        columns = len(distance)

        # Ajout d'une colonne et ligne au départ
        DynamicMatrix.__init__(self, lines + 1, columns + 1, gap)

        self.sequence = query
        self.distance = distance
        self.dope = dope

        self.get_score_matrix()

    def compute_low_level(self, args):
        """
        Construct a low level matrix.

        Parameters
        ----------
        args : tuple
            Contains the following :
                - gap : int
                    Gap penalty.
                - distance : numpy.ndarray
                    2D array of float, representing the distance between
                    all pairs of alpha carbon in template.
                - dope : pandas.DataFrame
                    Dataframe containing DOPE potentials based on amino acids
                    and distance.
                        - 'res1' : str
                            Three letter code of the first amino acid.
                        - 'res2' : str
                            Three letter code of the second amino acid.
                        - All other columns : float
                            Column names are distance from 0.25 to 14.75
                            with 0.5 incrementation. Values are the DOPE
                            potentials for said distance between first and
                            second amino acids.
                - sequence : str
                    Sequence to thread on template.
                - i : int
                    Residue index to freeze.
                - j : int
                    Position index to freeze.

        Returns
        -------
        int
            Frozen residue index.
        int
            Frozen position index.
        float
            Final score obtained.
        """
        gap, distance, dope, sequence, i, j = args
        frozen = {"seq_id": i, "pos_id": j}
        low_level = LowLevelMatrix(gap, frozen, distance, dope, sequence)
        max_score = low_level.fill_matrix()

        return i, j, max_score

    def get_score_matrix(self):
        """
        Construct score matrix.

        Score are based on maximum scores obtained in low level matrixes.
        The construction of all low level matrixes is parallelized.
        """
        self.score_matrix = np.zeros((self.lines - 1, self.columns - 1))

        # Parallélisation
        args = [
            (self.gap, self.distance, self.dope, self.sequence, i, j)
            for i in range(self.lines - 1)
            for j in range(self.columns - 1)
        ]

        with Pool() as pool:
            low_levels = pool.map(self.compute_low_level, args)

        for i, j, value in low_levels:
            self.score_matrix[i, j] = value

    def get_score(self, i, j):
        """
        Compute score for i residue and j position.

        Parameters
        ----------
        i : int
            Residue index.
        j : int
            Position index.

        Returns
        -------
        float
            Maximal score of low level matrix where (i, j) is frozen.
        """
        score = self.score_matrix[i, j]
        return score

    def fill_matrix(self):
        """
        Compute high level matrix by dynamic programming.

        Returns
        -------
        float
            Final score of high level matrix.
        """
        # Initialisation
        self.initialize_matrix(0, [0, 0], [self.lines - 1, self.columns - 1])

        # Remplissage
        for i in range(1, self.lines):
            for j in range(1, self.columns):
                score = self.get_score(i - 1, j - 1)
                self.matrix[i, j] = min(
                    self.matrix[i - 1, j - 1] + score,
                    self.matrix[i - 1, j] + self.gap,
                    self.matrix[i, j - 1] + self.gap,
                )
        max_score = self.matrix[self.lines - 1, self.columns - 1]

        return max_score

    def get_alignment(self):
        """
        Find optimal alignement.

        Returns
        -------
        str
            Aligned sequence.
        str
            Aligned structure.
        """
        structure_align = []
        sequence_align = []

        i = self.lines - 1
        j = self.columns - 1
        while not ((i == 0) and (j == 0)):
            print(i, j)
            square = self.matrix[i, j]
            score = self.score_matrix[i - 1, j - 1]
            # Match
            if square == self.matrix[i - 1, j - 1] + score:
                print("match")
                structure_align.insert(0, j)
                sequence_align.insert(0, self.sequence[i - 1])
                i = i - 1
                j = j - 1
            # Gap
            else:
                if square == self.matrix[i - 1, j] + self.gap:
                    print("gap structure")
                    structure_align.insert(0, "-")
                    sequence_align.insert(0, self.sequence[i - 1])
                    i = i - 1
                elif square == self.matrix[i, j - 1] + self.gap:
                    print("gap sequence")
                    structure_align.insert(0, j)
                    sequence_align.insert(0, "-")
                    j = j - 1

        return "".join(sequence_align), "".join(
            str(x) for x in structure_align
        )


def clean_dope_data(filename):
    """Clean DOPE potentials file to keep only alpha carbon information.

    Parameters
    ----------
    filename : str
        File of DOPE potentials.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing DOPE potentials based on amino acids
        and distance.
            - 'res1' : str
                Three letter code of the first amino acid.
            - 'res2' : str
                Three letter code of the second amino acid.
            - All other columns : float
                Column names are distance from 0.25 to 14.75 with 0.5
                incrementation. Values are the DOPE potentials for said
                distance between first and second amino acids.
    """
    ca_matrix = []

    with open(filename, "r", encoding='UTF-8') as dope:
        for ligne in dope:
            if ligne[3:7].strip() == "CA" and ligne[11:14].strip() == "CA":
                ca_matrix.append(ligne.split())

    columns = ["res1", "temp1", "res2", "temp2"] + list(
        np.arange(0.25, 15, 0.5)
    )
    dope_score = pd.DataFrame(ca_matrix, columns=columns)
    dope_score = dope_score.drop(["temp1", "temp2"], axis=1)

    return dope_score


def get_fasta_sequence(filename):
    """Retrieve sequence from FASTA file.

    Parameters
    ----------
    filename : str
        FASTA file.

    Returns
    -------
    str
        Obtained sequence.
    """
    sequence = ""
    with open(filename, "r", encoding='UTF-8') as fasta:
        for ligne in fasta:
            if ligne.startswith(">"):
                continue
            sequence += ligne.strip()
    return sequence


if __name__ == "__main__":
    # Séquence 'query'
    FASTA_FILE = "../data/5AWL.fasta"
    QUERY = get_fasta_sequence(FASTA_FILE)

    # Structure 'template'
    PDB_FILE = "../data/5awl.pdb"
    TEMPLATE = Template(PDB_FILE)

    # Matrice DOPE
    DOPE_FILE = "../data/dope.par"
    DOPE_MATRIX = clean_dope_data(DOPE_FILE)

    # Information(s) supplémentaire(s)
    GAP = 0

    # Algorithme principal
    HIGH_LEVEL = HighLevelMatrix(GAP, QUERY, TEMPLATE, DOPE_MATRIX)
    MAX_SCORE = HIGH_LEVEL.fill_matrix()
    print(MAX_SCORE)
    ALIGN_SEQ, ALIGN_STRUCT = HIGH_LEVEL.get_alignment()
    print(ALIGN_SEQ)
    print(ALIGN_STRUCT)
