import os

import numpy as np



# P. Pyykkö, M. Atsumi,Chemistry – A European Journal2009,15, 186–197
covalence_radii_single_bond = {
    "H": 0.32,
    "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.66, "F": 0.64, 
    "S": 1.03, "P": 1.11, "Cl": 0.99, 
    "Br": 1.14, 
    "I": 1.33
}


element_symbols = {
    1: "H", 2: "He",
    3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    15: "P"
}



class Structure:

    def __init__(self):
        self.num_atoms = 0
        self.elems = None
        self.coords = None
        self.bond_dict = None
    

    def set_structure(self, coords: np.array, elems: list=None, atomic_numbers: np.array=None) -> None:
        """
        Initialize structure from arrays

        Args:
            coords, float, shape (num_atoms, 3): Cartesian atomic coordinates
            elems, str, shape (num_atoms,): Element symbols
            atomic_numbers, int, shape (num_atoms,): Nuclear charges
        """

        # safety check
        if elems is None and atomic_numbers is None:
            print()
            print("*********************** ERROR *************************")
            print("STRUCTURE MODULE: No elements nor atomic numbers given.")
            print("*******************************************************")
            print()
            return
        
        # if atomic numbers and not elems passed convert
        if atomic_numbers is not None and elems is None:
            elems = []
            for atomic_number in atomic_numbers:
                elems.append(element_symbols[atomic_number])

        # store XYZ information
        self.elems = elems
        self.coords = coords
        self.num_atoms = len(elems)

        # compute connectivty
        self.__compute_graph_as_dict()
        

    def set_structure_from_xyz_file(self, filepath: str) -> None:
        """
        Read standard XYZ file

        Args:
            filepath, str: Path of XYZ file
        """
        
        # safety check
        if not os.path.isfile(filepath):
            print()
            print("******************* ERROR *********************")
            print("STRUCTURE MODULE: File not found at given path.")
            print("***********************************************")
            print()
            return
        
        # read XYZ information
        self.elems = list(np.loadtxt(filepath, skiprows=2, usecols=0, dtype=str))
        self.coords = np.loadtxt(filepath, skiprows=2, usecols=(1,2,3))
        self.num_atoms = len(self.elems)

        # compute connectivty
        self.__compute_graph_as_dict()
    

    def __compute_graph_as_dict(self) -> None:
        """
        Compute molecular graph, i.e. connectivity of atoms, as dictionary
        """

        # store connectivity as dictionary
        self.bond_dict = {i: [] for i in range(self.num_atoms)}

        # loop over atom pairs, avoid redundant checks
        for i in range(self.num_atoms):
            for j in range(i+1, self.num_atoms):
                bond = self.__check_for_bond(
                    self.elems[i],
                    self.coords[i],
                    self.elems[j],
                    self.coords[j]
                )
                if bond:
                    self.bond_dict[i].append(j)
                    self.bond_dict[j].append(i)
    

    def __check_for_bond(self, elem1: str, coord1: np.array, elem2: str, coord2: np.array) -> bool:
        """
        Check if two atoms are bonded based on element-specific covalence radii

        Args:
            elem1, str: Element symbol of first atom
            coord1, float, shape (3,): Cartesian coordinates of first atom
            elem2, str: Element symbol of second atom
            coord2, float, shape (3,): Cartesian coordinates of second atom
        Returns:
            True if bonded, False if not
        """

        # bond criterium with small tolerance term yields better results
        tol = 0.08

        # actual interatomic distance
        d = np.linalg.norm(coord1 - coord2)

        # maximum interatomic distance (plus tolerance) for a bond to exist given both atom types
        d_ref = covalence_radii_single_bond[elem1] + covalence_radii_single_bond[elem2] + tol

        # if actual distance is smaller than reference distance there is a bond
        bond = False
        if d <= d_ref:
            bond = True

        return bond
