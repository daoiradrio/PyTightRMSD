import numpy as np

from queue import Queue
from PyTightRMSD.structure import Structure
from typing import Optional



class TightRMSD:

    def __init__(self):
        pass


    def calc(
        self,
        coords1: np.array,
        coords2: np.array,
        file1: Optional[str]=None,
        file2: Optional[str]=None,
        elems1: Optional[list]=None,
        elems2: Optional[list]=None,
        atomic_numbers1: Optional[np.array]=None,
        atomic_numbers2: Optional[np.array]=None,
    ) -> float:
        '''
        Compute tight RMSD of two XYZ files

        Args:
            file1, str: Path to XYZ file of first molecular structure
            file2, str: Path to XYZ file of second molecular structure
        Returns:
            rmsd, float: Tight RMSD
        '''
        
        # safety checks and compute molecular graphs
        mol1 = Structure()
        mol2 = Structure()
        if file1 is not None and file2 is not None:
            mol1.set_structure_from_xyz_file(file1)
            mol2.set_structure_from_xyz_file(file2)
        elif elems1 is not None and elems2 is not None:
            mol1.set_structure(elems=elems1, coords=coords1)
            mol2.set_structure(elems=elems2, coords=coords2)
        elif atomic_numbers1 is not None and atomic_numbers2 is not None:
            mol1.set_structure(atomic_numbers=atomic_numbers1, coords=coords1)
            mol2.set_structure(atomic_numbers=atomic_numbers2, coords=coords2)
        else:
            print()
            print("********* ERROR ***********")
            print("RMSD MODULE: Invalid input.")
            print("***************************")
            print()
            return
        
        # reorder coordinates of second molecule to assign matching atom pairs
        coords1, coords2 = self.__match_coords(mol1, mol2)

        # align structures as close as possible (minimze RMSD)
        Rmat = self.__kabsch(coords1, coords2)
        coords2 = np.dot(coords2, Rmat)

        # calculate RMSD
        rmsd = float(np.sqrt(np.mean(np.linalg.norm(coords1 - coords2, axis=1)**2)))

        return rmsd


    def __kabsch(self, coords1: np.array, coords2: np.array) -> np.array:
        '''
        Rotation that minimizes RMSD between two molecular structures by aligning them

        Args:
            coords1, float, shape (num_atoms x 3): Cartesian coordinates of first molecule
            coords2, float, shape (num_atoms x 3): Cartesian coordinates of second molecule
        Returns:
            R, float, shape (3 x 3): Rotation matrix for second molecule to align it with first molecule
        '''

        # center structures
        center1 = np.mean(coords1, axis=0)
        center2 = np.mean(coords2, axis=0)
        coords1 -= center1
        coords2 -= center2

        # Kabsch algorithm, compute rotation matrix
        H = np.matmul(coords1.T, coords2)
        U, _, Vt = np.linalg.svd(H)
        det = np.linalg.det(np.matmul(Vt.T, U.T))
        if det >= 0:
            det = 1.0
        else:
            det = -1.0
        matrix = np.array([
            [1, 0,  0 ],
            [0, 1,  0 ],
            [0, 0, det]
        ])
        R = np.matmul(np.matmul(Vt.T, matrix), U.T)

        return R


    def __spheres(self, connectivity: dict, elems: list, start_atom: int) -> list:
        '''
        Compute coordination spheres (n bonds aways) of given atom

        Args:
            connectivity, dict: Connectivity of molecule
            elems, list: Atom types
            start_atom, int: Index of atom to start graph traversal at
        Returns:
            spheres, list: Coordination spheres of start atom, each sphere is list of atom types
        '''

        # initialize atoms queue
        atoms = Queue()

        # memory to prevent endless traversal, add start atom 
        memory = [start_atom]

        # add first sphere (= direct neighoring atoms of start atom) to queue and memory
        for atom in connectivity[start_atom]:
            atoms.put(atom)
            memory.append(atom)
        
        # coordination spheres of atom in question
        spheres = []
        
        # index of current sphere
        sphere_idx = 0

        # size sphere evaluated in a given iteration, to detect when current spheres ends and next one should start
        sphere_counter = len(connectivity[start_atom])

        # count size of next sphere (= neigbhors of atoms in sphere of given iteration)
        next_sphere_counter = 0

        # loop until all atoms have been visited once, i.e. until all spheres got collected
        while not atoms.empty():
            
            # storage for atoms in sphere of current iteration
            spheres.append([])

            # add atoms to sphere, sphere counter indicates end of current sphere
            while sphere_counter:
                
                # add atom types instead of indices for unambiguous comparison of spheres of two structures
                atom_idx = atoms.get()
                spheres[sphere_idx].append(elems[atom_idx])

                sphere_counter -= 1

                # add neighbors of neighbor in current sphere to next sphere
                for neighbor_idx in connectivity[atom_idx]:
                    
                    # but only if that neighbors has not been visited yet
                    if not neighbor_idx in memory:

                        memory.append(neighbor_idx)
                        atoms.put(neighbor_idx)

                        # count atoms in next sphere
                        next_sphere_counter += 1

            # sort atoms in current sphere by element symbols, again for unambiguous comparison
            spheres[sphere_idx] = sorted(spheres[sphere_idx])

            sphere_idx += 1
            sphere_counter = next_sphere_counter
            next_sphere_counter = 0

        return spheres


    def __match_coords(self, mol1: Structure, mol2: Structure) -> list:
        '''
        Reorder second set of coordinates to assign matching atom pairs

        Args:
            mol1, Structure: Structure instance of first molecule
            mol2, Structure: Structure instance of second molecule
        Returns:
            matched_coords1, float, shape (num_atoms x 3): Reorderd atomic coordinates of first molecule
            matched_coords2, float, shape (num_atoms x 3): Reorderd atomic coordinates of second molecule
        '''

        # initialization
        pairs = {}
        matched_coords1 = []
        matched_coords2 = []
        #assigned1 = np.array([False for _ in range(mol1.num_atoms)])
        #assigned2 = np.array([False for _ in range(mol2.num_atoms)])

        # compute coordination spheres for every atom in both molecules
        spheres1 = [self.__spheres(mol1.bond_dict, mol1.elems, atom) for atom in range(mol1.num_atoms)]
        spheres2 = [self.__spheres(mol2.bond_dict, mol2.elems, atom) for atom in range(mol2.num_atoms)]

        # for each atom in molecule 1 check which atom in molecule 2 are equivalent
        # regarding atom type and coordination spheres
        for atom1 in range(mol1.num_atoms):
            
            # collect equivalent atoms in molecule 2 in list
            eqs = []
            
            for atom2 in range(mol2.num_atoms):
                
                # 1st criterion: atom types must match s
                if mol1.elems[atom1] != mol2.elems[atom2]:
                    continue

                # 2nd criterion: coordination spheres must match
                if spheres1[atom1] == spheres2[atom2]:
                    eqs.append(atom2)

            # multiple equivalent atoms
            if len(eqs) > 1:
                pairs[atom1] = np.array(eqs)

            # just one equivalent atom, atom pair can be assigned
            elif len(eqs) == 1:
                matched_coords1.append(mol1.coords[atom1])
                matched_coords2.append(mol2.coords[eqs[0]])
                #assigned1[atom1] = True
                #assigned2[eqs[0]] = True

            # if not at least one equivalent atom something must be off
            else:
                print()
                print("***************** WARNING ******************")
                print("RMSD MODULE: Molecules seem to be different.")
                print("********************************************")
                print()
                return mol1.coords, mol2.coords

        # symmetric case where no atom pair can be assigned based on first two criteria
        if len(matched_coords1) == 0:
            # pick first atom pair that matches with respect to atom type
            # this is completely arbitrary but works because of symmetry
            pick_elem = None
            for atom1 in range(mol1.num_atoms):
                for atom2 in range(mol2.num_atoms):
                    if mol1.elems[atom1] == mol2.elems[atom2]:
                        matched_coords1.append(mol1.coords[atom1])
                        matched_coords2.append(mol2.coords[atom2])
                        #assigned1[atom1] = True
                        #assigned2[atom2] = True
                        pick_elem = mol1.elems[atom1]
                        break
                if pick_elem is not None:
                    break
        
        matched_coords1 = np.array(matched_coords1)
        matched_coords2 = np.array(matched_coords2)

        # check if all atom pairs assigned already
        if matched_coords1.shape[0] == mol1.num_atoms and matched_coords2.shape[0] == mol2.num_atoms:
            return matched_coords1, matched_coords2
        
        # check all unassigned atoms
        for atom, eq_atoms in pairs.items():
            
            # skip if atom already got assigned in symmetric case above
            #if assigned1[atom]:
            #    continue
            
            # calculate distances with assigned atoms in molecule 1 for reference
            # shape (num_assigned_atoms x num_possibly_equivalent_atoms)
            d_ref = np.linalg.norm(matched_coords1 - mol1.coords[atom], axis=-1, keepdims=True)
            d_ref = np.repeat(d_ref, repeats=len(eq_atoms), axis=-1)

            # calculate distances with assigned atoms in molecule 2
            # shape (num_assigned_atoms x num_possibly_equivalent_atoms)
            d = np.linalg.norm(matched_coords2[:, None, :] - mol2.coords[eq_atoms], axis=-1)

            # pick atom from molecule 2 whose distances resemble the reference distances the best
            min_atom = np.argmin(np.linalg.norm(d_ref - d, axis=0))
            min_atom = eq_atoms[min_atom]

            #assigned1[atom] = True
            #assigned2[min_atom] = True

            # update assigned coordinates
            matched_coords1 = np.vstack((matched_coords1, mol1.coords[atom]))
            matched_coords2 = np.vstack((matched_coords2, mol2.coords[min_atom]))

        return matched_coords1, matched_coords2
