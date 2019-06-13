import pandas as pd
import numpy as np
import csv
import os
import sys


dirname = sys.argv[1]
atom_features = ['molecule', 'atom_index', 'element', 'x', 'y', 'z', 'atom_type', 'charge']
bond_features = ['molecule', 'bond_index', 'atom_0', 'atom_1', 'bond_type']

with open('atoms.csv', 'w', newline='') as atoms_csv:
    writer = csv.DictWriter(atoms_csv, atom_features)
    writer.writeheader()
    
with open('bonds.csv', 'w', newline='') as bonds_csv:
    writer = csv.DictWriter(bonds_csv, bond_features)
    writer.writeheader()


nwritten = 1
for filename in os.listdir(dirname):
    with open(dirname + filename) as f:
        text = f.read()

    lines = text.split('\n')

    molecule = os.path.basename(filename).split('.')[0]

    n_structs = lines[2].split()
    n_atoms = int(n_structs[0])
    n_bonds = int(n_structs[1])

    atom_begin = lines.index('@<TRIPOS>ATOM') + 1
    with open('atoms.csv', 'a', newline='') as atoms_csv:
        writer = csv.DictWriter(atoms_csv, atom_features)
        for i in range(atom_begin, atom_begin + n_atoms):
            atom_info = lines[i].split()

            idx = atom_info[0]
            elt = atom_info[1]
            x = atom_info[2]
            y = atom_info[3]
            z = atom_info[4]
            atom_type = atom_info[5]
            charge = atom_info[8]
            atom_vals = [molecule, idx, elt, x, y, z, atom_type, charge]

            writer.writerow(dict(zip(atom_features, atom_vals)))

            
    bond_begin = lines.index('@<TRIPOS>BOND') + 1
    with open('bonds.csv', 'a', newline='') as bonds_csv:
        writer = csv.DictWriter(bonds_csv, bond_features)
        for i in range(bond_begin, bond_begin + n_bonds):
            bond_info = lines[i].split()

            idx = bond_info[0]
            atom_0 = bond_info[1]
            atom_1 = bond_info[2]
            bond_order = bond_info[3]
            bond_vals = [molecule, idx, atom_0, atom_1, bond_order]

            writer.writerow(dict(zip(bond_features, bond_vals)))
    
    if nwritten % 10000 == 0:
        print(f'Files written: {nwritten}')
    nwritten += 1
