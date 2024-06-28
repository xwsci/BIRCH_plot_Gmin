# README

## Dependencies
Please ensure you have the following Python libraries installed:
- matplotlib
- adjustText
- numpy
- pandas
- scikit-learn

You can install the dependencies using the following command:
```bash
pip install matplotlib adjustText numpy pandas scikit-learn

## Usage
Run the following command to execute the code:
python3 clustering-Gmin.py

The PCA and BIRCH methods are implemented in clustering-Gmin.py. This script reads data from 'data-summary.csv' and 'new.csv' to cluster high- and low-performing nanoclusters. You can create your own 'new.csv' file to input your data. The columns should be organized as follows:
- System label
- ΔG_min
- Partial charge of M1
- Partial charge of M2
- Partial charge of M3
- Partial charge of O
- Partial charge of Cu
- Spin density of M1
- Spin density of M2
- Spin density of M3
- Spin density of O
- Spin density of Cu
- d-band center of M1
- d-band center of M2
- d-band center of M3
- d-band center of O
- d-band center of Cu
- Spin up bandgap
- Spin down bandgap
- Minimum of spin up and down bandgaps
- Bond length of M1-O
- Bond length of M2-O
- Bond length of M3-O
- Bond angle of M1-O-M2
- Bond angle of M1-O-M3
- Bond angle of M2-O-M3
- Dihedral angle of M1-M2-M3-O
- Number of d-electrons of free-standing M1 atom
- Number of d-electrons of free-standing M2 atom
- Number of d-electrons of free-standing M3 atom
