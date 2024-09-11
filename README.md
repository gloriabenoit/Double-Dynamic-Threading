# Protein sequence threading by double dynamic programming (M2BI)

## Introduction
This project aims to reimplement the method described by David Jones in 'THREADER: protein sequence threading
by double dynamic programming' (Computational Methods in Molecular Biology, Chapter 13, 1998). Some choices were made to 
simplify the implementation, such as using a gap penalty of 0, using the classic Needleman & Wunsch dynamic programming algorithm, or
using only the optimal score of a low level matrix rather than the sum of its optimal path.

## Setup

To install the algorithm and its dependencies, you need to perform the following steps:

### Clone the repository

```bash
git clone https://github.com/gloriabenoit/Double-Dynamic-Threading.git

cd Double-Dynamic-Threading
```

### Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create a Conda environment

```bash
conda env create -f environment.yml
```

### Activate the Conda environment

```bash
conda activate ddt
```

## Usage (command line interface)

```bash
python src/double_dynamic_threading.py template query1 ... queryN
```

## Data used

### Small proteins
To test my reimplementation, I have used 6 small proteins with a single chain :     
- [1BW6](https://www.rcsb.org/structure/1BW6), 56 residue, mainly helix     
- [1BA4](https://www.rcsb.org/structure/1BA4), 40 residue, mainly helix     
- [1AFP](https://www.rcsb.org/structure/1AFP), 51 residue, mainly strand     
- [1APF](https://www.rcsb.org/structure/1APF), 49 residue, mainly strand     
- [1AYJ](https://www.rcsb.org/structure/1AYJ), 51 residue, both strand and helix     
- [1BK8](https://www.rcsb.org/structure/1BK8), 50 residue, both strand and helix     

### Big proteins
The results obtained with small proteins have led me to believe the algorithm works better with bigger protein. However, it takes some time to compute
a high level matrix with bigger templates and sequences. Therefore, I have used only 4 big proteins :     
- [1A1W](https://www.rcsb.org/structure/1A1W), 91 residue, mainly helix     
- [1AB3](https://www.rcsb.org/structure/1AB3), 88 residue, mainly helix     
- [1AE2](https://www.rcsb.org/structure/1AE2), 87 residue, mainly strand     
- [1AB7](https://www.rcsb.org/structure/1AB7), 89 residue, both strand and helix          
