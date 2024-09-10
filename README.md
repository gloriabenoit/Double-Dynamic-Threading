# Protein sequence threading by double dynamic programming (M2BI)

## Introduction
This project aims to reimplement the method described by David Jones in 'THREADER: protein sequence threading
by double dynamic programming' ([Computational Methods in Molecular Biology](google.fr)). Some choices were made to 
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
python src/double_dynamic_threading.py data/template data/query1 ... data/queryN
```

## Data used

To test my reimplementation, I have used 6 proteins:
    - 1BW6, 56 residue, mainly helix
    - 1BA4, 40 residue, mainly helix
    - 1AFP, 51 residue, mainly strand
    - 1APF, 49 residue, mainly strand
    - 1AYJ, 51 residue, both strand and helix
    - 1BK8, 50 residue, both strand and helix
We expect each protein to fit best with itself and the other protein of the same main secondary structure.