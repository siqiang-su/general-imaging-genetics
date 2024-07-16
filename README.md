# A General framework of Brain Region Detection and Genetic Variants Selection in Imaging Genetics
This repository includes the Python code for the paper `A General framework of Brain Region Detection and
Genetic Variants Selection in Imaging Genetics`.
## Citations
Su, S. *, Li, Z.*, Feng, L.† and Li, T. (2024+). "A General Framework of Brain Region Detection and Genetic Variants Selection in Imaging Genetics"
## Environment and usage
- Python 3.9
- 
To install it,
```
pip install -i https://test.pypi.org/simple/ GeneralCCA==0.0.1
```
The detailed usage refer to the **Examples.ipynb**

## Description

The `simulation` directory houses the primary scripts and functions required for conducting simulation studies. The `Alternating_Minimization.py` file encompasses the alternating minimization algorithm and a function for hyper-parameter selection using a modified BIC criterion. The `Joint_Generation.py` file contains functions for generating simulation data. The true signal shapes utilized in the associated paper are stored as `1-block.npy`, `3-block.npy`, and `butterfly.npy` within this directory.

The `example` directory includes a Python script that guides you through the process of data generation, hyper-parameter tuning, model fitting, and model evaluation.




