# C-SECSI
Minimal implementation for the **SE**mi-algebraic framework for the approximate coupled **C**P decompositions via **SI**multaneous Matrix Diagonalizations (SECSI) [1].


C-SECSI is a framework for the coupled semi-algebraic CP-decomposition of three-way multidimensional arrays (tensors).

 * **C-SECSI** provides an enhanced performance compared to other solvers in difficult scenarios, e.g., correlated factors
 * **C-SECSI** offers complexity-accuracy tradeoff by selecting appropriate heuristics


Note: this is my personal minimal implementation of the framework, which highlights the use of the 'REC PS' heuristic. 


# Dependencies
The sources for C-SECSI are self-contained. C-SECSI operates on conventional multi-dimensional arrays as provided by MATLAB.


# References
[1] A. Manina, M. Grasis, L. Khamidullina, A. Korobkov, J. Haueisen, and M. Haardt, “Coupled CP Decomposition of EEG and MEG Magnetometer and Gradiometer Measurements via the Coupled SECSI Framework,” in 2021 55th Asilomar Conference on Signals, Systems, and Computers, 2021, pp. 1661–1667.

[2] F. Roemer and M. Haardt, “A Semi-Algebraic Framework for Approximate CP Decompositions via Simultaneous Matrix Diagonalizations (SECSI),” Signal Processing, vol. 93, no. 9, pp. 2722–2738, 2013.


# Citation
If you use this code as part of any published research, please acknowledge the following paper.

```
@inproceedings{manina2021coupled,
author = {Manina, Alla and Grasis, Mikus and Khamidullina, Liana and Korobkov, Alexey and Haueisen, Jens and Haardt, Martin},
booktitle = {2021 55th Asilomar Conference on Signals, Systems, and Computers},
doi = {10.1109/IEEECONF53345.2021.9723118},
isbn = {978-1-6654-5828-3},
month = {oct},
pages = {1661--1667},
publisher = {IEEE},
title = {{Coupled CP Decomposition of EEG and MEG Magnetometer and Gradiometer Measurements via the Coupled SECSI Framework}},
url = {https://ieeexplore.ieee.org/document/9723118/},
year = {2021}
}

```
