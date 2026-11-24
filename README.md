# Constant powered-distance sums characterize the regular simplex and the cross-polytope

Yitzchak Shmalo, Einstein Institute of Mathematics, The Hebrew University of Jerusalem.

## Abstract

Let n+1 points of R^n be such that the sums of the second and fourth powers of their distances to a point M do not
depend on M as M runs over a sphere. Nikolov and Rafailov conjectured that the points are then the vertices of a
regular simplex. They made the same conjecture for 2n distinct points, the powers 2, 4 and 6, and the cross-polytope.
We prove both conjectures in every dimension, also with positive weights, and show that for n+1 points the fourth
power alone forces a regular simplex. For the cross-polytope we show that a degree-3 cubature formula with positive
weights and 2n points for a centrally symmetric measure not supported on a hyperplane has its points in antipodal
pairs. We also show that the
corresponding characterization of the cube fails in every dimension n >= 3.

Archived at Zenodo: https://doi.org/10.5281/zenodo.24516799

## Contents

- `paper/main.tex`, `paper/main.pdf`: the paper.
- `code/nr_check.py`: the computer checks of Section 6 (exact rational arithmetic for the twisted cube in dimensions
  3, 4, 5; least-squares searches for the simplex and cross-polytope statements; controls).
- `code/nr_check_out.txt`: its output (Python 3, numpy 2.2.3, scipy 1.15.2; about 20 seconds).
- `SHA256SUMS`: checksums of the files above.

Run the checks with `python code/nr_check.py`.

The conjecture numbers follow arXiv:1211.2975v1 (N. Nikolov and R. Rafailov, On extremums of sums of powered
distances to a finite set of points; Geom. Dedicata 167 (2013) 69-89).
