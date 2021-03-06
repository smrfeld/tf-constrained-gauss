# Examples for tfConstrainedGauss

These are example notebooks for the `tfConstrainedGauss` package.

* [example_n3_me.ipynb](example_n3_me.ipynb) is a simple example for the MaxEnt-based solution for a `3x3` matrix.
* [example_n8_me.ipynb](example_n8_me.ipynb) is a larger example for the MaxEnt-based solution for a `8x8` matrix.
* [example_n8_id.ipynb](example_n8_id.ipynb) is the same example for the Identity-based solution.

## 3x3 MaxEnt-based example

<img src="figures/n3_me_constraints.png" alt="drawing" width="800"/>

* Left: precision matrix structure, where white indicates zero elements.
* Right: covariance matrix constraints.

<img src="figures/n3_me_cov.png" alt="drawing" width="800"/>

* Left: initial starting guess for the algorithm.
* Middle: target covariance matrix constraints.
* Right: Learned constraints after solving.

<img src="figures/n3_me_cov_full.png" alt="drawing" width="800"/>

* Left: learned precision matrix.
* Right: full learned covariance matrix (inverse of left), including elements such as `(0,2)` which are constrained by the precision (rather than covariance) matrix.

## 8x8 MaxEnt-based example

<img src="figures/n8_me_constraints.png" alt="drawing" width="800"/>

* Left: precision matrix structure, where white indicates zero elements.
* Right: covariance matrix constraints.

<img src="figures/n8_me_cov.png" alt="drawing" width="800"/>

* Left: initial starting guess for the algorithm.
* Middle: target covariance matrix constraints.
* Right: Learned constraints after solving.

<img src="figures/n8_me_cov_full.png" alt="drawing" width="800"/>

* Left: learned precision matrix.
* Right: full learned covariance matrix (inverse of left), including elements such as `(0,7)` which are constrained by the precision (rather than covariance) matrix.

## 8x8 Identity-based example

<img src="figures/n8_id_constraints.png" alt="drawing" width="800"/>

* Left: precision matrix structure, where white indicates zero elements.
* Right: covariance matrix constraints.

<img src="figures/n8_id_cov.png" alt="drawing" width="800"/>

* Left: initial starting guess for the algorithm.
* Middle: target covariance matrix constraints.
* Right: Learned constraints after solving.

<img src="figures/n8_id_prec.png" alt="drawing" width="400"/>

* Learned precision matrix.
