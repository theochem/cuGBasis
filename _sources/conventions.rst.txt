.. _conventions:

Conventions
############

This following page outlines the basis-set conventions that cuGBasis uses to evaluate various quantities.


Gaussian Primitives
-------------------

CuGBasis supports both Cartesian and Spherical primitive Gaussian type orbitals.

Cartesian
=========

Normalized Cartesian primitive Gaussian with order :math:`(i, j, k)`
centered at position :math:`\mathbf{A} = (A_x, A_y, A_z)` is written as

.. math::
    b_j^C(x, y, z) = N (x - A_x)^i (y - A_y)^j (z - A_z)^k e^{-\alpha ||(x,y,z) - \mathbf{A}||^2},

where :math:`i, j, k` are positive integers, :math:`N` is the L2-normalization constant and :math:`\alpha` is the Gaussian width parameter.
The degree :math:`l` of a Cartesian primitive Gaussian is defined as :math:`l = i + j + k`.

Spherical/Pure
==============

Normalized spherical primitives Gaussians with degree :math:`l` and order :math:`m`
centered at position :math:`\mathbf{A} = (A_x, A_y, A_z)` is written as

.. math::
    b_j^S(r, \theta, \phi) = N ||r - \mathbf{A}||^l Y_{lm}(\theta, \phi) e^{-\alpha ||(x,y,z) - \mathbf{A}||^2}

where :math:`Y_{lm}` is the real spherical harmonics, :math:`N` is the L2-normalization constant, and :math:`\alpha` is the Gaussian width parameter.

CuGBasis uses the real regular solid harmonics, i.e. :math:`S_{lm} = r_A^l Y_{lm}`, since when :math:`l=0,1`, it matches its
Cartesian Gaussian primitives. Further, it uses the Cartesian representation for :math:`S_{lm}` in-order to avoid
the conversion from Cartesian :math:`(x, y, z)` to spherical coordinates :math:`(r, \theta, \phi)`.

For example, the regular solid harmonics for degree 2 is

.. math::
    \begin{align*}
        S_{2,0}(x, y, z) &= 0.5 (3z^2 - (x^2 + y^2 + z^2)) \\
        S_{2,1}(x, y, z) &= \sqrt{3}xz \\
        S_{2,-1}(x,y,z) &= \sqrt{3}yz \\
         \vdots
    \end{align*}



Contractions
------------
CuGBasis uses linear combination of primitives Gaussian as basis-functions (known as contracted Gaussian
type-orbitals or contractions):

.. math::
    \Phi_i^{K_i} = \sum_j d_{j}^i b_j^K(x, y, z; \alpha_j),

where :math:`K_i \in \{C, S\}` denotes whether it is Cartesian or Spherical, :math:`d_j` are the
contraction coefficients and :math:`b_j` is a primitive Gaussian with fixed parameters, except for :math:`\alpha_j`,
for all :math:`j`.  The contractions are not required to be normalized, and if they are, the contraction coeffcients
are assumed to contain the normalization constant.


Basis-set Groupings (Shells)
----------------------------

The grouping of all basis-functions/contractions :math:`\{\Phi_i^{K_i}: 0 \leq i \leq M \}` is the basis-set. However,
the contractions are grouped together more simply, called Generalized Contraction Shells, based on having
identical exponents :math:`\alpha`, and centers :math:`A`.

If they are instead grouped together, based on having identical :math:`\alpha`,
centers :math:`A`, :math:`K \in \{C, S\}`, and degree :math:`l`, then they are called Segmented Contraction Shells.

The following shows an example of a generalized contracted shell:

.. image:: contracted_shell_example.png
  :width: 800

|

The following shows an example of conversion to an segmented contracted shell:

.. image:: contracted_segmented_example.png
  :width: 800


Basis Set Ordering
------------------

CuGBasis follows the same basis-set ordering that Gaussian computational chemistry software uses.
Note that for spherical orders, the positive orders :math:`m\geq0` are associated with cosine function, denoted as c, and
the negative orders :math:`m<0` are associated with sine function, denoted as s. Thus "s21" implies
it is a solid harmonic with :math:`l=2` and :math:`m=-1`.

.. csv-table:: Basis-set Order for Cartesian and Spherical Orders
   :file: ./basis_set_orders.csv
   :widths: 30, 50, 50
   :header-rows: 1
   :class: longtable

