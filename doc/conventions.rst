.. _conventions:

Conventions
############

Basis set conventions
=====================

Contractions
------------


Gaussian Primitives
-------------------

Cartesian
---------


Pure/Harmonic
-------------


Basis Set Ordering
------------------




.. math::
    \begin{align}
        r &= \sqrt{x^2 + y^2 + z^2}\\
        \theta &= \text{arctan2} \bigg(\frac{y}{x}\bigg)\\
        \phi &= arc\cos \bigg(\frac{z}{r}\bigg)
    \end{align}

such that when the radius is zero :math:`r=0`, then the angles are zero :math:`\theta,\phi = 0`.

Grid offers a :class:`utility<grid.utils.convert_cart_to_sph>` function to convert to spherical coordinates

.. code-block::
    python

    import numpy as np
    from grid.utils import convert_cart_to_sph

    # Generate random set of points
    cart_pts = np.random.uniform((-100, 100), size=(1000, 3))
    # Convert to spherical coordinates
    spher_pts = convert_cart_to_sph(cart_pts)
    # Convert to spherical coordinates, centered at [1, 1, 1]
    spher_pts = convert_cart_to_sph(cart_pts, center=np.array([1.0, 1.0, 1.0]))
