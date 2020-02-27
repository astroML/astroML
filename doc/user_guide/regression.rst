.. _astroML_regression:

Supervised Learning: Regression
===============================


Simple linear regression
^^^^^^^^^^^^^^^^^^^^^^^^

Regression defined as the relation between a dependent variable, :math:`y`,
and a set of independent variables, :math:`x`, that describes the expectation
value of y given x: :math:`E[y|x]`.

We will start with the most familiar linear regression, a straight-line fit
to data. A straight-line fit is a model of the form :math:`y = ax + b` where
:math:`a` is commonly known as the slope, and :math:`b` is commonly known as the intercept.

We can use Scikit-Learn's LinearRegression estimator to fit this data and
construct the best-fit line.

.. plot::

   import numpy as np
   from matplotlib import pyplot as plt
   from sklearn.linear_model import LinearRegression as LinearRegression_sk
   from astropy.cosmology import LambdaCDM
   from astroML.datasets import generate_mu_z
   z_sample, mu_sample, dmu = generate_mu_z(100, random_state=0)
   cosmo = LambdaCDM(H0=70, Om0=0.30, Ode0=0.70, Tcmb0=0)
   z = np.linspace(0.01, 2, 1000)
   mu_true = cosmo.distmod(z)
   linear_sk = LinearRegression_sk()
   linear_sk.fit(z_sample[:,None], mu_sample)
   mu_fit_sk = linear_sk.predict(z[:, None])
   #------------------------------------------------------------
   # Plot the results
   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111)
   ax.plot(z, mu_fit_sk, '-k')
   ax.plot(z, mu_true, '--', c='gray')
   ax.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)
   ax.set_xlim(0.01, 1.8)
   ax.set_ylim(36.01, 48)
   ax.set_ylabel(r'$\mu$')
   ax.set_xlabel(r'$z$')
   plt.show()


Measurement Errors in Linear Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Measurement errors in both dependent and independent variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use simulation data from `Kelly 2007
<https://iopscience.iop.org/article/10.1086/519947/pdf>`__  where
there is measurement error on the observed values :math:`x_i` and
:math:`y_i` as well as intrinsic scatter in the regression relationship:

.. math::
   :nowrap:

   \begin{gather*}
      \eta_{i} = \alpha + \beta xi_i + \epsilon_{i}\\
      x_i = xi_i + \epsilon_{x,i} \\
      y_i = \eta_i + \epsilon_{y,i} \\
   \end{gather*}
