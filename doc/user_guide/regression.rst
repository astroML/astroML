.. _astroML_regression:

Supervised Learning: Regression
===============================


Lets use simulated data for the examples below
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First we will model the distance of 100 supernovas (for a particular
cosmology as a function of redshift.

We rely on that astroML has a common API with scikit-learn, extending
the functionality of the latter.

.. plot::
  :include-source:
  :context: reset
  :nofigs:

  import numpy as np
  from astropy.cosmology import LambdaCDM
  from astroML.datasets import generate_mu_z
  z_sample, mu_sample, dmu = generate_mu_z(100, random_state=0)
  cosmo = LambdaCDM(H0=70, Om0=0.30, Ode0=0.70, Tcmb0=0)
  z = np.linspace(0.01, 2, 1000)
  mu_true = cosmo.distmod(z)


Simple linear regression
^^^^^^^^^^^^^^^^^^^^^^^^

Regression defined as the relation between a dependent variable, :math:`y`,
and a set of independent variables, :math:`x`, that describes the expectation
value of y given x: :math:`E[y|x]`.

We will start with the most familiar linear regression, a straight-line fit
to data. A straight-line fit is a model of the form :math:`y = ax + b` where
:math:`a` is commonly known as the slope, and :math:`b` is commonly known as
the intercept.

We can use Scikit-Learn's LinearRegression estimator to fit this data and
construct the best-fit line:

.. code:: python

    from sklearn.linear_model import LinearRegression as LinearRegression_sk
    linear_sk = LinearRegression_sk()
    linear_sk.fit(z_sample[:,None], mu_sample)
    mu_fit_sk = linear_sk.predict(z[:, None])

.. plot::
   :context:

   import numpy as np
   from matplotlib import pyplot as plt
   from sklearn.linear_model import LinearRegression as LinearRegression_sk
   from astropy.cosmology import LambdaCDM
   from astroML.datasets import generate_mu_z

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

Modifications to LinearRegression in astroML take measurement errors into
account on the dependent variable. The API is the same as for the
Scikit-Learn version above:

.. code:: python

    from astroML.linear_model import LinearRegression
    linear = LinearRegression()
    linear.fit(z_sample[:,None], mu_sample, dmu)
    mu_fit = linear.predict(z[:, None])

.. plot::
   :context:

   from astroML.linear_model import LinearRegression
   linear = LinearRegression()
   linear.fit(z_sample[:,None], mu_sample, dmu)
   mu_fit = linear.predict(z[:, None])

   #------------------------------------------------------------
   # Plot the results
   #fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111)
   ax.plot(z, mu_fit_sk, '-k')
   ax.plot(z, mu_fit, '-k', color='red')
   ax.plot(z, mu_true, '--', c='gray')
   ax.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)
   ax.set_xlim(0.01, 1.8)
   ax.set_ylim(36.01, 48)
   ax.set_ylabel(r'$\mu$')
   ax.set_xlabel(r'$z$')
   plt.show()


.. container:: binder-badge

  .. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/bsipocz/astroml.github.com/notebooks_regression?filepath=notebooks/astroml_regression_example.ipynb
    :width: 150 px


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


.. TODO: set a seed to we always get the same plot

.. plot::
   :context: reset
   :nofigs:

   np.random.seed(0)


.. plot::
  :nofigs:
  :context:
  :include-source:

  from astroML.datasets import simulation_kelly

  simulated_values = simulation_kelly(size=100, scalex=0.2, scaley=0.2,
                                      alpha=2, beta=1)
  ksi, eta, xi, yi, xi_error, yi_error, alpha_in, beta_in = simulated_values



.. TODO: fix API links

Now we take into account errors both on the dependent and independent
variables. The functionality is provided in the new class, ``LinearRegressionwithErrors``:

.. code:: python

  from astroML.linear_model import LinearRegressionwithErrors
  linreg_xy_err = LinearRegressionwithErrors()
  linreg_xy_err.fit(xi, yi, yi_error, xi_error)


Now plot the regression:

.. plot::
  :context:

   x0 = np.arange(-5, 5)
   y0 = alpha_in + beta_in[0] * x0

   from sklearn.linear_model import LinearRegression as LinearRegression_sk
   from astroML.linear_model import LinearRegression
   from astroML.linear_model import LinearRegressionwithErrors

   linreg_sk = LinearRegression_sk()
   linreg_sk.fit(xi[0][:, None], yi)
   linreg_sk_y_fit = linreg_sk.predict(x0[:, None])

   linreg = LinearRegression()
   linreg.fit(xi[0][:, None], yi, yi_error)
   linreg_y_fit = linreg.predict(x0[:, None])

   linreg_xy_err = LinearRegressionwithErrors()
   linreg_xy_err.fit(xi, yi, yi_error, xi_error)

   linreg_xy_err_y_fit = linreg_xy_err.coef_[0] + linreg_xy_err.coef_[1] * x0

   # Plot the results
   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111)
   ax.plot(x0, linreg_sk_y_fit, '-k', color='grey', label='sklearn, no errors')
   ax.plot(x0, linreg_y_fit, '-k', color='blue', label='astroML y errors only')
   ax.plot(x0, linreg_xy_err_y_fit, '-k', color='red', label='astroML x and y errors')
   ax.plot(x0, y0, '--', c='black')
   ax.errorbar(xi[0], yi, yi_error, xi_error[0], fmt='.k', ecolor='gray', lw=1)

   ax.set_ylabel(r'$y$')
   ax.set_xlabel(r'$x$')
   ax.legend()

   plt.show()


.. SANDBOX
..  plot_figure(ksi, eta, xi, yi, xi_error, yi_error, add_regression_lines=True, alpha_in=alpha_in, beta_in=beta_in)
..  plot_trace(linreg_xy_err, (xi, yi, xi_error, yi_error), ax=plt.gca(), chains=50)

.. TODO, link to https://scikit-learn.org/stable/modules/linear_model.html

.. container:: binder-badge

  .. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/bsipocz/astroml.github.com/notebooks_regression?filepath=notebooks/astroml_regression_example_with_errors.ipynb
    :width: 150 px
