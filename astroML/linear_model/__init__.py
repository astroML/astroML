import warnings

from .linear_regression import LinearRegression, PolynomialRegression, BasisFunctionRegression
try:
    from .linear_regression_errors import LinearRegressionwithErrors
except ImportError:
    warnings.warn('LinearRegressionwithErrors requires PyMC3 to be installed')
from .kernel_regression import NadarayaWatson
from .TLS import TLS_logL
