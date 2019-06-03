import numpy as np
from numpy.testing import assert_allclose
from astroML.correlation import uniform_sphere, ra_dec_to_xyz, angular_dist_to_euclidean_dist


def test_uniform_sphere():
	np.random.seed(42)

	# check number of points in 3 axis-aligned cones is approximately the same
	ra, dec = uniform_sphere((-180,180), (-90,90), 10000)
	x, y, z = ra_dec_to_xyz(ra, dec)

	assert_allclose(x ** 2 + y ** 2 + z ** 2, np.ones_like(x))

	in_x_cone = (y**2 + z**2 < 0.25).mean()
	in_y_cone = (x**2 + z**2 < 0.25).mean()
	in_z_cone = (x**2 + y**2 < 0.25).mean()

	# with prop > 0.999999 should not differ for more than 5 standard deviations
	assert_allclose(in_x_cone, in_y_cone, atol=5e-2)
	assert_allclose(in_x_cone, in_z_cone, atol=5e-2)
	assert_allclose(in_y_cone, in_z_cone, atol=5e-2)


def test_angular_d_to_euclidean_d():
	assert_allclose(angular_dist_to_euclidean_dist(180.), 2.)
	assert_allclose(angular_dist_to_euclidean_dist(60.), 1.)
	assert_allclose(angular_dist_to_euclidean_dist(0.), 0.)