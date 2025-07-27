import pytest
import numpy as np
from astroML.datasets import (
    fetch_great_wall, fetch_dr7_quasar, fetch_sdss_galaxy_colors,
    fetch_LIGO_bigdog, fetch_LIGO_large, fetch_moving_objects,
    fetch_sdss_sspp, fetch_sdss_specgals, fetch_LINEAR_sample,
    fetch_LINEAR_geneva, fetch_nasa_atlas, fetch_sdss_spectrum,
    fetch_sdss_S82standards, fetch_imaging_sample, fetch_rrlyrae_mags,
    fetch_rrlyrae_combined, fetch_rrlyrae_templates, fetch_wmap_temperatures,
    fetch_sdss_galaxy_images, fetch_sdss_corrected_spectra,
    fetch_sdss_filter, fetch_vega_spectrum, generate_mu_z,
    simulation_kelly, fetch_hogg2010test
)


@pytest.mark.remote_data
def test_fetch_great_wall():
    """Test Great Wall dataset download and basic properties."""
    data = fetch_great_wall()
    assert data.shape == (8014, 2)
    assert np.all(np.isfinite(data))


@pytest.mark.remote_data
def test_fetch_dr7_quasar():
    """Test DR7 quasar catalog download and basic properties."""
    data = fetch_dr7_quasar()
    assert len(data) > 100000  # Should have > 100k quasars
    assert 'RA' in data.dtype.names
    assert 'dec' in data.dtype.names
    assert 'redshift' in data.dtype.names
    # Check RA range
    assert np.all((data['RA'] >= 0) & (data['RA'] <= 360))
    # Check dec range
    assert np.all((data['dec'] >= -90) & (data['dec'] <= 90))


@pytest.mark.remote_data
def test_fetch_sdss_galaxy_colors():
    """Test SDSS galaxy colors dataset."""
    data = fetch_sdss_galaxy_colors()
    assert len(data) > 600000  # Should have > 600k galaxies
    assert data.shape[1] == 7  # Should have 7 columns
    assert np.all(np.isfinite(data[data > -9999]))  # Check finite values (excluding -9999 flags)


@pytest.mark.remote_data
def test_fetch_LIGO_datasets():
    """Test LIGO gravitational wave datasets."""
    # Test big dog event
    data_bigdog = fetch_LIGO_bigdog()
    assert data_bigdog.shape[1] == 3  # time, H1, L1
    assert len(data_bigdog) > 1000
    
    # Test large dataset
    data_large = fetch_LIGO_large()
    assert data_large.shape[1] == 3  # time, H1, L1
    assert len(data_large) > len(data_bigdog)  # Large should be bigger than bigdog


@pytest.mark.remote_data
def test_fetch_moving_objects():
    """Test moving objects dataset."""
    data = fetch_moving_objects(Parker2008_cuts=True)
    assert len(data) > 10000  # Should have > 10k objects
    assert data.shape[1] == 11  # Should have 11 columns
    
    # Test without cuts
    data_no_cuts = fetch_moving_objects(Parker2008_cuts=False)
    assert len(data_no_cuts) >= len(data)  # No cuts should have more or equal data


@pytest.mark.remote_data
def test_fetch_sdss_sspp():
    """Test SDSS SSPP stellar parameters dataset."""
    data = fetch_sdss_sspp()
    assert len(data) > 300000  # Should have > 300k stars
    assert 'RA' in data.dtype.names
    assert 'dec' in data.dtype.names
    assert 'FeH' in data.dtype.names  # Metallicity


@pytest.mark.remote_data
def test_fetch_linear_samples():
    """Test LINEAR survey datasets."""
    # Test main LINEAR sample
    data_linear = fetch_LINEAR_sample()
    assert len(data_linear) > 7000  # Should have > 7k light curves
    assert 'RA' in data_linear.dtype.names
    assert 'dec' in data_linear.dtype.names
    
    # Test Geneva sample
    data_geneva = fetch_LINEAR_geneva()
    assert len(data_geneva) > 1400  # Should have > 1.4k objects
    assert 'RA' in data_geneva.dtype.names


@pytest.mark.remote_data
def test_fetch_nasa_atlas():
    """Test NASA Atlas dataset."""
    data = fetch_nasa_atlas()
    assert len(data) > 1700  # Should have > 1.7k objects
    assert 'apclass' in data.dtype.names
    assert 'a' in data.dtype.names  # Semi-major axis


@pytest.mark.remote_data
def test_fetch_sdss_spectrum():
    """Test SDSS spectrum download (single spectrum)."""
    # Test a known spectrum from the documentation examples
    spectrum = fetch_sdss_spectrum(plate=1615, mjd=53166, fiber=513)
    assert hasattr(spectrum, 'wavelength')
    assert hasattr(spectrum, 'spectrum')
    assert hasattr(spectrum, 'error')
    assert len(spectrum.wavelength()) > 1000  # Should have > 1k wavelength points


@pytest.mark.remote_data
def test_fetch_s82_standards():
    """Test SDSS S82 standards dataset."""
    data = fetch_sdss_S82standards()
    assert len(data) > 1000  # Should have > 1k standards
    assert 'RA' in data.dtype.names
    assert 'dec' in data.dtype.names


@pytest.mark.remote_data
def test_fetch_imaging_sample():
    """Test imaging sample dataset."""
    data = fetch_imaging_sample()
    assert len(data) > 600000  # Should have > 600k objects
    assert 'RA' in data.dtype.names
    assert 'dec' in data.dtype.names


@pytest.mark.remote_data
def test_fetch_rrlyrae_datasets():
    """Test RR Lyrae related datasets."""
    # Test magnitude data
    mags = fetch_rrlyrae_mags()
    assert len(mags) > 480  # Should have > 480 RR Lyrae
    assert 'RA' in mags.dtype.names
    
    # Test combined dataset
    combined = fetch_rrlyrae_combined()
    assert len(combined) >= len(mags)  # Combined should have at least as many as mags
    
    # Test templates
    templates = fetch_rrlyrae_templates()
    assert len(templates) > 90  # Should have > 90 templates
    assert 'phase' in templates.dtype.names


@pytest.mark.remote_data
def test_fetch_wmap_temperatures():
    """Test WMAP temperature data."""
    # Test unmasked
    data = fetch_wmap_temperatures(masked=False)
    assert hasattr(data, 'shape')
    assert len(data) > 3000000  # Should have > 3M pixels
    
    # Test masked
    data_masked = fetch_wmap_temperatures(masked=True)
    assert hasattr(data_masked, 'shape')


@pytest.mark.remote_data
def test_fetch_sdss_galaxy_images():
    """Test SDSS galaxy images dataset."""
    data = fetch_sdss_galaxy_images()
    assert len(data) > 4000  # Should have > 4k galaxy images
    assert 'RA' in data.dtype.names
    assert 'dec' in data.dtype.names


@pytest.mark.remote_data
def test_fetch_sdss_corrected_spectra():
    """Test SDSS corrected spectra dataset."""
    data = fetch_sdss_corrected_spectra()
    assert len(data) > 4000  # Should have > 4k spectra
    assert 'z' in data.dtype.names  # Redshift
    assert 'coeffs' in data.dtype.names  # PCA coefficients


@pytest.mark.remote_data
def test_fetch_sdss_filters():
    """Test SDSS filter and Vega spectrum."""
    # Test filter data
    filter_data = fetch_sdss_filter('g')
    assert len(filter_data) > 100  # Should have > 100 wavelength points
    assert filter_data.shape[1] == 2  # wavelength and transmission
    
    # Test Vega spectrum
    vega = fetch_vega_spectrum()
    assert len(vega) > 1000  # Should have > 1k wavelength points
    assert vega.shape[1] == 2  # wavelength and flux


# Non-remote data tests (these don't require internet)
def test_generate_mu_z():
    """Test mu-z relation generation (no remote data needed)."""
    z_sample, mu_sample, dmu = generate_mu_z(size=100)
    assert len(z_sample) == len(mu_sample) == len(dmu) == 100
    assert np.all(dmu > 0)  # Errors should be positive
    assert np.all(z_sample > 0)  # Redshifts should be positive


def test_simulation_kelly():
    """Test Kelly 2007 simulation (no remote data needed)."""
    result = simulation_kelly(size=100)
    assert len(result) == 8  # Returns 8-tuple
    xi_ksi, yi, xi_eta, sigma_x, sigma_y, sigma_xi, sigma_yi, rho_xy = result
    assert len(yi) == 100  # yi should have size elements
    # Just check that we get finite values
    assert np.all(np.isfinite(sigma_x))
    assert np.all(np.isfinite(sigma_y))


def test_fetch_hogg2010test():
    """Test Hogg 2010 test data (no remote data needed)."""
    data = fetch_hogg2010test()
    assert len(data) == 20  # Should have exactly 20 points
    assert len(data.dtype.names) == 6  # ID, x, y, sigma_x, sigma_y, rho_xy
    assert 'x' in data.dtype.names
    assert 'y' in data.dtype.names
