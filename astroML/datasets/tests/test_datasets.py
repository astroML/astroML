import pytest
from astroML.datasets import fetch_great_wall
from astroML.datasets.tools.download import url_content_length
from astroML.datasets.tools.sdss_fits import SDSS_URL, SDSSfits, sdss_fits_url


@pytest.mark.remote_data
def test_fetch_great_wall():
    data = fetch_great_wall()
    assert data.shape == (8014, 2)


class _ResponseInfo:
    def __init__(self, headers):
        self._headers = headers

    def __getitem__(self, key):
        return self._headers[key.lower()]


class _ResponseHandle:
    def __init__(self, headers):
        self._info = _ResponseInfo(headers)

    def info(self):
        return self._info


def test_url_content_length_handles_case_insensitive_headers():
    handle = _ResponseHandle({'content-length': '123'})
    assert url_content_length(handle) == 123


def test_sdss_fits_url_uses_https():
    assert SDSS_URL.startswith('https://')
    assert sdss_fits_url(1, 2, 3).startswith('https://')


def test_sdssfits_loads_https_urls(monkeypatch):
    loaded = {}

    def fake_load_url(self, source):
        loaded['source'] = source

    def fake_load_file(self, source):
        raise AssertionError("HTTPS URLs should be treated as remote FITS files")

    monkeypatch.setattr(SDSSfits, '_load_fits_url', fake_load_url)
    monkeypatch.setattr(SDSSfits, '_load_fits_file', fake_load_file)

    SDSSfits('https://example.org/spec.fit')

    assert loaded['source'] == 'https://example.org/spec.fit'
