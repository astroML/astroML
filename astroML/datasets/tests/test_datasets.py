import pytest
from astroML.datasets import fetch_great_wall


@pytest.mark.remote_data
def test_fetch_great_wall():
    data = fetch_great_wall()
    assert data.shape == (8014, 2)
