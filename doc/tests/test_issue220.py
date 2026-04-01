from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / 'astroML' / 'plotting' / 'settings.py'
EXAMPLE_PATH = ROOT / 'examples' / 'algorithms' / 'fig_volume_ratio.py'
EXAMPLE_BUILDER_PATH = ROOT / 'doc' / 'sphinxext' / 'example_builder.py'


def load_settings_module():
    spec = spec_from_file_location('astroML_plotting_settings', SETTINGS_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_text_plots_can_disable_usetex():
    matplotlib.rcdefaults()
    settings = load_settings_module()

    settings.setup_text_plots(fontsize=11, usetex=False)

    assert matplotlib.rcParams['text.usetex'] is False
    assert matplotlib.rcParams['font.size'] == 11


def test_docs_gallery_does_not_force_latex():
    assert "matplotlib.rc('text', usetex=False)" in EXAMPLE_BUILDER_PATH.read_text()
    assert "setup_text_plots(fontsize=8, usetex=False)" in EXAMPLE_PATH.read_text()
