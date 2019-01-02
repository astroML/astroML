import warnings

from astroML.utils.exceptions import AstroMLDeprecationWarning
from astroML.utils.decorators import pickle_results


warnings.warn("'decorators' has been moved to 'astroML.utils' and will be "
              "removed from the main namespace in the future.",
              AstroMLDeprecationWarning)
