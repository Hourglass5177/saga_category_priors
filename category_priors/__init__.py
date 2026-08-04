"""Train-only category priors for SAGA instance post-processing.

The package deliberately keeps descriptive priors separate from validation-tuned
mapping coefficients.  Importing it has no CUDA or SAGA side effects.
"""

from .taxonomy import Taxonomy, load_taxonomy

__all__ = ["Taxonomy", "load_taxonomy"]
__version__ = "0.1.0"
