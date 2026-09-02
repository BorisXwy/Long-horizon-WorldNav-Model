"""GigaWorld-style navigation ablation on the local Wan2.1-1.3B backbone.

The package is deliberately separate from ``nav.v1``: it is an ablation that
keeps GigaWorld-Policy's state/reference/action/future token ordering and
causal policy path, while using the Wan2.1-1.3B checkpoint already available
on the server and the project's discrete R2R action ontology.
"""

from .config import GigaNavConfig
from .data import GigaNavR2RBatchBuilder, GigaNavDataConfig
from .model import GigaNavModel

__all__ = ["GigaNavConfig", "GigaNavDataConfig", "GigaNavModel", "GigaNavR2RBatchBuilder"]
