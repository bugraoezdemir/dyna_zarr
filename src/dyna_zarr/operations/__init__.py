"""operations: lazy array ops for DynamicArray, decentralized by the ome_zarr_pro
taxonomy locality axis. Each category module owns BOTH its transforms and its public op
functions; this package is a thin flat re-export hub. Adding an op touches exactly one
module.

    from dyna_zarr import operations as ops
    ops.add(a, b)                              # flat
    ops.gaussian_filter(x, 2.0)                # flat
    ops.neighborhood.gaussian_filter(x, 2.0)   # grouped access, also available

Modules by locality: structural (coordinate ops), pointwise (map_blocks + ufuncs),
neighborhood (map_overlap + filters), reductions (min/max, streaming reduce to come).
"""

from ._base import Transform, _is_int_index, _perm_on_surviving

# submodules kept importable for grouped access: operations.neighborhood.gaussian_filter, ...
from . import structural, pointwise, reductions, neighborhood, differences, creation

# ...and re-exported flat: operations.gaussian_filter, operations.add, ...
from .structural import *      # noqa: F401,F403
from .pointwise import *       # noqa: F401,F403
from .reductions import *      # noqa: F401,F403
from .neighborhood import *    # noqa: F401,F403
from .differences import *     # noqa: F401,F403
from .creation import *        # noqa: F401,F403

__all__ = (
    ["Transform", "structural", "pointwise", "reductions", "neighborhood",
     "differences", "creation"]
    + structural.__all__
    + pointwise.__all__
    + reductions.__all__
    + neighborhood.__all__
    + differences.__all__
    + creation.__all__
)
