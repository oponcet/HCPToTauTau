# coding: utf-8

from __future__ import annotations

__all__ = ["trigger_object_matching", "IF_NANO_V9", "IF_NANO_V11"]

from typing import Any

from columnflow.util import maybe_import
from columnflow.columnar_util import ArrayFunction, deferred_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    threshold: float = 0.25,
    axis: int = 2,
) -> ak.Array:
    """
    Helper to check per object in *vectors1* if there is at least one object in *vectors2* that
    leads to a delta R metric below *threshold*. The final reduction is applied over *axis* of the
    resulting metric table containing the full combinatorics. When *return_all_matches* is *True*,
    the matrix with all matching decisions is returned as well.
    """
    #print(f"Vectors1: {vectors1[0,:]}")
    #print(f"Vectors2: {vectors2[0,:]}")
    # delta_r for all combinations
    dr = vectors1.metric_table(vectors2)
    #print(f"DeltaR: {dr[0,:]}")
    # check per element in vectors1 if there is at least one matching element in vectors2
    any_match = ak.any(dr < threshold, axis=axis)
    #print(f"any_match: {any_match[0,:]}")
    return any_match


@deferred_column
def IF_NANO_V9(self, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 9 else None


@deferred_column
def IF_NANO_V11(self, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 11 else None
