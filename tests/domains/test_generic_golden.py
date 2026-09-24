"""Monthly behaviour of the generic adapter is byte-identical to the goldens written before daily,
weekly and seasonal support existed (us-unemployment config on a synthetic frame)."""

from __future__ import annotations

import pytest
from generic_golden import VARIANTS, load, run_variant


@pytest.mark.parametrize("tag", list(VARIANTS))
def test_unemployment_matches_golden(tag):
    assert run_variant(tag) == load(tag)
