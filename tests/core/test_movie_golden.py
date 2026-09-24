"""Golden regression: the refactored movie domain reproduces intel-1.1.0 (plus additive fields).

Intentional differences, filtered before comparing (docs/platform.md, implementation notes):
``early_warning_level`` decisions are added to ``decisions`` (and their batch to
``decision_batches``), so ``summary.counts.decisions``/``decisions_abstained`` grow by their count;
and each batch's ``state_hash`` changes, because the hashed input snapshot now carries the additive
fields of trends/anomalies/forecasts (the fingerprint of different bytes, not a behaviour change).

Goldens regenerated on 2026-09-24 for two intentional, measured core-1.1.0 policy changes
(docs/INTELLIGENCE_ENGINE_AUDIT.md), and nothing else (verified: with both switched off the new code
reproduces the old goldens exactly):
- the change-point AR(1) null takes phi from the series' history before the trend window
  (``change_point_history_*``): false alarms on the synthetic AR(1) study 9.9 % -> 1.3 % at a nominal
  1 %, so some weak change points (and their signals) are no longer reported;
- a warning raised from a risk carries the risk's ``confidence_kind`` (``evidence``/``rule``/``margin``)
  instead of a blanket ``margin``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from movie_golden import REAL_NOW, REAL_VARIANTS, diff, essentials, load, synthetic_variants

from jev_ml.core.pipeline import run_domain
from jev_ml.domains.movie import MovieAdapter
from jev_ml.intel import load_default_inputs, run_pipeline

ROOT = Path(__file__).resolve().parents[2]
EWL = "early_warning_level"


def comparable(d: dict) -> dict:
    d = copy.deepcopy(essentials(d))
    ewl = [x for x in d["decisions"] if x["key"] == EWL]
    d["decisions"] = [x for x in d["decisions"] if x["key"] != EWL]
    d["decision_batches"] = [b for b in d["decision_batches"] if b["name"] != "early_warning"]
    for b in d["decision_batches"]:
        b.pop("state_hash")
    c = d["summary"]["counts"]
    c["decisions"] -= len(ewl)
    c["decisions_abstained"] -= sum(x["abstained"] for x in ewl)
    return d


def strip_hash(g: dict) -> dict:
    for b in g["decision_batches"]:
        b.pop("state_hash")
    return g


@pytest.fixture(scope="module")
def synth():
    return synthetic_variants()


@pytest.mark.parametrize("tag", ["default", "as_of_2013-06-15", "seed1", "full"])
def test_movie_matches_golden_synthetic(synth, tag):
    golden = strip_hash(load(f"movie_synth_{tag}.json"))
    new = comparable(run_pipeline(synth[tag]).to_dict())
    problems = diff(golden, new)
    assert not problems, "\n".join(problems[:40])


def test_run_domain_equals_shim(synth):
    inp = synth["full"]
    a = run_pipeline(inp).to_dict()
    b = run_domain(MovieAdapter(inp), inp.as_of, inp.now, inp.suppressed_keys).to_dict()
    for d in (a, b):
        d["run"].pop("stage_ms")
    assert a == b


def test_every_warning_is_downstream_of_an_escalated_decision(synth):
    d = run_pipeline(synth["full"]).to_dict()
    dec = {x["id"]: x for x in d["decisions"]}
    assert d["warnings"]
    for w in d["warnings"]:
        x = dec[w["decision_id"]]
        assert x["key"] == EWL and x["answer"] in ("WARNING", "URGENT_ACTION")
        assert w["early_warning_level"] == x["answer"]
        # a high/critical component puts its situation at URGENT_ACTION; a medium one at least WARNING
        if w["severity"] in ("high", "critical"):
            assert x["answer"] == "URGENT_ACTION"
        comps = [c for c in x["state"]["components"] if c.get("key") == w["key"]]
        assert comps and comps[0]["warnable"] and comps[0]["points"] >= 3.0


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "interactions.csv").exists(), reason="no processed MovieLens data"
)
@pytest.mark.parametrize("tag", list(REAL_VARIANTS))
def test_movie_matches_golden_real(tag):
    golden = strip_hash(load(f"movie_real_{tag}.json.gz"))
    inputs = load_default_inputs(
        as_of=REAL_VARIANTS[tag],
        now=REAL_NOW,
        processed_dir=ROOT / "data" / "processed",
        models_dir=ROOT / "models",
        experiments_dir=ROOT / "experiments",
    )
    if inputs.dataset_meta and inputs.dataset_meta.get("dataset_version") != golden["run"]["data_version"]:
        pytest.skip("processed data differs from the golden's data version")
    new = comparable(run_pipeline(inputs).to_dict())
    problems = diff(golden, new)
    assert not problems, "\n".join(problems[:40])
