"""Multi-source merge: provenance now carries WHICH CLUB, not only which
provider (wrong-team audit 2026-09-02, revision 3).

Before: ``merge_field`` compared VALUES only. Two sources agreeing on a form
window could both describe another club and still be reported
``confidence=HIGH``; a secondary source's value for another pair could win a
field the primary lacked. Now every ``FieldSample`` may carry ``entity``
(the pair it was fetched for) and the merge verifies identity BEFORE value:

  M1  a sample describing another pair is rejected (never wins, never agrees)
  M2  a reversed sample is swapped onto the analysed orientation
  M3  identity is read from the value when no entity is given (match names,
      H2H meetings, per-side team names)
  M4  canonical ids beat spellings
  M5  the LiveScore adapter never orients a fixture by guessing
  M6  end to end: the merged dataset exposes identity + rejections
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.datasources import (  # noqa: E402
    CONF_HIGH,
    FIELD_FORM,
    FIELD_H2H,
    FIELD_MATCH,
    FootballDataSource,
    aggregate_collected,
    available,
    merge_field,
    sample_identity,
)
from agents.football.livescore import LiveScoreDataSource  # noqa: E402

REF = {"home": "Southampton", "away": "Birmingham City", "kickoff": "2026-09-01T19:00:00Z"}


def _form(h, a):
    return {"home": {"sequence": h, "recent_goals": [(1, 0)], "gf_avg": 1.0, "ga_avg": 0.0, "sample_size": 1},
            "away": {"sequence": a, "recent_goals": [(0, 1)], "gf_avg": 0.0, "ga_avg": 1.0, "sample_size": 1}}


def test_m1_other_pair_is_rejected_and_never_agrees():
    samples = {
        "flashscore": available(_form("W", "L"), entity={"home": "Southampton", "away": "Birmingham City"}),
        "livescore": available(_form("W", "L"), entity={"home": "South Carolina United", "away": "Charlotte Independence 2"}),
    }
    fv = merge_field(FIELD_FORM, samples, priority={"flashscore": 100, "livescore": 80}, ref=REF)
    assert fv.source == "flashscore" and fv.identity == "verified"
    assert fv.agreement is None                       # the other pair never counts as a second source
    assert fv.confidence != CONF_HIGH
    assert [r["source"] for r in fv.identity_rejected] == ["livescore"]
    assert "South Carolina" in fv.identity_rejected[0]["reason"]
    # the rejected sample can never win either, even when it is the only one
    fv2 = merge_field(FIELD_FORM, {"livescore": samples["livescore"]}, ref=REF)
    assert fv2.status == "unavailable" and fv2.identity_rejected


def test_m2_reversed_sample_is_swapped_onto_the_analysed_orientation():
    samples = {"livescore": available(_form("W", "L"), entity={"home": "Birmingham City", "away": "Southampton"})}
    fv = merge_field(FIELD_FORM, samples, ref=REF)
    assert fv.identity == "reversed"
    assert fv.value["home"]["sequence"] == "L" and fv.value["away"]["sequence"] == "W"


def test_m3_identity_read_from_the_value_itself():
    # H2H meetings that name another pair -> reject; meetings of this pair -> verified
    bad = available({"wins": 3, "draws": 0, "losses": 0, "meetings": [
        {"home": "Stoke City", "away": "Norwich City", "home_score": 1, "away_score": 0}]})
    assert sample_identity(FIELD_H2H, bad, REF)[0] == "reject"
    good = available({"wins": 1, "draws": 0, "losses": 0, "meetings": [
        {"home": "Birmingham City", "away": "Southampton", "home_score": 0, "away_score": 2}]})
    assert sample_identity(FIELD_H2H, good, REF)[0] == "verified"
    # per-side team names inside the form value (livescore_event form)
    named = available({"home": {"team_name": "Southampton", "sequence": "W"}, "away": {"team_name": "Birmingham City", "sequence": "L"}})
    assert sample_identity(FIELD_FORM, named, REF)[0] == "verified"
    swapped = available({"home": {"team_name": "Birmingham City"}, "away": {"team_name": "Southampton"}})
    assert sample_identity(FIELD_FORM, swapped, REF)[0] == "reversed"
    wrong = available({"home": {"team_name": "Southampton U21"}, "away": {"team_name": "Birmingham City"}})
    assert sample_identity(FIELD_FORM, wrong, REF)[0] == "reject"
    # a match record of another fixture / reversed orientation is not this match
    assert sample_identity(FIELD_MATCH, available({"home": "Stoke City", "away": "Norwich City"}), REF)[0] == "reject"
    assert sample_identity(FIELD_MATCH, available({"home": "Southampton", "away": "Birmingham City"}), REF)[0] == "verified"
    # nothing to check -> unknown, accepted as before
    assert sample_identity(FIELD_FORM, available(_form("W", "L")), REF)[0] == "unknown"
    assert sample_identity(FIELD_FORM, available(_form("W", "L")), {})[0] == "unknown"


def test_m4_canonical_ids_beat_spellings():
    ref = dict(REF, home_cid="t:efl:southampton-fc", away_cid="t:efl:birmingham-city-fc")
    same_names_other_ids = available(_form("W", "L"), entity={
        "home": "Southampton", "away": "Birmingham City",
        "home_cid": "t:usl:south-carolina-united", "away_cid": "t:efl:birmingham-city-fc"})
    assert sample_identity(FIELD_FORM, same_names_other_ids, ref)[0] == "reject"
    crossed = available(_form("W", "L"), entity={
        "home": "Birmingham City", "away": "Southampton",
        "home_cid": "t:efl:birmingham-city-fc", "away_cid": "t:efl:southampton-fc"})
    assert sample_identity(FIELD_FORM, crossed, ref)[0] == "reversed"


def test_m5_livescore_adapter_never_orients_by_guessing():
    ds = LiveScoreDataSource.__new__(LiveScoreDataSource)
    fx = {"source_id": "1", "home": "Birmingham City", "away": "Southampton", "home_id": "2858", "away_id": "2902",
          "kickoff": "x", "status": "scheduled", "status_raw": "NS", "competition": "Championship",
          "country": "England", "category": None, "stage": None, "score": {"home": None, "away": None}}
    swapped = ds._orient(fx, REF)
    assert swapped["home"] == "Southampton" and swapped["home_id"] == "2902"
    assert ds._orient(fx, {"home": "Inter", "away": "Inter"}) is None          # ambiguous
    assert ds._orient(fx, {"home": "Stoke City", "away": "Norwich City"}) is None  # another pair


class _Secondary(FootballDataSource):
    name = "livescore"

    def __init__(self, fields):
        super().__init__()
        self._fields = fields

    async def fetch_fields(self, ref):
        return dict(self._fields)


def test_m6_end_to_end_metadata_exposes_identity_and_rejections():
    primary = {
        FIELD_MATCH: available({"home": "Southampton", "away": "Birmingham City", "kickoff": REF["kickoff"], "competition": "championship"},
                               entity={"home": "Southampton", "away": "Birmingham City"}),
        FIELD_FORM: available(_form("W", "L"), entity={"home": "Southampton", "away": "Birmingham City"}),
    }
    secondary = _Secondary({
        FIELD_FORM: available(_form("W", "L"), entity={"home": "Stoke City", "away": "Norwich City"}),   # other pair, same numbers
        FIELD_H2H: available({"wins": 2, "draws": 0, "losses": 0, "meetings": [
            {"home": "Norwich City", "away": "Stoke City", "home_score": 0, "away_score": 1}]}),          # other pair, no entity
    })
    unified = asyncio.run(aggregate_collected(
        primary_name="flashscore", primary_fields=primary, secondary=secondary, ref=REF,
        config={"enabled": ["flashscore", "livescore"], "priority": {"flashscore": 100, "livescore": 80},
                "validation": {"enabled": True}},
    ))
    d = unified.to_dict()
    form = unified.fields[FIELD_FORM]
    assert form.source == "flashscore" and form.agreement is None and form.confidence != CONF_HIGH
    assert d["source_metadata"]["form"]["identity"] == "verified"
    assert d["source_metadata"]["form"]["identity_rejected"] == ["livescore"]
    # the H2H of another pair does not fill the gap "for free"
    assert unified.fields[FIELD_H2H].status == "unavailable"
    assert d["source_metadata"]["h2h"]["identity_rejected"] == ["livescore"]
