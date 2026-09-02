"""Wrong-team data post-mortem 2026-09-02.

Evidence (VPS analyses 31 Aug-1 Sep 2026, LiveScore-resolved fixtures): the
by-NAME LiveScore form rebuilt Southampton's window as L-D-L-L-D while the
club had really gone W-L-W-L-W, Stoke's as L-L-L-W-W (real W-L-L-D-L),
Portsmouth's as L-L-L-L-L (real L-L-W-L). Root cause: every provider matched
team names by SUBSTRING ("south" in "southampton", "stoke" in "basingstoke",
"port" in "portsmouth") with no reserve/youth/women guard, so other clubs'
results -- and their gf/ga -> attack/defence -> lambda -> pick -- were
attributed to the analysed team, even though verified provider ids for the
match were already known.

Each test encodes ONE general rule, never a per-club patch:

  I1  team_identity.names_match   token-level identity, marker asymmetry,
                                  qualifier agreement, <=1 extra token
  I2  match_side / same_fixture   an ambiguous name is never assigned a side
  I3  LiveScore form by TEAM ID   the per-event payload wins over any by-name
                                  path and cannot return another club
  I4  LiveScore by-name form      country + competition scope, strict identity
                                  outside the analysed league, no youth rows
  I5  NowGoal rows fail CLOSED    a row naming neither side is dropped, never
                                  scored as "home"
  I6  H2H window attribution      accent-safe, never zeroes a real record
  I7  tie state                   a different competition is never a first leg
  I8  settle Elo update           canonical-first key, never forks "Lille"
  I9  oddspapi fixture            orientation-locked + ambiguity-guarded
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.analyse import _teams_match  # noqa: E402
from agents.football.datasources import teams_match  # noqa: E402
from agents.football.elo import EloModel  # noqa: E402
from agents.football.livescore import apply_h2h_window, parse_form, team_form_by_id  # noqa: E402
from agents.football.nowgoal import NowGoalClient, _same_team as ng_same_team  # noqa: E402
from agents.football.oddspapi import OddspapiClient  # noqa: E402
from agents.football.team_identity import (  # noqa: E402
    Identity,
    has_marker,
    match_side,
    names_match,
    same_fixture,
)
from agents.football.tie_state import tie_state_from_h2h  # noqa: E402


# --------------------------------------------------------------------------
# I1 -- names_match
# --------------------------------------------------------------------------

INCIDENT_NEGATIVES = [
    ("Southampton", "South Carolina United FC"),      # VPS 1 Sep: form L-D-L-L-D from USL
    ("Stoke City", "Basingstoke"),
    ("Portsmouth", "Port City FC"),
    ("Birmingham City", "Birmingham City U18"),
    ("Birmingham City", "Birmingham City U21"),
    ("Parma", "Parma U20"),
    ("Ajax", "Jong Ajax"),
    ("Cadiz", "Cadiz B"),
    ("Rapid Wien", "Rapid Wien II"),
    ("Austria Wien", "Austria Wien Women"),
    ("Lincoln City", "Lincoln United"),
    ("Sheffield United", "Sheffield Wednesday"),
    ("Real Madrid", "Real Sociedad"),
    ("Manchester City", "Manchester United"),
    ("Paris FC", "Paris Saint-Germain"),
    ("Hapoel Tel Aviv", "Hapoel Beer Sheva"),
    ("Cadiz B", "Genclerbirligi"),
    ("Dep. A Coruna", "Las Palmas"),
    ("Dep. A Coruna", "Albacete"),
    ("Charlotte Independence", "Charlotte Independence 2"),
    ("Tobol (Kaz)", "Astana"),
    ("Athletic Club", "Atletico Madrid"),
    ("Barcelona", "Manchester City"),
]

LEGIT_POSITIVES = [
    ("Stoke", "Stoke City"),
    ("FK Bodø/Glimt", "Bodø/Glimt"),
    ("Bodo/Glimt", "Bodø/Glimt"),
    ("NK Celje", "Celje"),
    ("Royale Union Saint-Gilloise", "Union Saint-Gilloise"),
    ("Tobol (Kaz)", "Tobol Kostanay"),
    ("Partizan (Srb)", "FK Partizan Belgrade"),
    ("RFS (Lat)", "FC RFS"),
    ("Dyn. Kyiv (Ukr)", "FC Dynamo Kyiv"),
    ("Qarabag (Aze)", "Qarabag FK"),
    ("Ilves (Fin)", "Tampereen Ilves"),
    ("Rijeka (Cro)", "HNK Rijeka"),
    ("Flora (Est)", "Tallinna FC Flora"),
    ("Inter Escaldes (And)", "Inter Club de Escaldes"),
    ("Heart of Midlothian", "Hearts"),
    ("hearts", "heart of midlothian fc"),
    ("Bayern Munich", "FC Bayern München"),
    ("Beşiktaş", "Besiktas"),
    ("Sporting Clube de Braga", "SC Braga"),
    ("Atl. Madrid", "Atletico Madrid"),
    ("Copenhagen", "FC Copenhagen"),
    ("Jong Ajax", "Jong Ajax"),
    ("Inter", "Inter Milan"),
]


def test_names_match_rejects_every_incident_pair():
    bad = [(a, b) for a, b in INCIDENT_NEGATIVES if names_match(a, b) or names_match(b, a)]
    assert not bad, bad


def test_names_match_keeps_every_legitimate_pair():
    bad = [(a, b) for a, b in LEGIT_POSITIVES if not (names_match(a, b) and names_match(b, a))]
    assert not bad, bad


def test_strict_mode_allows_no_extra_identity_token():
    assert names_match("Lyon", "Lyon la Duchere")                 # tolerant: 1 extra token
    assert not names_match("Lyon", "Lyon la Duchere", strict=True)
    assert names_match("Tobol", "Tobol Kostanay")
    assert not names_match("Tobol", "Tobol Kostanay", strict=True)
    assert names_match("Stoke", "Stoke City", strict=True)          # qualifier only, no new identity


def test_identity_tokens_and_markers():
    assert Identity("FC Birmingham City U21").markers == {"youth"}
    assert Identity("SL Benfica B").markers == {"ii"}
    assert Identity("Rapid Wien II").markers == {"ii"}
    assert Identity("Austria Wien (W)").markers == {"women"}
    assert Identity("Ii").markers == set()          # Finnish club, lowercase "ii" is NOT a roman numeral
    assert has_marker("Jong PSV") and not has_marker("PSV Eindhoven")
    assert Identity("Dep. A Coruna").abbrevs == {"dep"}


def test_every_provider_matcher_delegates_to_the_shared_rule():
    for a, b in INCIDENT_NEGATIVES:
        assert not _teams_match(a, b), (a, b)
        assert not ng_same_team(a, b), (a, b)
        assert not teams_match(a, b), (a, b)
    for a, b in LEGIT_POSITIVES:
        assert _teams_match(a, b) and ng_same_team(a, b) and teams_match(a, b), (a, b)
    # the alias bridge in datasources.teams_match still resolves short codes
    assert teams_match("Man Utd", "Manchester United FC")


# --------------------------------------------------------------------------
# I2 -- side assignment is never ambiguous
# --------------------------------------------------------------------------

def test_match_side_refuses_ambiguous_and_unknown_names():
    assert match_side("Inter", "FC Inter Turku", "Inter Milan") is None      # both -> None
    assert match_side("Southampton", "Birmingham City", "Southampton") == "away"
    assert match_side("Everton", "Birmingham City", "Southampton") is None   # neither
    assert same_fixture("Stoke", "Norwich", "Norwich City", "Stoke City") == "reversed"
    assert same_fixture("Stoke", "Norwich", "Stoke City", "Norwich City") == "ordered"
    assert same_fixture("Inter", "Inter", "Inter Milan", "FC Inter Turku") is None  # both orders fit -> ambiguous


# --------------------------------------------------------------------------
# I3 -- LiveScore form by TEAM ID (never by side, never by name)
# --------------------------------------------------------------------------

def _ev(eid, home_id, home, away_id, away, tr1, tr2, esd, comp="Championship"):
    return {"Eid": eid, "T1": [{"ID": home_id, "Nm": home}], "T2": [{"ID": away_id, "Nm": away}],
            "Tr1": str(tr1), "Tr2": str(tr2), "Eps": "FT", "Esd": esd, "Stg": {"Snm": comp}}


def _form_payload():
    # T1 = Stoke (3005), T2 = Norwich (2811); both event lists newest first.
    stoke = [
        _ev("5", "2985", "Wolves", "3005", "Stoke City", 4, 1, 20260829190000),
        _ev("4", "3005", "Stoke City", "2871", "Hull City", 1, 1, 20260825190000),
        _ev("3", "2902", "Southampton", "3005", "Stoke City", 3, 1, 20260822190000),
        _ev("2", "3005", "Stoke City", "3917", "Swansea City", 1, 2, 20260815190000),
        _ev("1", "3005", "Stoke City", "9", "Oldham Athletic", 2, 0, 20260808190000, comp="EFL Cup"),
        _ev("0", "3005", "Stoke City", "8", "Valencia", 1, 1, 20260801190000, comp="Club Friendlies"),
    ]
    norwich = [
        _ev("15", "2811", "Norwich City", "2900", "Burnley", 4, 1, 20260829190000),
        _ev("14", "2870", "Cardiff City", "2811", "Norwich City", 1, 2, 20260825190000),
        _ev("13", "2860", "Millwall", "2811", "Norwich City", 3, 0, 20260822190000),
    ]
    return {"Eid": "1802367", "T1": [{"ID": "3005", "Nm": "Stoke City", "EL": stoke}],
            "T2": [{"ID": "2811", "Nm": "Norwich City", "EL": norwich}]}


def test_team_form_by_id_selects_the_team_by_id_and_scores_each_row_from_its_side():
    p = _form_payload()
    stoke = team_form_by_id(p, 3005)
    assert stoke["team_name"] == "Stoke City" and stoke["source"] == "livescore_event"
    # oldest -> newest, friendly dropped: W(2-0), L(1-2), L(1-3 away), D(1-1), L(1-4 away)
    assert stoke["sequence"] == "W-L-L-D-L"
    assert stoke["recent_goals"] == [(2, 0), (1, 2), (1, 3), (1, 1), (1, 4)]
    norwich = team_form_by_id(p, "2811")
    assert norwich["sequence"] == "L-W-W"                      # away 0-3, away 2-1, home 4-1
    assert norwich["recent_goals"] == [(0, 3), (2, 1), (4, 1)]
    assert team_form_by_id(p, "2902") is None                   # Southampton is not a side of this event
    assert team_form_by_id(p, None) is None and team_form_by_id(None, 3005) is None
    # side-agnostic: the id decides, T1/T2 order does not
    flipped = {"Eid": "x", "T1": p["T2"], "T2": p["T1"]}
    assert team_form_by_id(flipped, 3005)["sequence"] == "W-L-L-D-L"
    # limit trims the window and re-averages
    assert team_form_by_id(p, 3005, limit=2)["sequence"] == "D-L"
    assert parse_form(p)["home"]["sequence"] == "W-L-L-D-L"


class _LS:
    available = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def fetch_form(self, eid, limit=10):
        self.calls += 1
        assert eid == "1802367"
        return self.payload

    async def fetch_soccer_date(self, date8, page):  # never used on the id path
        raise AssertionError("by-name date feed must not be consulted when the event id is known")


def _fetcher(livescore):
    from agents.football.multi_source import MultiSourceStatsFetcher

    f = MultiSourceStatsFetcher.__new__(MultiSourceStatsFetcher)
    f.cache = None
    f.fc = None
    f.livescore = livescore
    return f


def test_form_chain_uses_the_event_id_before_any_by_name_path():
    ls = _LS(_form_payload())
    f = _fetcher(ls)
    meta = {"_league_key": "EFL Championship", "country": "England",
            "_livescore_match": {"source_id": "1802367", "home_id": "3005", "away_id": "2811"},
            "_team_names": {"3005": "Stoke City", "2811": "Norwich City"}}
    home = asyncio.run(f._fetch_team_form_uncached("3005", meta, 5))
    away = asyncio.run(f._fetch_team_form_uncached(2811, meta, 5))
    assert home["sequence"] == "W-L-L-D-L" and home["source"] == "livescore_event"
    assert away["sequence"] == "L-W-W" and away["team_id"] == "2811"
    assert ls.calls == 2


# --------------------------------------------------------------------------
# I4 -- by-name LiveScore form: geography + competition scope + no youth rows
# --------------------------------------------------------------------------

def _feed_row(sid, home, away, hg, ag, comp, country, ko):
    return {"Eid": sid, "T1": [{"ID": sid + "h", "Nm": home}], "T2": [{"ID": sid + "a", "Nm": away}],
            "Tr1": str(hg), "Tr2": str(ag), "Eps": "FT", "Esd": ko,
            "Stg": {"Snm": comp, "Cnm": country, "CompN": comp, "Sid": sid, "Scd": comp.lower()}}


class _LSFeed:
    available = True

    def __init__(self, rows_by_day):
        self.rows_by_day = rows_by_day

    async def fetch_soccer_date(self, date8, page):
        if page != 0:
            return {"Stages": []}
        rows = self.rows_by_day.get(date8, [])
        if not rows:
            return {"Stages": []}
        stages = []
        for r in rows:
            stg = r["Stg"]
            stages.append({"Sid": stg["Sid"], "Snm": stg["Snm"], "Cnm": stg["Cnm"], "CompN": stg["CompN"],
                           "Scd": stg["Scd"], "Ccd": stg["Cnm"].lower(), "Events": [r]})
        return {"Stages": stages}


def test_livescore_by_name_form_ignores_other_countries_youth_and_other_clubs():
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc)
    d = lambda back: (today - timedelta(days=back)).strftime("%Y%m%d")  # noqa: E731
    esd = lambda back: int((today - timedelta(days=back)).strftime("%Y%m%d") + "190000")  # noqa: E731
    rows = {
        d(1): [_feed_row("a1", "Southampton", "Millwall", 5, 1, "Championship", "England", esd(1)),
               _feed_row("a2", "South Carolina United FC", "Charlotte Independence 2", 0, 3, "USL League Two", "USA", esd(1))],
        d(3): [_feed_row("b1", "Watford", "Southampton", 2, 1, "Championship", "England", esd(3)),
               _feed_row("b2", "Southampton U21", "Reading U21", 0, 4, "Premier League 2", "England", esd(3))],
        d(5): [_feed_row("c1", "Southampton", "Stoke City", 3, 1, "Championship", "England", esd(5)),
               _feed_row("c2", "Southampton W", "Bristol City W", 0, 2, "Women's Championship", "England", esd(5))],
        d(8): [_feed_row("d1", "Colchester United", "Southampton", 0, 2, "EFL Cup", "England", esd(8))],
    }
    f = _fetcher(_LSFeed(rows))
    form = asyncio.run(f._livescore_form("Southampton", 5, lookback_days=10, league_key="EFL Championship", league_country="England"))
    # oldest -> newest: W, L, W -- the EFL Cup row belongs to another
    # registered competition and stays out of the LEAGUE window (G5), the
    # USL row is another country, the U21 / women rows are other teams.
    assert form["sequence"] == "W-L-W", form
    assert form["recent_goals"] == [(3, 1), (1, 2), (5, 1)]
    # overlapping feed pages (the same event on page 0 and page 1) count once
    class _Dup(_LSFeed):
        async def fetch_soccer_date(self, date8, page):
            return await super().fetch_soccer_date(date8, 0)

    f_dup = _fetcher(_Dup(rows))
    dup = asyncio.run(f_dup._livescore_form("Southampton", 5, lookback_days=10, league_key="EFL Championship", league_country="England"))
    assert dup["sequence"] == "W-L-W" and dup["sample_size"] == 3
    # the same feed for a USA club never borrows England rows
    f2 = _fetcher(_LSFeed(rows))
    usl = asyncio.run(f2._livescore_form("South Carolina United FC", 5, lookback_days=10, league_key="dyn:usl-league-two", league_country="USA"))
    assert usl["sequence"] == "L" and usl["recent_goals"] == [(0, 3)]


# --------------------------------------------------------------------------
# I5 -- NowGoal analysis rows fail closed
# --------------------------------------------------------------------------

def _ng_row(tid, idx, fh, fa, home, away, res, league="Championship"):
    cls = {"W": "win", "D": "draw", "L": "lose"}[res]
    return (
        f'<tr id="tr{tid}_{idx}" vs="1" name="1" index="{100 + idx}" info="{fh},{fa},1,0,0">'
        f'<td>{league}</td><td><span data-t=\'2026-08-2{idx} 19:00:00\'>x</span></td>'
        f'<td><a onclick="soccerDbPage.team(1)"><span>{home}</span></a></td>'
        f'<td><span class="fscore_1">{fh}-{fa}</span></td>'
        f'<td><a onclick="soccerDbPage.team(2)"><span>{away}</span></a></td>'
        f'<td class="hbg-td1"><span class="o-{cls}">{res}</span></td></tr>'
    )


def test_nowgoal_rows_naming_neither_side_are_dropped_not_scored_as_home():
    html = "".join([
        _ng_row(1, 1, 3, 1, "Southampton", "Millwall", "W"),          # home, gf 3 ga 1
        _ng_row(1, 2, 2, 1, "Watford", "Southampton", "L"),           # away, gf 1 ga 2
        _ng_row(1, 3, 6, 0, "South Carolina United", "Wake FC", "W"),  # neither side -> dropped
        _ng_row(1, 4, 4, 0, "Southampton U21", "Reading U21", "W"),   # youth -> dropped
    ])
    out = NowGoalClient._parse_analysis(html, "Southampton", "Birmingham City")
    form = out["home_form"]
    assert form["sequence"] == "W-L"
    assert form["recent_goals"] == [(1, 2), (3, 1)]          # oldest -> newest, away row scored from Southampton's side
    assert form["gf_avg"] == 2.0 and form["ga_avg"] == 1.5
    assert [r["team_perspective"] for r in form["match_list"]] == ["home", "away"]


# --------------------------------------------------------------------------
# I6 -- H2H window attribution is accent-safe and never zeroes a real record
# --------------------------------------------------------------------------

def test_apply_h2h_window_attributes_by_identity_and_keeps_provider_counts_on_miss():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=30)).strftime("%Y-%m-%dT19:00:00Z")
    h2h = {"wins": 1, "draws": 0, "losses": 1, "meetings": [
        {"home": "Bodø/Glimt", "away": "Rosenborg", "home_score": 2, "away_score": 0, "status": "finished", "kickoff": recent},
        {"home": "Rosenborg", "away": "Bodø/Glimt", "home_score": 1, "away_score": 0, "status": "finished", "kickoff": recent},
    ]}
    out = apply_h2h_window(dict(h2h), home_name="Bodo/Glimt", now=now)
    assert (out["wins"], out["draws"], out["losses"]) == (1, 0, 1)
    # a home name that matches NO meeting must not overwrite the tally with zeros
    out2 = apply_h2h_window(dict(h2h), home_name="Molde", now=now)
    assert (out2["wins"], out2["draws"], out2["losses"]) == (1, 0, 1)


# --------------------------------------------------------------------------
# I7 -- tie state: another competition is never a first leg
# --------------------------------------------------------------------------

def test_tie_state_ignores_meetings_from_a_different_competition():
    h2h = {"meetings": [{"home": "B", "away": "A", "home_score": 0, "away_score": 3, "status": "finished",
                         "kickoff": "2026-08-20T19:00:00Z", "competition": "Premier League"}]}
    assert tie_state_from_h2h(h2h, home="A", away="B", kickoff="2026-08-27T19:00:00Z", competition="Premier League")
    assert tie_state_from_h2h(h2h, home="A", away="B", kickoff="2026-08-27T19:00:00Z", competition="EFL Cup") is None
    assert tie_state_from_h2h(h2h, home="A", away="B", kickoff="2026-08-27T19:00:00Z")  # unknown -> tolerant
    youth = {"meetings": [{"home": "B U21", "away": "A U21", "home_score": 0, "away_score": 3, "status": "finished",
                           "kickoff": "2026-08-20T19:00:00Z"}]}
    assert tie_state_from_h2h(youth, home="A", away="B", kickoff="2026-08-27T19:00:00Z") is None


# --------------------------------------------------------------------------
# I8 -- settle Elo update lands on the canonical key
# --------------------------------------------------------------------------

def test_settle_elo_update_resolves_display_name_through_canonical(tmp_path):
    elo = EloModel(path=tmp_path / "elo.json")
    elo.ratings = {"Lille OSC": 2027.0, "Paris Saint-Germain": 2315.0}
    elo.games = {k: 30 for k in elo.ratings}
    elo._rebuild_indexes()
    # the display name alone is refused by the K2 guard (would fork a new key)
    assert elo.resolve("Lille") is None
    key = elo.resolve_first(("Lille OSC", "Lille"))
    assert key == "Lille OSC"
    before = elo.ratings["Lille OSC"]
    elo.update_from_results([{"home": key, "away": "Paris Saint-Germain", "home_goals": 2, "away_goals": 2}])
    assert "Lille" not in elo.ratings and elo.ratings["Lille OSC"] != before


# --------------------------------------------------------------------------
# I9 -- oddspapi fixture: orientation-locked and ambiguity-guarded
# --------------------------------------------------------------------------

class _Papi(OddspapiClient):
    def __init__(self, items):
        self._feed_cache = {}
        self._items = items

    async def _get(self, path, params=None):
        return self._items


def test_oddspapi_find_fixture_rejects_reversed_and_ambiguous_pairs():
    items = [
        {"fixtureId": 1, "hasOdds": True, "participant1Name": "Inter Turku", "participant2Name": "KuPS", "startTime": "2026-08-31T16:00:00Z"},
        {"fixtureId": 2, "hasOdds": True, "participant1Name": "Inter Milan", "participant2Name": "Cagliari", "startTime": "2026-08-30T18:45:00Z"},
        {"fixtureId": 3, "hasOdds": True, "participant1Name": "Cagliari", "participant2Name": "Inter", "startTime": "2026-08-30T18:45:00Z"},
    ]
    c = _Papi(items)
    # requested "Cagliari v Inter": fixture 2 is reversed -> not accepted; fixture 3 matches in orientation
    fx = asyncio.run(c.find_fixture("Cagliari", "Inter", kickoff="2026-08-30T18:45:00Z"))
    assert fx["fixtureId"] == 3
    # "Inter v KuPS": only the Turku fixture matches in orientation
    assert asyncio.run(c.find_fixture("Inter", "KuPS", kickoff="2026-08-31T16:00:00Z"))["fixtureId"] == 1
    # a pair that is a different match everywhere -> None
    assert asyncio.run(c.find_fixture("Southampton", "Stoke City", kickoff="2026-08-31T16:00:00Z")) is None


# --------------------------------------------------------------------------
# I10 -- context provenance: the flashscore url / event context must name THIS pair
# --------------------------------------------------------------------------

def test_flashscore_url_must_name_the_analysed_pair():
    from agents.football.team_identity import flashscore_url_matches

    lincoln = "https://www.flashscore.com/match/football/blackburn-6Nl8nagD/lincoln-city-hrHTRs5B/?mid=ABM8x0O9"
    stoke = "https://www.flashscore.com/match/football/norwich-Qo6off6p/stoke-city-hSUajdSS/?mid=2owrvDaE"
    assert flashscore_url_matches(lincoln, "Lincoln City", "Blackburn Rovers") == "reversed"  # slug order is not home-first
    assert flashscore_url_matches(lincoln, "Blackburn", "Lincoln") == "ordered"
    assert flashscore_url_matches(stoke, "Lincoln City", "Blackburn Rovers") is None          # another fixture's url
    assert flashscore_url_matches(None, "A", "B") is None
    # unreadable / placeholder slugs are "unknown" (fail-open), never a veto
    assert flashscore_url_matches("https://www.flashscore.com/team/lincoln-city/hrHTRs5B/", "Lincoln", "Blackburn") == "unknown"
    assert flashscore_url_matches("https://www.flashscore.com/match/football/a-AAAA1111/b-BBBB2222/?mid=1", "Arsenal", "Chelsea") == "unknown"


# --------------------------------------------------------------------------
# I11 -- by-name provider results are verified against the requested club
# --------------------------------------------------------------------------

def test_fallback_identity_rejects_other_club_and_reads_the_registry(tmp_path, monkeypatch):
    from agents.football import multi_source as ms
    from agents.football.entity_registry import EntityRegistry, canonical_team_id

    reg = EntityRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(ms, "_entity_registry", lambda: reg)
    ok = ms.MultiSourceStatsFetcher._fallback_identity_ok
    # both names canonicalise: same club -> ok, different club -> rejected
    assert ok("football_data", {"id": 1, "name": "Real Madrid CF"}, "Real Madrid", "LaLiga")
    assert not ok("football_data", {"id": 2, "name": "Real Madrid CF"}, "Atletico Madrid", "LaLiga")
    assert not ok("football_data", {"id": 3, "name": "RCD Espanyol de Barcelona"}, "Barcelona", "LaLiga")
    # the registry is READ at fetch time: an id already mapped to another club is rejected
    reg.register("thesportsdb", "999", "EPL", "Manchester City FC")
    assert not ok("thesportsdb", {"id": "999", "provider": "thesportsdb", "name": "Hull"}, "Hull City", "EPL")
    assert canonical_team_id("EPL", "Hull City") != reg.resolve("thesportsdb", "999")
    # no alias evidence: the names themselves must identify the same club
    assert ok("thesportsdb", {"id": "5", "name": "Tobol Kostanay"}, "Tobol", "dyn:kazakhstan")
    assert not ok("thesportsdb", {"id": "6", "name": "Lens"}, "Lorient", "dyn:x")
    assert not ok("thesportsdb", {"id": "7", "name": ""}, "Lorient", "dyn:x")


# --------------------------------------------------------------------------
# I12 -- ambiguous single-token names are REFUSED, never guessed
# --------------------------------------------------------------------------

def test_livescore_by_name_form_refuses_when_rows_come_from_two_clubs():
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc)
    d = lambda back: (today - timedelta(days=back)).strftime("%Y%m%d")  # noqa: E731
    esd = lambda back: int((today - timedelta(days=back)).strftime("%Y%m%d") + "190000")  # noqa: E731
    rows = {
        d(1): [_feed_row("m1", "Inter Milan", "Cagliari", 1, 0, "Serie A", "Italy", esd(1))],
        d(3): [_feed_row("m2", "FC Inter Turku", "KuPS", 2, 2, "Serie A", "Italy", esd(3))],  # same league/country on purpose
    }
    f = _fetcher(_LSFeed(rows))
    assert asyncio.run(f._livescore_form("Inter", 5, lookback_days=10, league_key="Serie A", league_country="Italy")) is None
    # a single club (both spellings) is not ambiguous
    rows2 = {d(1): [_feed_row("m1", "Inter Milan", "Cagliari", 1, 0, "Serie A", "Italy", esd(1))],
             d(3): [_feed_row("m3", "Inter", "Torino", 2, 0, "Serie A", "Italy", esd(3))]}
    f2 = _fetcher(_LSFeed(rows2))
    assert asyncio.run(f2._livescore_form("Inter", 5, lookback_days=10, league_key="Serie A", league_country="Italy"))["sequence"] == "W-W"


def test_flashscore_suggest_pickers_refuse_ambiguous_partial_hits():
    import json

    from agents.football.flashscore import _pick_sphinx_team, _pick_suggest_team, _squash_variants

    def item(name, slug, tid):
        return {"type": {"name": "Team"}, "sport": {"name": "Soccer"}, "name": name, "url": slug, "id": tid}

    sq = _squash_variants("Inter")
    # two different clubs as partial hits -> None; an exact entry wins outright
    assert _pick_sphinx_team([item("Inter Milan", "inter", "1"), item("FC Inter Turku", "inter-turku", "2")], sq) is None
    assert _pick_sphinx_team([item("Inter Milan", "inter-milan", "1"), item("Inter", "inter", "3")], sq) == ("inter", "3")
    # two spellings of ONE club are not ambiguous
    assert _pick_sphinx_team([item("Inter Milan", "inter-milan", "1"), item("FC Inter Milan", "inter-milan", "1")], sq) == ("inter-milan", "1")
    suggest = json.dumps([{"name": "Inter Milan", "url": "/team/inter-milan/AAAAAAAA"}, {"name": "FC Inter Turku", "url": "/team/inter-turku/BBBBBBBB"}])
    assert _pick_suggest_team(suggest, sq) is None
    suggest_one = json.dumps([{"name": "Inter Milan", "url": "/team/inter-milan/AAAAAAAA"}, {"name": "Inter", "url": "/team/inter/CCCCCCCC"}])
    assert _pick_suggest_team(suggest_one, sq) == ("inter", "CCCCCCCC")
