"""Tests for the NowGoal context parsers + realtime odds leg.

Covers the server-rendered sections of the h2h / match-detail pages
(standings, fixtures, injuries, team stats, HT/FT, goal timing, lineups),
the enriched analysis rows (corners / HT / date / match id), the realtime
``r`` odds leg (fetch_live_odds + live snapshot in fetch_odds_history), and
the type=18 / type=22 structured AJAX (lineups, market splits).

All HTML/JSON is modeled on verified live captures (2026-08-15,
www.nowgoal.net match 2996109, FC Utrecht vs AZ Alkmaar) and all network
calls are mocked -- no live request is ever made.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.nowgoal import (  # noqa: E402
    NowGoalClient,
    _realtime_leg,
)


def _fixture(match_id="2996109"):
    return {"match_id": match_id, "home": "FC Utrecht", "away": "AZ Alkmaar",
            "kickoff": "2026-08-15T16:45:00Z"}


# ---- standings / fixtures / injuries (analysis-page sections) ------------

_STANDINGS_HTML = """
<h2 class="team-table-title">Standings</h2>
<div class="team-div">
  <div class='home-div'>
    <table class='team-table-home'>
      <tr class="team-home"><td colspan="10"><a>[HOL D1-12] FC Utrecht</a></td></tr>
      <tr><th>FT</th><th>Matches</th><th>Win</th><th>Draw</th><th>Lose</th>
          <th>Scored</th><th>Conceded</th><th>Pts</th><th>Rank</th><th>Rate</th></tr>
      <tr><td>Total</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>2</td>
          <td class='red'>0</td><td class='red'>12</td><td>0.0%</td></tr>
      <tr><td><span class='team-home-f'>Home</span></td><td>0</td><td>0</td><td>0</td><td>0</td>
          <td>0</td><td>0</td><td>0</td><td>8</td><td>0.0%</td></tr>
      <tr><td>Last 6</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>2</td>
          <td class='red'>0</td><td class='red'></td><td>0.0%</td></tr>
      <tr><th class='ht-desc'>HT</th><th>Matches</th><th>Win</th><th>Draw</th><th>Lose</th>
          <th>Scored</th><th>Conceded</th><th>Pts</th><th>Rank</th><th>Rate</th></tr>
      <tr><td>Total</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td>
          <td class='red'>1</td><td class='red'>16</td><td>0.0%</td></tr>
    </table>
  </div>
  <div class='guest-div'>
    <table class='team-table-guest'>
      <tr class="team-guest"><td colspan="10"><a>[HOL D1-3] AZ Alkmaar</a></td></tr>
      <tr><th>FT</th><th>Matches</th><th>Win</th><th>Draw</th><th>Lose</th>
          <th>Scored</th><th>Conceded</th><th>Pts</th><th>Rank</th><th>Rate</th></tr>
      <tr><td>Total</td><td>1</td><td>1</td><td>0</td><td>0</td><td>2</td><td>0</td>
          <td class='red'>3</td><td class='red'>3</td><td>100.0%</td></tr>
      <tr><th class='ht-desc'>HT</th><th>Matches</th><th>Win</th><th>Draw</th><th>Lose</th>
          <th>Scored</th><th>Conceded</th><th>Pts</th><th>Rank</th><th>Rate</th></tr>
      <tr><td>Total</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td>
          <td class='red'>1</td><td class='red'>9</td><td>0.0%</td></tr>
    </table>
  </div>
</div>
"""

_FIXTURES_HTML = """
<h2 class="team-table-title2"><span>Fixture (3 Matches)</span></h2>
<div class="team-div">
  <div class="home-div">
    <div class="team-table-home" style="margin-bottom:0px;"><a class="vv">FC Utrecht</a></div>
    <table class="team-table-home table-add-lh">
      <tr><th>League/Cup</th><th>Date</th><th>Type</th><th>VS</th><th>Countdown</th></tr>
      <tr><td title="Holland Eredivisie">HOL D1</td>
          <td><span name='timeData' data-t='2026-08-22 16:45:00' data-tf='30'></span></td>
          <td>Away</td><td>Sparta Rotterdam</td><td>7 Days</td></tr>
    </table>
  </div>
  <div class="guest-div">
    <div class="team-table-guest" style="margin-bottom:0px;"><a class="vv">AZ Alkmaar</a></div>
    <table class="team-table-guest table-add-lh">
      <tr><th>League/Cup</th><th>Date</th><th>Type</th><th>VS</th><th>Countdown</th></tr>
      <tr><td title="Holland Eredivisie">HOL D1</td>
          <td><span name='timeData' data-t='2026-08-22 16:45:00' data-tf='30'></span></td>
          <td>Home</td><td>Fortuna Sittard</td><td>7 Days</td></tr>
    </table>
  </div>
</div>
"""

_INJURIES_HTML = """
<div class="team-table-title">
  <span class="team-table-xq-home">FC Utrecht</span> Injury and Suspension
  <span class="team-table-xq-guest">AZ Alkmaar</span>
</div>
<div class="home-div" id="injuryH">
  <div class="player-list" onclick="toPlayerInfoPage(event)">
    <div playerid="162512" class="player-row"><b>CM</b><span>7</span><a>Victor Jensen</a></div>
    <div playerid="204677" class="player-row"><b>FC</b><span>&nbsp;</span><a>Noah Ohio</a></div>
  </div>
</div>
<div class="guest-div" id="injuryG">
  <div class="player-list" onclick="toPlayerInfoPage(event)">
    <div playerid="123" class="player-row"><b>CB</b><span>4</span><a>Suspended Player</a></div>
  </div>
</div>
"""


def test_parse_standings():
    out = NowGoalClient._parse_standings(_STANDINGS_HTML)
    assert out is not None
    home = out["home"]
    assert home["team"] == "FC Utrecht"
    assert home["league"] == "HOL D1"
    assert home["rank"] == 12
    assert home["ft"]["total"] == {
        "matches": 1, "win": 0, "draw": 0, "lose": 1, "scored": 1,
        "conceded": 2, "pts": 0, "rank": 12, "rate": "0.0%",
    }
    assert home["ht"]["total"]["rank"] == 16
    # empty Last-6 rank cell -> None, not junk
    assert home["ft"]["last_6"]["rank"] is None
    away = out["away"]
    assert away["team"] == "AZ Alkmaar"
    assert away["rank"] == 3
    assert away["ft"]["total"]["pts"] == 3


def test_parse_fixtures():
    out = NowGoalClient._parse_fixtures(_FIXTURES_HTML)
    assert out is not None
    assert out["home"][0] == {
        "league": "HOL D1", "date": "2026-08-22 16:45:00",
        "type": "Away", "opponent": "Sparta Rotterdam",
    }
    assert out["away"][0]["type"] == "Home"


def test_parse_injuries():
    out = NowGoalClient._parse_injuries(_INJURIES_HTML)
    assert out is not None
    assert out["home"][0] == {
        "player_id": "162512", "position": "CM", "number": "7",
        "name": "Victor Jensen",
    }
    assert out["home"][1]["number"] is None  # empty shirt number
    assert out["away"][0]["name"] == "Suspended Player"


def test_parse_analysis_enriched_rows():
    """Form/h2h rows now carry date, match id, HT score and corners."""
    rows = (
        '<tr id="tr1_1" vs="1" name="16" index="2991084" info="2,1,236,1,0">'
        "<td title='Holland Eredivisie'>HOL D1</td>"
        "<td><span name='timeData' data-t='2026-08-09 12:30:00'></span></td>"
        '<td><a onclick="soccerDbPage.team(236)"><span class="">Groningen</span></a></td>'
        '<td onclick="x"><span class="fscore_1 red2">2-1</span>'
        '<span class="hscore_1">(1-0)</span></td>'
        '<td><a onclick="soccerDbPage.team(237)"><span class="team-home-f">FC Utrecht</span></a></td>'
        '<td><span class="fcorner_1">5-1</span><span class="hcorner_1">(1-0)</span></td>'
        '<td class="hbg-td1"><span class=o-lose>L </span> </td></tr>'
    )
    html = f'<table id="table_v1"><tbody>{rows}</tbody></table>'
    out = NowGoalClient._parse_analysis(html, "FC Utrecht", "AZ Alkmaar")
    row = out["home_form"]["match_list"][0]
    assert row["date"] == "2026-08-09 12:30:00"
    assert row["match_id"] == "2991084"
    assert row["league_id"] == "16"
    assert row["score"] == "2-1"
    assert row["ht_score"] == "1-0"
    assert row["corners"] == "5-1"
    assert row["result"] == "L"


# ---- match detail page: team stats / HT/FT / goal timing / lineups -------

_TEAM_STATS_HTML = """
<table class="team-table-other">
  <tr><th>Home</th><th>Recent 3 Matches</th><th>Away</th>
      <th>Home</th><th>Recent 10 Matches</th><th>Away</th></tr>
  <tr><td class="">0.3</td><td><b>Goal</b></td><td class=" red">3</td>
      <td class="">1.2</td><td><b>Goal</b></td><td class="red">2.3</td></tr>
  <tr><td class="red">17</td><td><b>Fouls</b></td><td class="">9.3</td>
      <td class="red">12.4</td><td><b>Fouls</b></td><td class="">9.6</td></tr>
  <tr><td class="">50%</td><td><b>Possession</b></td><td class=" red">55.7%</td>
      <td class="">46.1%</td><td><b>Possession</b></td><td class="red">57.7%</td></tr>
</table>
"""


def test_parse_team_stats():
    out = NowGoalClient._parse_team_stats(_TEAM_STATS_HTML)
    assert out is not None
    assert out["Goal"] == {
        "home_recent3": 0.3, "away_recent3": 3.0,
        "home_recent10": 1.2, "away_recent10": 2.3,
    }
    assert out["Fouls"]["home_recent3"] == 17.0
    # "%" suffix stripped -> float
    assert out["Possession"]["away_recent10"] == 57.7


_HTFT_HTML = """
<h2 class="team-table-title">HT/FT Statistics (Last 2 Seasons)</h2>
<table class="team-table-other">
  <tr><th rowspan='2' class='rl'></th>
      <th colspan='2' class='rl home-m'>FC Utrecht ( 37 Matches)</th>
      <th colspan='2' class='guest-m'>AZ Alkmaar ( 35 Matches)</th></tr>
  <tr><th>Home</th><th>Away</th><th>Home</th><th>Away</th></tr>
  <tr><td class="rl">HT-W / FT-W</td><td>8</td><td class="rl">3</td><td>7</td><td>1</td></tr>
  <tr><td class="rl">HT-L / FT-L</td><td>2</td><td class="rl">7</td><td>2</td><td>5</td></tr>
</table>
"""


def test_parse_htft():
    out = NowGoalClient._parse_htft(_HTFT_HTML)
    assert out is not None
    assert out["home_team"] == "FC Utrecht"
    assert out["home_matches"] == 37
    assert out["away_matches"] == 35
    assert out["rows"]["HT-W/FT-W"] == {
        "home": {"home": 8, "away": 3},
        "away": {"home": 7, "away": 1},
    }


_GOAL_TIMING_HTML = """
<div id="rateOfScored1" class="rateOfScored">
  <div class="fx-comparision scoreComp ">
    <ul class="fx-data-left"><li class="hScoredLi hScoredLi1">
      <span class="fx-c-l home-bg" style="width: 15.2%"></span>
      <span class='fx-c2 '>19</span></li></ul>
    <span class="fx-c-3"><span>1~15</span></span>
    <ul class="fx-data-right"><li class="gScoredLi gScoredLi1">
      <span class="fx-c-r away-bg" style="width: 16%"></span>
      <span class='fx-c2 '>20</span></li></ul>
  </div>
  <div class="fx-comparision missComp">
    <ul class="fx-data-left"><li class="hScoredLi hScoredLi2">
      <span class="fx-c-l home-bg" style="width: 16%"></span>
      <span class='fx-c2 '>20</span></li></ul>
    <span class="fx-c-3 "><span>1~15</span></span>
    <ul class="fx-data-right"><li class="gScoredLi gScoredLi2">
      <span class="fx-c-r away-bg" style="width: 6.4%"></span>
      <span class='fx-c2 '>8</span></li></ul>
  </div>
  <div class="fx-comparision scoreComp ">
    <ul class="fx-data-left"><li class="hScoredLi hScoredLi1">
      <span class='fx-c2 '>6</span></li></ul>
    <span class="fx-c-3"><span>16~30</span></span>
    <ul class="fx-data-right"><li class="gScoredLi gScoredLi1">
      <span class='fx-c2 '>6</span></li></ul>
  </div>
  <div class="fx-comparision missComp">
    <ul class="fx-data-left"><li class="hScoredLi hScoredLi2">
      <span class='fx-c2 '>11</span></li></ul>
    <span class="fx-c-3"><span>16~30</span></span>
    <ul class="fx-data-right"><li class="gScoredLi gScoredLi2">
      <span class='fx-c2 '>5</span></li></ul>
  </div>
</div>
"""


def test_parse_goal_timing():
    out = NowGoalClient._parse_goal_timing(_GOAL_TIMING_HTML)
    assert out is not None
    b0 = out["last30"][0]
    assert b0["minutes"] == "1~15"
    assert b0["home_scored"] == 19
    assert b0["away_scored"] == 20
    assert b0["home_conceded"] == 20
    assert b0["away_conceded"] == 8
    assert out["last30"][1]["minutes"] == "16~30"
    assert out["last30"][1]["home_conceded"] == 11
    # no last-50 block in this fixture -> key absent
    assert "last50" not in out


_LINEUP_HTML = """
<div id="lineupBox">
  <div class="teamNames">
    <span class="tn-home home-bg"><a href="//football.nowgoal.net/team/summary/237">FC Utrecht</a>
      <span>3-4-2-1</span></span>
    <span class="tn-away away-bg"><span>4-2-3-1</span>
      <a href="//football.nowgoal.net/team/summary/234">AZ Alkmaar</a></span>
  </div>
  <div id="matchBox">
    <div class="plays ">
      <div class="home five">
        <div class="playBox"><div class='play' techWinId='117136'>
          <div class='number'>1</div><div class='name'>Barkas V.</div></div></div>
        <div class='play' techWinId='178261'>
          <div class='number'>44</div><div class='name'>Eerdhuijzen M.</div></div>
      </div>
      <div class="guest five">
        <div class="playBox"><div class='play' techWinId='223913'>
          <div class='number'>35</div><div class='name'>Meerdink M.</div></div></div>
      </div>
      <div class="home">
        <div class='play' techWinId='116550'>
          <div class='number'>25</div><div class='name'>Brouwer M.</div></div>
      </div>
      <div class="guest">
        <div class='play' techWinId='999999'>
          <div class='number'>99</div><div class='name'>Bench Keeper</div></div>
      </div>
    </div>
  </div>
</div>
"""


def test_parse_lineups():
    out = NowGoalClient._parse_lineups(_LINEUP_HTML)
    assert out is not None
    assert out["home_team"] == "FC Utrecht"
    assert out["away_team"] == "AZ Alkmaar"
    assert out["home_formation"] == "3-4-2-1"
    assert out["away_formation"] == "4-2-3-1"
    assert [p["name"] for p in out["lineups"]["home"]["starters"]] == [
        "Barkas V.", "Eerdhuijzen M."]
    assert [p["name"] for p in out["lineups"]["away"]["starters"]] == ["Meerdink M."]
    assert [p["name"] for p in out["lineups"]["home"]["bench"]] == ["Brouwer M."]
    assert [p["name"] for p in out["lineups"]["away"]["bench"]] == ["Bench Keeper"]


def test_parse_detail_bundle():
    html = (_TEAM_STATS_HTML + _HTFT_HTML + _GOAL_TIMING_HTML + _LINEUP_HTML)
    out = NowGoalClient._parse_detail(html, _fixture())
    assert out["source"] == "nowgoal_detail"
    assert "team_stats" in out
    assert "htft" in out
    assert "goal_timing" in out
    assert "lineups" in out
    assert out["match_id"] == "2996109"


# ---- realtime in-play leg (``r``) ----------------------------------------

_LIVE_MIXODDS = {
    "ErrCode": 0,
    "Data": {"mixodds": [
        {
            "cid": 8,
            "cn": "Bet365",
            "euro": {
                "f": {"u": "2.2", "g": "3.25", "d": "3.3"},
                "l": {"u": "2.25", "g": "2.7", "d": "4"},
                "r": {"u": "15", "g": "1.03", "d": "41"},
                "hr": True,
            },
            "ou": {
                "f": {"u": "0.8", "g": "2", "d": "1.05"},
                "l": {"u": "0.85", "g": "1.5", "d": "1"},
                "r": {"u": "5.6", "g": "0.5", "d": "0.11"},
                "hr": True,
            },
            "ah": {
                "f": {"u": "0.88", "g": "0.25", "d": "0.98"},
                "l": {"u": "0.88", "g": "0.25", "d": "0.98"},
                "r": {"u": "0.13", "g": "0", "d": "5"},
                "hr": True,
            },
        },
    ]},
}


def test_realtime_leg_extracts_r():
    item = _LIVE_MIXODDS["Data"]["mixodds"][0]["euro"]
    leg = _realtime_leg(item)
    assert leg == {"u": "15", "g": "1.03", "d": "41"}
    # a plain price dict (no wrapper) carries no realtime snapshot
    plain = {"u": "2.0", "g": "3.0", "d": "4.0"}
    assert _realtime_leg(plain) is None
    # wrapper without usable realtime -> nothing live
    assert _realtime_leg({"l": {"u": "2.0"}, "hr": False}) is None


def test_fetch_live_odds_uses_r_leg():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=_LIVE_MIXODDS)):
            payload = await client.fetch_live_odds(_fixture())
        assert payload is not None
        assert payload["is_live"] is True
        bm = payload["bookmakers"][0]
        markets = {m["key"]: m for m in bm["markets"]}
        assert [o["price"] for o in markets["h2h"]["outcomes"]] == [15.0, 1.03, 41.0]
        assert markets["totals"]["outcomes"][0]["point"] == 0.5
        assert markets["asian_handicap"]["outcomes"][0]["point"] == 0.0

    asyncio.run(runner())


def test_fetch_live_odds_none_when_no_realtime():
    data = {"ErrCode": 0, "Data": {"mixodds": [
        {"cid": 8, "cn": "Bet365", "euro": {"l": {"u": "2.2", "g": "3.2", "d": "3.4"}}},
    ]}}

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=data)):
            payload = await client.fetch_live_odds(_fixture())
        assert payload is None

    asyncio.run(runner())


def test_fetch_odds_history_has_live_snapshot_rows():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=_LIVE_MIXODDS)):
            out = await client.fetch_odds_history(_fixture())
        assert out is not None
        assert out["has_live"] is True
        live_rows = [m for m in out["markets"] if m.get("snapshot") == "live"]
        assert live_rows
        assert {r["market"] for r in live_rows} == {"h2h", "totals", "asian_handicap"}
        euro_live = [r for r in live_rows if r["market"] == "h2h"]
        prices = {r["selection"]: r["latest_price"] for r in euro_live}
        assert prices["Home"] == 15.0
        assert prices["Draw"] == 1.03

    asyncio.run(runner())


# ---- type=18 lineups + type=22 market splits -----------------------------

def test_fetch_lineups_parses_hlist_glist():
    data = {"ErrCode": 0, "Data": {"hList": [
        {"id": 117136, "name": "Barkas V.", "pName": "Goalkeeper", "no": 1,
         "valid": True, "rating": 6.5},
        {"id": 116550, "name": "Brouwer M.", "pName": "Goalkeeper", "no": 25,
         "valid": False, "rating": None},
    ], "gList": [
        {"id": 223913, "name": "Meerdink M.", "pName": "Forward", "no": 35,
         "valid": True, "rating": 7.1},
    ]}}

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=data)):
            out = await client.fetch_lineups(_fixture())
        assert out is not None
        assert out["home"][0]["starter"] is True
        assert out["home"][1]["starter"] is False
        assert out["home"][0]["rating"] == 6.5
        assert out["away"][0]["position"] == "Forward"

    asyncio.run(runner())


def test_fetch_market_splits_parses_groups():
    data = {"ErrCode": 0, "Data": {
        "AHAllSclass": {"Sum": 10, "Up": 4, "Draw": 2, "Down": 4},
        "OUMainSclass": {"Sum": 8, "Up": 5, "Draw": 0, "Down": 3},
        "OPAllSclass": {"Sum": 10, "Up": 4, "Draw": 3, "Down": 3},
    }}

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=data)):
            out = await client.fetch_market_splits(_fixture())
        assert out is not None
        assert out["groups"]["AH_All"] == {"sum": 10, "up": 4, "draw": 2, "down": 4}
        assert out["groups"]["OU_Main"] == {"sum": 8, "up": 5, "draw": 0, "down": 3}

    asyncio.run(runner())


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
