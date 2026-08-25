"""Team alias resolver. Maps short codes / nicknames to canonical names.

Each league's teams.json maps alias -> canonical team name. Lookup is
exact-match on alias (lowercased), with fallback to word-boundary match.

Word boundaries (not bare substrings) are used for the fallback so a short
code like "INT" does not match inside "saINT-Gilloise" and map Union
Saint-Gilloise to Inter Milan.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TEAMS_PATH = Path(__file__).parent / "teams.json"

# P0 (2026-08-24): club-name noise words stripped BEFORE token-containment
# comparisons, so "FC Barcelona" and bare "Barcelona" share the significant
# token set {barcelona} and containment can fire.
_NAME_STOPWORDS = {"fc", "cf", "sc", "ac", "fk", "rcd", "us", "cd", "ud", "de"}


def load_teams() -> dict[str, dict[str, str]]:
    return json.loads(TEAMS_PATH.read_text(encoding="utf-8"))


def _flatten(teams: dict[str, dict[str, str]] | None = None) -> dict[str, tuple[str, str]]:
    """Return {alias_lower: (league_key, canonical_name)}."""
    teams = teams if teams is not None else load_teams()
    idx: dict[str, tuple[str, str]] = {}
    for league_key, aliases in teams.items():
        for alias, canonical in aliases.items():
            idx[alias.lower()] = (league_key, canonical)
    return idx


def resolve_team_alias(query: str, league_key: str | None = None) -> str | None:
    """Resolve a short alias / nickname to canonical team name.

    If league_key is given, restrict search to that league. Returns the
    canonical name (which is then passed to API-Football / football-data
    for ID resolution).
    """
    if not query:
        return None
    q = query.strip().lower()
    teams = load_teams()
    idx = _flatten(teams)
    if q in idx:
        league, canonical = idx[q]
        if league_key is None or league == league_key:
            return canonical

    def _boundary_match(a: str, b: str) -> bool:
        """Word-boundary containment: a appears as a word inside b."""
        return re.search(rf"\b{re.escape(a)}\b", b) is not None

    # Exact canonical-name pass (ANY league): a full team name must resolve
    # to ITSELF before the fuzzy boundary fallback runs, otherwise a short
    # generic alias that is also a common word hijacks it -- the "madrid"
    # alias (-> Real Madrid) maps BOTH "Atlético Madrid" and "Rayo
    # Vallecano de Madrid" to Real Madrid (verified live 2026-08-17). The
    # pass iterates the RAW per-league maps, NOT the flattened index: short
    # codes collide across leagues in the flattened index ("LEI" is both
    # Leicester City and CS Marítimo) and the loser would hide the exact
    # canonical hit.
    #
    # F3 (2026-08-17, wrong-team bug): the comparison MUST be
    # accent-insensitive. The user query "Atletico Madrid" (ASCII) failed
    # the raw ``q == canonical.lower()`` test against the canonical
    # "Atlético Madrid" and fell through to the fuzzy word-boundary pass,
    # where the generic "MADRID" alias hijacked ANY name containing the
    # word "madrid" -> "Real Madrid CF". The bot then analyzed Real
    # Madrid vs Malaga for a query about Atlético (phantom fixture on the
    # flashscore team-fixtures fallback; nowgoal + flashscore league page
    # both showed Atl. Madrid vs Malaga). ``_abbr_key`` strips accents +
    # punctuation + case, so both spellings compare equal.
    for lg, aliases in teams.items():
        if league_key and league_key != lg:
            continue
        for alias, canonical in aliases.items():
            if _abbr_key(q) == _abbr_key(canonical) or q == alias.lower():
                return canonical

    # F3 (2026-08-17, wrong-team bug, second vector): the flashscore
    # STANDINGS spellings are the other side of the same hijack. A query /
    # resolved name that arrives as the standings abbreviation ("Atl.
    # Madrid", "Espanyol", "Dep. A Coruna") must map BACK to its canonical
    # club BEFORE the fuzzy word-boundary pass: "Atl. Madrid" contains the
    # word "madrid", so the generic "MADRID" alias turned it into "Real
    # Madrid CF" and the canonical-id / match_id of the REAL fixture became
    # t:laliga:real-madrid-cf (verified live 2026-08-17 on Atletico vs
    # Malaga). ``_STANDINGS_SPELLINGS`` is keyed by the canonical abbr_key
    # with the standings forms as values; reverse it so a standings spelling
    # resolves to its club.
    _rev = _standings_reverse_index()
    hit = _rev.get(_abbr_key(q))
    if hit:
        for _lg, _canonical in hit:
            if league_key is None or _lg == league_key:
                return _canonical
        # League-scoped lookup missed; accept the first pair as best-effort
        # only when the caller did not restrict the league.
        if league_key is None:
            return hit[0][1]

    # P0 (plan v3 follow-up, 2026-08-24): full-name containment beats generic
    # boundary hits. A provider spelling that CONTAINS a canonical name's
    # significant tokens ("Club Atletico de Madrid" ⊇ {atletico, madrid};
    # bare "Barcelona" = the significant token of "FC Barcelona") must resolve
    # to THAT club -- never to a generic single-word alias or to another club
    # whose longer alias key merely contains one of the query's words.
    # Verified live before this guard:
    #   "Club Atletico de Madrid" -> Real Madrid CF  (via the MADRID alias)
    #   "Barcelona"               -> RCD Espanyol de Barcelona (via its alias key)
    # Ambiguous queries containing NO complete canonical (bare "madrid") keep
    # the legacy passes below unchanged.
    if league_key and league_key in teams:
        _qt = set(_abbr_key(q).split()) - _NAME_STOPWORDS
        if _qt:
            _full: set[str] = set()
            for _canon in set(teams[league_key].values()):
                _ct = set(_abbr_key(_canon).split()) - _NAME_STOPWORDS
                if _ct and _ct <= _qt:
                    _full.add(_canon)
            if len(_full) == 1:
                return next(iter(_full))

    if league_key and league_key in teams:
        for alias, canonical in teams[league_key].items():
            al = alias.lower()
            if q == al or _boundary_match(al, q) or _boundary_match(q, al):
                return canonical
    # Cross-league exact match: when the team is not in the requested league
    # but IS in another league (e.g. Coventry in EFL Championship when user
    # queries EPL), resolve it. This prevents the fuzzy boundary match from
    # hijacking the query (e.g. 'coventry city' -> 'Manchester City FC' because
    # 'city' matches 'Manchester City').
    for alias, (league, canonical) in idx.items():
        if league_key and league == league_key:
            continue  # already checked above
        al = alias.lower()
        if q == al:
            return canonical
    # Fuzzy boundary match: only when no exact match found in any league.
    for alias, (league, canonical) in idx.items():
        if league_key and league != league_key:
            continue
        al = alias.lower()
        if _boundary_match(al, q) or _boundary_match(q, al):
            return canonical
    # F4 (2026-08-18, split-identity bug): a query that is a WORD inside
    # EXACTLY ONE canonical club name in the league resolves to that club
    # ("Lecce" -> "US Lecce"). Ambiguous words ("Madrid" -> Real Madrid /
    # Atlético Madrid / Rayo Vallecano de Madrid) are NEVER guessed: the
    # wrong mapping splits one real match across two match_ids and starves
    # the card's movement + statistical components (verified live 2026-08-17:
    # Palermo vs US Lecce was logged as BOTH "...||US Lecce||..." by the
    # odds poll and "...||Lecce||..." by the analyse run, so the pinned
    # opening snapshot never joined and movement read n/a). Comparison uses
    # ``_abbr_key`` (accents + punctuation stripped) so "Atletico" matches
    # "Atlético Madrid" exactly like the earlier passes.
    if league_key and league_key in teams:
        _q = _abbr_key(q)
        _cands = [
            canonical
            for canonical in set(teams[league_key].values())
            if _boundary_match(_q, _abbr_key(canonical))
        ]
        if len(_cands) == 1:
            return _cands[0]
    return None


# Unambiguous cross-league abbreviations -> canonical name. Token-containment
# matching cannot resolve "Man Utd" -> "Manchester United" (different tokens),
# so this map bridges the gap. Applied ONLY on exact normalized-token match so
# ambiguous short names ("Inter", "Paris", "City") are NEVER guessed here: a
# wrong mapping is worse than no mapping.
_UNAMBIGUOUS_ABBREVIATIONS = {
    "man utd": "Manchester United",
    "man utd fc": "Manchester United",
    "man united": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "psg": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
}


def canonical_abbreviation(name: str) -> str | None:
    """Canonical team name for an UNAMBIGUOUS abbreviation, else None."""
    if not name:
        return None
    q = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    q = " ".join(q.split())
    return _UNAMBIGUOUS_ABBREVIATIONS.get(q)


def _abbr_key(s: str) -> str:
    """Lowercase token string, accents + punctuation stripped (comparable)."""
    import unicodedata

    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


# Canonical (full) team name -> flashscore STANDINGS-table spellings that are
# NOT derivable from ``_UNAMBIGUOUS_ABBREVIATIONS`` or teams.json aliases.
# Flashscore standings tables render short forms ("Atl. Madrid", "Paris SG",
# "B. Monchengladbach", "Man Utd") that plain token containment cannot map
# back to the full name (verified live 2026-08-17 against the real EPL / LaLiga
# / Serie A / Bundesliga / Ligue 1 tables; the verified gaps are marked below).
# Keyed by ``_abbr_key`` (lowercase, accents + punctuation stripped). Values
# are the standings spellings in order of preference -- the first that
# normalizes onto a table row wins.
_STANDINGS_SPELLINGS: dict[str, tuple[str, ...]] = {
    # ---- EPL (verified against the live table) ----
    "manchester united fc": ("Manchester Utd", "Man Utd"),
    "manchester united": ("Manchester Utd", "Man Utd"),
    "manchester city fc": ("Man City",),
    "nottingham forest fc": ("Nottingham", "Nott'm Forest"),
    "nottingham forest": ("Nottingham", "Nott'm Forest"),
    "wolverhampton wanderers fc": ("Wolves",),
    "wolverhampton wanderers": ("Wolves",),
    "sheffield united fc": ("Sheff Utd",),
    "sheffield united": ("Sheff Utd",),
    "west ham united fc": ("West Ham",),
    "west ham united": ("West Ham",),
    "newcastle united fc": ("Newcastle",),
    "newcastle united": ("Newcastle",),
    "tottenham hotspur fc": ("Tottenham",),
    "tottenham hotspur": ("Tottenham",),
    "leicester city fc": ("Leicester",),
    "leeds united fc": ("Leeds",),
    "southampton fc": ("Southampton",),
    "brighton hove albion fc": ("Brighton",),
    "crystal palace fc": ("Crystal Palace",),
    # ---- LaLiga (verified against the live table) ----
    "atletico madrid": ("Atl. Madrid",),
    "athletic club": ("Ath Bilbao",),
    "athletic bilbao": ("Ath Bilbao",),
    "real betis balompie": ("Betis",),
    "rcd espanyol de barcelona": ("Espanyol",),
    "real sociedad de futbol": ("Real Sociedad",),
    "rc celta de vigo": ("Celta Vigo",),
    "rayo vallecano de madrid": ("Rayo Vallecano",),
    "deportivo la coruna": ("Dep. A Coruna",),
    "deportivo de la coruna": ("Dep. A Coruna",),
    # ---- Serie A (verified against the live table) ----
    "fc internazionale milano": ("Inter",),
    "hellas verona fc": ("Hellas Verona",),
    "us salernitana 1919": ("Salernitana",),
    "empoli fc": ("Empoli",),
    # ---- Bundesliga (verified against the live table) ----
    "fc bayern munchen": ("Bayern Munich",),
    "borussia monchengladbach": ("B. Monchengladbach", "M'gladbach"),
    "1 fc koln": ("FC Koln",),
    "borussia dortmund": ("Dortmund",),
    "bayer 04 leverkusen": ("Bayer Leverkusen", "Leverkusen"),
    "fc schalke 04": ("Schalke",),
    "1 fc union berlin": ("Union Berlin",),
    "sv werder bremen": ("Werder Bremen",),
    "tsg 1899 hoffenheim": ("Hoffenheim",),
    "1 fsv mainz 05": ("Mainz",),
    "sc freiburg": ("Freiburg",),
    "fc augsburg": ("Augsburg",),
    "vfb stuttgart": ("Stuttgart",),
    "vfl bochum 1848": ("Bochum",),
    "vfl wolfsburg": ("Wolfsburg",),
    "1 fc heidenheim 1846": ("Heidenheim",),
    "sc paderborn 07": ("Paderborn",),
    "fc st pauli": ("St. Pauli",),
    # ---- Ligue 1 (verified against the live table) ----
    "paris saint germain fc": ("Paris SG", "PSG"),
    "paris saint germain": ("Paris SG", "PSG"),
    "aj auxerre": ("Auxerre",),
    "le havre ac": ("Le Havre",),
    "lille osc": ("Lille",),
    "ogc nice": ("Nice",),
    "rc strasbourg alsace": ("Strasbourg",),
    "angers sco": ("Angers",),
    "stade brestois 29": ("Brest",),
    "es troyes ac": ("Troyes",),
    "olympique de marseille": ("Marseille",),
    "rc lens": ("Lens",),
    "stade rennais fc 1901": ("Rennes",),
    "olympique lyonnais": ("Lyon",),
    "stade de reims": ("Reims",),
    "montpellier hsc": ("Montpellier",),
    "fc nantes": ("Nantes",),
    "fc metz": ("Metz",),
    # ---- Eredivisie ----
    "afc ajax": ("Ajax",),
    "az alkmaar": ("AZ", "AZ Alkmaar"),
    "feyenoord rotterdam": ("Feyenoord",),
    "fc twente": ("Twente",),
    "fc utrecht": ("Utrecht",),
    "sc heerenveen": ("Heerenveen",),
    "fc groningen": ("Groningen",),
    "nec nijmegen": ("Nijmegen", "NEC Nijmegen"),
    "go ahead eagles deventer": ("Go Ahead Eagles",),
    "pec zwolle": ("Zwolle", "PEC Zwolle"),
    "heracles almelo": ("Heracles", "Heracles Almelo"),
    "almere city fc": ("Almere City",),
    "psv eindhoven": ("PSV",),
    "excelsior rotterdam": ("Excelsior",),
    "vvv venlo": ("VVV", "VVV-Venlo"),
    "fc emmen": ("Emmen",),
    # ---- Primeira Liga ----
    "sl benfica": ("Benfica",),
    "fc porto": ("Porto",),
    "sporting cp": ("Sporting",),
    "sc braga": ("Braga",),
    "vitoria sc": ("Guimaraes", "Vitoria Guimaraes"),
    "boavista fc": ("Boavista",),
    "rio ave fc": ("Rio Ave",),
    "fc famalicao": ("Famalicao",),
    "gil vicente fc": ("Gil Vicente",),
    "gd estoril praia": ("Estoril",),
    "fc arouca": ("Arouca",),
    "casa pia ac": ("Casa Pia",),
    "moreirense fc": ("Moreirense",),
    "sc farense": ("Farense",),
    "cd nacional": ("Nacional",),
    "cs maritimo": ("Maritimo",),
    # ---- MLS ----
    "inter miami cf": ("Inter Miami",),
    "los angeles fc": ("LAFC", "Los Angeles FC"),
    "new york city fc": ("NYCFC", "New York City"),
    "atlanta united fc": ("Atlanta United",),
    "austin fc": ("Austin",),
    "portland timbers": ("Portland", "Portland Timbers"),
    "seattle sounders fc": ("Seattle", "Seattle Sounders"),
    "sporting kansas city": ("Kansas City", "Sporting KC"),
    # ---- Saudi Pro League ----
    "al hilal saudi fc": ("Al-Hilal",),
    "al nassr fc": ("Al-Nassr",),
    "al ittihad jeddah": ("Al-Ittihad",),
    "al ittihad": ("Al-Ittihad",),
    "al-ittihad": ("Al-Ittihad",),
    "ittihad": ("Al-Ittihad",),
    "al ahli saudi fc": ("Al-Ahli",),
    "al qadsiah": ("Al-Qadsiah",),
    "al-qadsiah": ("Al-Qadsiah",),
    "al qadisiyah": ("Al-Qadsiah",),
    "al kadijah": ("Al-Qadsiah",),
    "al shabab fc": ("Al-Shabab",),
    "al ettifaq fc": ("Al-Ettifaq",),
    "al fateh sc": ("Al-Fateh",),
    "damac fc": ("Damac",),
    # ---- Scottish Premiership ----
    "celtic fc": ("Celtic",),
    "rangers fc": ("Rangers",),
    "heart of midlothian": ("Hearts",),
    "hibernian fc": ("Hibs", "Hibernian"),
    "dundee united fc": ("Dundee Utd",),
    "aberdeen fc": ("Aberdeen",),
    "st mirren fc": ("St. Mirren",),
    "ross county fc": ("Ross County",),
    "st johnstone fc": ("St. Johnstone",),
    # ---- Belgian Pro League ----
    "club brugge kv": ("Club Brugge",),
    "krc genk": ("Genk",),
    "rsl anderlecht": ("Anderlecht",),
    "rsl anderlecht fc": ("Anderlecht",),
    "royale union saint gilloise": ("Union SG",),
    "royal antwerp fc": ("Antwerp",),
    "kaa gent": ("Gent",),
    "standard liege": ("Standard Liege",),
    "kv mechelen": ("Mechelen",),
    "oh leuven": ("OH Leuven",),
    "sporting charleroi": ("Charleroi",),
    # ---- Super Lig ----
    "galatasaray sk": ("Galatasaray",),
    "galatasaray": ("Galatasaray",),
    "fenerbahce sk": ("Fenerbahce",),
    "besiktas jk": ("Besiktas",),
    "besiktas": ("Besiktas",),
    "istanbul basaksehir": ("Basaksehir",),
    "adana demirspor": ("Adana Demirspor",),
    "caykur rizespor": ("Rizespor",),
    "gaziantep fk": ("Gaziantep",),
    # ---- UEL/UECL Teams ----
    "kauno zalgiris": ("Kauno Zalgiris",),
    "zalgiris": ("FK Zalgiris",),
    "fc copenhagen": ("FC Copenhagen",),
    "copenhagen": ("FC Copenhagen",),
    "inter turku": ("Inter Turku",),
    "larne": ("Larne FC",),
    # ---- Serie B ----
    "sassuolo calcio": ("Sassuolo",),
    "palermo fc": ("Palermo",),
    "pisa sc": ("Pisa",),
    "catanzaro calcio": ("Catanzaro",),
    "ternana calcio": ("Ternana",),
    # ---- Ligue 2 ----
    "sm caen": ("Caen",),
    "en avant guingamp": ("Guingamp",),
    "fc ajaccio": ("Ajaccio",),
    "sc bastia": ("Bastia",),
    "grenoble foot 38": ("Grenoble",),
    "pau fc": ("Pau",),
    "rodez af": ("Rodez",),
    "amiens sc": ("Amiens",),
    # ---- EFL Championship ----
    "west bromwich albion fc": ("West Brom",),
    "stoke city fc": ("Stoke",),
    "hull city fc": ("Hull",),
    "coventry city fc": ("Coventry",),
    "coventry city": ("Coventry",),
    "coventry": ("Coventry",),
    "bristol city fc": ("Bristol City",),
    "swansea city fc": ("Swansea",),
    "cardiff city fc": ("Cardiff",),
    "queens park rangers fc": ("QPR",),
    "preston north end fc": ("Preston",),
    "blackburn rovers fc": ("Blackburn",),
    "derby county fc": ("Derby",),
    "sheffield wednesday fc": ("Sheff Wed",),
    "oxford united fc": ("Oxford Utd",),
    "ipswich town fc": ("Ipswich",),
    # ---- Segunda ----
    "cordoba cf": ("Cordoba",),
    "cordoba": ("Cordoba",),
    "córdoba": ("Cordoba",),
    "córdoba cf": ("Cordoba",),
    "girona fc": ("Girona",),
}


_STANDINGS_REVERSE_CACHE: dict[str, list[tuple[str, str]]] | None = None


def _standings_reverse_index() -> dict[str, list[tuple[str, str]]]:
    """Standings spelling -> [(league_key, canonical_name)].

    Reverse of ``_STANDINGS_SPELLINGS`` (keyed by canonical abbr_key with
    the standings forms as values): built from teams.json canonicals so
    "Atl. Madrid" resolves back to ("LaLiga", "Atlético Madrid") instead
    of being hijacked by the generic "MADRID" alias -> Real Madrid CF
    (F3, verified live 2026-08-17). A spelling shared by the same club in
    several registered leagues accumulates all (league, canonical) pairs;
    the resolver picks the league-matching one first.
    """
    global _STANDINGS_REVERSE_CACHE
    if _STANDINGS_REVERSE_CACHE is not None:
        return _STANDINGS_REVERSE_CACHE
    out: dict[str, list[tuple[str, str]]] = {}
    try:
        teams = load_teams()
    except Exception:  # noqa: BLE001 -- reverse index is best-effort
        teams = {}
    for lg, aliases in teams.items():
        for canonical in aliases.values():
            ck = _abbr_key(canonical)
            spellings = _STANDINGS_SPELLINGS.get(ck)
            if not spellings:
                continue
            for sp in spellings:
                sk = _abbr_key(sp)
                if not sk:
                    continue
                pair = (lg, canonical)
                if sk not in out:
                    out[sk] = [pair]
                elif pair not in out[sk]:
                    out[sk].append(pair)
    _STANDINGS_REVERSE_CACHE = out
    return out


def standings_abbreviations(name: str) -> list[str]:
    """Standings-table spellings for a team name, most specific first.

    Built from (1) the reverse of ``_UNAMBIGUOUS_ABBREVIATIONS`` (full name
    -> short code, e.g. "Paris Saint-Germain" -> "PSG" / "Paris SG"),
    (2) the reverse of teams.json aliases (canonical -> codes) and (3) the
    curated ``_STANDINGS_SPELLINGS`` map. An abbreviation input ("PSG") is
    first expanded to its canonical name(s) so the reverse indexes can find
    its standings spelling. Returns [] when the name has no known
    abbreviation. Callers should also pass the club-token-stripped name
    ("Paris Saint-Germain FC" -> "Paris Saint-Germain") since the reverse
    indexes are keyed by the bare canonical form.
    """
    q = _abbr_key(name)
    if not q:
        return []
    out: list[str] = []
    seen: set[str] = set()
    bases = [name]
    resolved = resolve_team_alias(name, None)
    if resolved and _abbr_key(resolved) != q:
        bases.append(resolved)
    ca = canonical_abbreviation(name)
    if ca and _abbr_key(ca) != q:
        bases.append(ca)
    for base in bases:
        bq = _abbr_key(base)
        for key, canonical in _UNAMBIGUOUS_ABBREVIATIONS.items():
            if _abbr_key(canonical) == bq and key not in seen:
                seen.add(key)
                out.append(key)
        for alias, (_league, canonical) in _flatten().items():
            if _abbr_key(canonical) == bq and alias not in seen:
                seen.add(alias)
                out.append(alias)
        for cand in _STANDINGS_SPELLINGS.get(bq, ()):
            if cand.lower() not in seen:
                seen.add(cand.lower())
                out.append(cand)
    return out
