"""Hard gates for BEST PICK emission — post-mortem 2026-08-22.

Every gate here is a PURE function returning ``(passed, reasons)``. No I/O, no
randomness, no mutation of caller state. They are deliberately independent of
the composite score, because the composite score is a proven non-predictor of
profit:

    config/football.json -> models.decision.weights_validation_note
    "Spearman(decision score, realized ROI) = 0.0015 on EPL 2022-26
     walk-forward (n=1520). The 7-component score does NOT rank profitable
     bets; tercile ROI is non-monotonic and negative at every tercile."

So a BEST PICK must clear BINARY gates, and the score is only used to order
what survives. Full evidence trail:
``reports/bestpick_postmortem_2026-08-22.md``.

Gate map (IDs match the report):

    G2  agreement_gate        |model - market| <= max_dev_pp
    G3  lambda_1x2_gate       lambda direction must agree with the 1X2 favourite
    G4  lambda_total_gate     lambda_total inside a physically sane band
    G5  elo_integrity_gate    seeded + in range + no duplicate-rating collision
    G6  entity_integrity_gate no asymmetric Women/youth/reserve pairing
    G7  price_gate            a price and enough books actually exist
    G11 total_favor_gate      Total/BTTS: model underdog AND market underdog
                              is a contradiction, not value (2026-09-02)

G1 (respect the independent 1X2 decision layer), G8 (published pick ==
ranking[0]) and G10 (1X2 selection must be the model favourite) are
enforced at the decision site in ``signal_engine`` / ``analyse``, not here,
because they need the ranking object itself.

K6 ``lambda_direction_conflict`` (2026-09-02) is NOT a gate: it returns a
note the engine uses to CAP confidence at MEDIUM when the Elo-led blend and
the Poisson lambdas disagree on direction (Coventry / Cercle 2026-08-29
lost, LASK the same day won -- measured via failure_class K6 before it can
ever become a veto).
"""
from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------
# G2 — agreement, not divergence
# --------------------------------------------------------------------------
# The one bucket where this repo's own walk-forward audit (n=1520 EPL, real
# Pinnacle closing odds, reports/signal_audit_2026_08_12.md) shows the model
# BEATING the closing line is |deviation| 0-2pp (Brier gap -0.0016). Every
# wider bucket is worse than the market and loses money at closing odds
# (2-5pp: -3.3% ROI, 5-10pp: -8.9%). Live 2026-08-20/21 Totals reproduce it:
# |dev| <= 8pp went 4W-0L (+2.09u), +10..13pp went 1W-3L (-2.30u).
#
# NOTE this replaces the 2026-08-22 ``disagreement_gate`` thresholds
# (max_total_dev_pp=20 / _unseeded=8), which were tuned ONLY on the 21 Aug
# losing set and are demonstrably overfit: on 20 Aug (out-of-sample) a 20pp
# veto kills OFI Crete Over 2.5 (+26.5pp dev, WON +1.15u) and Gent Under 2.5
# (+21.1pp dev, WON +0.74u) to save SV Ried (-1.00u) -> net -0.89u.
# A symmetric REQUIREMENT of agreement is the supported rule; an
# upper-tail-only veto is not.
DEFAULT_MAX_DEV_PP = 8.0

# SCOPE CAVEAT (be honest about what is measured): the 8pp figure is derived
# from TOTALS picks only -- the 11 Over/Under picks of 2026-08-20/21 and the
# O/U 2.5 arm of the n=1520 EPL audit. There is NO Asian-Handicap deviation
# measurement in either sample, so applying the same threshold to AH is an
# EXTRAPOLATION, not a finding. It is applied anyway because the direction of
# the error is known (a tighter gate bets less, and "volume is the enemy" is
# the one conclusion the whole post-mortem supports), but it must be measured
# on AH before anyone treats 8pp as validated there. Verified side effect: in
# tests/test_signal_consistency.py an AH Home +1.25 candidate at ~+8.8pp
# deviation is vetoed by this threshold.

# --------------------------------------------------------------------------
# G4 — lambda sanity band
# --------------------------------------------------------------------------
# Verified failure: SV Ried v Grazer AK carried lambda_h 2.56 + lambda_a 1.528
# = 4.09 expected goals for two sides on L-L-D-W-L / L-W-L-W-L form, producing
# P(over2.5)=77.5% against a market at 49.0%. Result 1-0. A lambda_total that
# far outside the football-wide band is a broken estimate, not an edge.
DEFAULT_LAMBDA_TOTAL_MIN = 1.6
DEFAULT_LAMBDA_TOTAL_MAX = 3.6


def resolve_lambda_total_band(
    cfg: dict[str, Any] | None,
    league: str | None = None,
) -> tuple[float, float]:
    """G4 band: global min/max with optional per-league overrides (2026-08-22).

    The football-wide [1.6, 3.6] band ignores league scoring baselines:
    the Eredivisie historically averages ~3.2 goals/game (Europe's highest-
    scoring major league), so a legitimate top-vs-bottom matchup can produce
    a lambda_total the global ceiling rejects -- exactly the Fortuna Sittard
    v AZ card (lambda 3.96, market Over 2.5 @ 1.44). Overrides come from
    ``models.signal_engine.pick_gates.lambda_total_band_by_league`` keyed by
    case-insensitive league name; each entry is ``{"min": ..., "max": ...}``
    with either key optional, or a bare number meaning "max". Unknown
    leagues keep the global band.
    """
    cfg = cfg or {}
    lo = float(cfg.get("lambda_total_min", DEFAULT_LAMBDA_TOTAL_MIN))
    hi = float(cfg.get("lambda_total_max", DEFAULT_LAMBDA_TOTAL_MAX))
    overrides = cfg.get("lambda_total_band_by_league") or {}
    if league:
        entry = overrides.get(str(league).strip().lower())
        if isinstance(entry, dict):
            if entry.get("min") is not None:
                lo = float(entry["min"])
            if entry.get("max") is not None:
                hi = float(entry["max"])
        elif isinstance(entry, (int, float)):
            hi = float(entry)
    return lo, hi

# --------------------------------------------------------------------------
# G5 — Elo integrity
# --------------------------------------------------------------------------
# Verified failure: Real Betis v Real Sociedad logged elo_h 2036.0 / elo_a
# 2361.0 -- and the identical 2361.0 also appeared as Arsenal's rating on the
# same matchday. That is a lookup collision, not a strength assessment; it
# drove a -28.2pp 1X2 error (model home 16.5% vs market 44.7%; Betis won 1-0).
DEFAULT_ELO_MIN = 1300.0
# 2026-08-28: ceiling raised 2100 -> 2450. The live store legitimately rates
# the strongest clubs above 2100 (Barcelona 2298, Real Madrid 2243, Arsenal
# 2361 -- all verified BEST PICK winners 25-27 Aug); the Sociedad incident
# was a COLLISION (identical value on two clubs), which the collision check
# below catches independently of the ceiling.
DEFAULT_ELO_MAX = 2450.0
DEFAULT_ELO_COLLISION_EPS = 0.01


def resolve_elo_band(
    cfg: dict[str, Any] | None,
    league: str | None = None,
) -> tuple[float, float]:
    """G5 band: global min/max with optional per-league overrides.

    The senior-club floor (1300) sweeps legitimate reserve-XI ratings:
    the Jong teams of the Eerste Divisie rate 1238-1353 in the live
    elofootball.com store (Jong PSV v Jong Ajax 2026-09-01 saw every pick
    vetoed on a REAL away rating of 1238). Overrides come from
    ``models.signal_engine.pick_gates.elo_band_by_league`` keyed by
    case-insensitive league name; each entry is ``{"min": ..., "max": ...}``
    with either key optional, or a bare number meaning "max". Unknown
    leagues keep the global band. The collision and identical-rating
    checks are unaffected -- they do not depend on the band.
    """
    cfg = cfg or {}
    lo = float(cfg.get("elo_min", DEFAULT_ELO_MIN))
    hi = float(cfg.get("elo_max", DEFAULT_ELO_MAX))
    overrides = cfg.get("elo_band_by_league") or {}
    if league:
        entry = overrides.get(str(league).strip().lower())
        if isinstance(entry, dict):
            if entry.get("min") is not None:
                lo = float(entry["min"])
            if entry.get("max") is not None:
                hi = float(entry["max"])
        elif isinstance(entry, (int, float)):
            hi = float(entry)
    return lo, hi

# --------------------------------------------------------------------------
# G6 — entity integrity
# --------------------------------------------------------------------------
# Verified pollution in cache/football/predictions.jsonl match_ids:
#   UECL||Braga||Austria Wien Women          (women's side vs men's side)
#   UEL||KI Women||Lech Poznan               (women's side vs men's side)
#   UECL||Hearts of Oak||Rapid Wien II       (Ghanaian club vs reserve XI)
# The Elo/form that feed lambda are looked up by name, so a fabricated pairing
# silently models two teams that never met.
#
# Rule: a marker is only disqualifying when it is ASYMMETRIC. Two women's
# teams are a legitimate women's fixture (it just needs women's data, flagged
# via ``notes``); a women's team against a men's team is a fabricated pairing.
_MARKER_PATTERNS: tuple[tuple[str, str, bool], ...] = (
    # (marker kind, pattern, case_insensitive)
    ("women", r"\b(?:women|women's|ladies|feminin[ao]?|femenino|femminile|damen|frauen|kobiet|dames)\b|\(w\)", True),
    ("youth", r"\b(?:u1[5-9]|u2[0-3]|youth|academy|jugend|juniors?|primavera)\b", True),
    ("reserve", r"\b(?:reserves?|b-team)\b", True),
    # Bare Roman numerals mark a reserve XI ("Rapid Wien II"), but ONLY in
    # uppercase: the Finnish first-team club "Ii" (municipality of Ii) would be
    # a false positive under case-insensitive matching. Matched on the RAW name.
    ("reserve", r"\b(?:II|III)\b", False),
)

# Naming artifacts that indicate a SOURCE labelling problem rather than a
# wrong fixture. Notably "Hapoel Tel Aviv BC" in UECL 2026-08-20 was a REAL
# football fixture (Atalanta v Hapoel Tel Aviv 0-0) whose name carried a
# basketball-club suffix from the feed. Vetoing on it would have rejected a
# winner (+0.97u), so these WARN and never veto.
_ARTIFACT_PATTERNS: dict[str, str] = {
    "basketball_suffix": r"\bB\.?C\.?\b",
}


def _norm(name: str | None) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _markers(name: str | None) -> set[str]:
    """Return the set of marker kinds present in ``name``.

    Case-insensitive patterns run against the whitespace-normalised lowercase
    name; case-sensitive ones run against the raw name (see the Roman-numeral
    note on ``_MARKER_PATTERNS``).
    """
    low = _norm(name)
    raw = (name or "")
    found: set[str] = set()
    for kind, pat, ci in _MARKER_PATTERNS:
        if ci:
            hit = re.search(pat, low, re.I) is not None
        else:
            hit = re.search(pat, raw) is not None
        if hit:
            found.add(kind)
    return found


def _artifacts(name: str | None) -> set[str]:
    raw = (name or "")
    return {k for k, pat in _ARTIFACT_PATTERNS.items() if re.search(pat, raw)}


# ==========================================================================
# Gates
# ==========================================================================

def agreement_gate(
    model_prob: float | None,
    implied_prob: float | None,
    *,
    max_dev_pp: float = DEFAULT_MAX_DEV_PP,
) -> tuple[bool, list[str]]:
    """G2: the model must AGREE with the margin-free market price.

    Passes when ``|model_prob - implied_prob| <= max_dev_pp`` (percentage
    points). Missing inputs pass (a separate gate owns the missing-price
    case) -- never invent a veto from absent data.
    """
    if model_prob is None or implied_prob is None:
        return True, []
    dev_pp = abs(float(model_prob) - float(implied_prob)) * 100.0
    # Epsilon: the threshold is inclusive, and binary float subtraction makes
    # |0.58 - 0.50| * 100 == 8.000000000000007. Without this, a value the
    # operator configured as "exactly 8pp is allowed" would be rejected.
    if dev_pp <= max_dev_pp + 1e-9:
        return True, []
    return False, [
        f"deviasi model-pasar {dev_pp:.1f}pp > {max_dev_pp:.0f}pp — "
        "deviasi adalah alarm error, bukan value "
        "(audit n=1520: model paling buruk justru saat paling menyimpang)"
    ]


def lambda_total_gate(
    lambda_home: float | None,
    lambda_away: float | None,
    *,
    lo: float = DEFAULT_LAMBDA_TOTAL_MIN,
    hi: float = DEFAULT_LAMBDA_TOTAL_MAX,
) -> tuple[bool, list[str]]:
    """G4: lambda_total must sit inside a physically sane band.

    The rejection reason is deliberately CONTEXT-FREE here (no incident
    references): the call site appends a dynamic segment
    ``[band_source=..., ceiling=..., model_total_lambda=...,
    market_implied_total=..., model_market_gap=...]`` so every card shows
    WHERE the ceiling came from and HOW FAR the model sat from the market --
    a +28pp divergence (SV Ried 2026-08-21) and a +7.8pp one (Club Brugge
    v Cercle 2026-08-23) both trip this gate but are materially different
    situations the reader must be able to tell apart.
    """
    if lambda_home is None or lambda_away is None:
        return True, []
    total = float(lambda_home) + float(lambda_away)
    if lo <= total <= hi:
        return True, []
    return False, [
        f"lambda_total {total:.2f} di luar band [{lo:.1f}, {hi:.1f}] — "
        "estimasi gol di luar batas wajar"
    ]


def band_source(cfg: dict[str, Any] | None, league: str | None = None) -> str:
    """Where G4's applied band came from: ``league_override`` or ``global``.

    Mirrors ``resolve_lambda_total_band``'s lookup so the label can never
    disagree with the numbers actually enforced.
    """
    cfg = cfg or {}
    overrides = cfg.get("lambda_total_band_by_league") or {}
    if league:
        entry = overrides.get(str(league).strip().lower())
        if entry is not None:
            return "league_override"
    return "global"


def market_implied_total(market_totals: dict[str, Any] | None) -> float | None:
    """Market's expected total goals from the stored O/U ladder (pure).

    Devigs each push-free half-line pair (2.5 / 3.5 / 4.5) proportionally,
    then solves Poisson P(X >= k) = fair_over for the ladder's mean.
    Returns None when no usable pair exists -- never invents a number.
    Whole-number lines (3.0/4.0) carry push probability and are excluded.
    """
    import math

    mt = market_totals or {}

    def _devig_over(o: Any, u: Any) -> float | None:
        try:
            io, iu = 1.0 / float(o), 1.0 / float(u)
        except (TypeError, ZeroDivisionError, ValueError):
            return None
        if io <= 0 or iu <= 0:
            return None
        return io / (io + iu)

    def _tail_ge(lam: float, k: int) -> float:
        p = math.exp(-lam)
        s = p
        for i in range(1, k):
            p *= lam / i
            s += p
        return 1.0 - s

    def _solve(target: float, k: int) -> float | None:
        lo_, hi_ = 0.2, 8.0
        if not (0.0 < target < 1.0):
            return None
        for _ in range(60):
            mid = (lo_ + hi_) / 2.0
            if _tail_ge(mid, k) < target:
                lo_ = mid
            else:
                hi_ = mid
        return (lo_ + hi_) / 2.0

    estimates: list[float] = []
    for line, k in ((2.5, 3), (3.5, 4), (4.5, 5)):
        o = (mt.get(f"Over {line}") or {}).get("odds")
        u = (mt.get(f"Under {line}") or {}).get("odds")
        fair = _devig_over(o, u)
        if fair is None:
            continue
        lam = _solve(fair, k)
        if lam is not None:
            estimates.append(lam)
    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def lambda_1x2_gate(
    model_probs: dict[str, Any] | None,
    *,
    favourite_prob: float = 0.60,
) -> tuple[bool, list[str]]:
    """G3: the lambda matrix must not contradict the 1X2 favourite.

    When the ensemble 1X2 gives one side >= ``favourite_prob`` but the Poisson
    lambdas point the OTHER way, the two halves of the model disagree on the
    same card. Verified 2026-08-21: Erzurumspor lam_h 1.372 > lam_a 1.208 while
    1X2 said away 60.9% -> AH Home +1 inherited "prob 0.766", lost 0-4. Same
    shape on 2026-08-19 Hapoel v Sabah (1X2 home 55.9%, lam_h 0.728 <
    lam_a 1.526 -> Under 2.5, FT 2-1).
    """
    mp = model_probs or {}
    p = mp.get("1x2") or {}
    lh, la = mp.get("lambda_home"), mp.get("lambda_away")
    if lh is None or la is None:
        return True, []
    ph, pa = float(p.get("home", 0.0)), float(p.get("away", 0.0))
    if max(ph, pa) < favourite_prob:
        return True, []
    fav = "home" if ph >= pa else "away"
    lam_fav = "home" if float(lh) >= float(la) else "away"
    if fav == lam_fav:
        return True, []
    return False, [
        f"kontradiksi internal: 1X2 favorit {fav} "
        f"({max(ph, pa):.0%}) tapi lambda menunjuk {lam_fav} "
        f"(lam_h {float(lh):.3f} / lam_a {float(la):.3f}) — "
        "dua bagian model saling bertentangan (pola Erzurumspor 2026-08-21)"
    ]


def lambda_direction_conflict(
    model_probs: dict[str, Any] | None,
    market: str | None,
    selection: str | None,
    side: str | None = None,
) -> tuple[bool, str | None]:
    """K6 (2026-09-02): the pick backs the blend favourite, but the lambda
    matrix has that side scoring LESS than its opponent.

    Two halves of the same model disagree on WHO is stronger: the Elo-led
    ensemble (weight 0.6) says one side, the feature Poisson says the other.
    Coventry v Hull (1X2 home 55%, lam 1.41 v 1.43, FT 0-1) and Cercle v
    Lommel (home 57%, lam 1.45 v 1.76, FT 0-1) both lost; LASK v Altach
    (home 65%, lam 1.54 v 1.90, FT 3-0) won -- so this is a confidence CAP
    plus a tracked failure class, never a veto. Only directional picks on the
    blend favourite are affected (an AH underdog line never needs its side
    to be the stronger one). Returns ``(conflict, note)``.
    """
    if not is_directional_selection(market, selection):
        return False, None
    mp = model_probs or {}
    p = mp.get("1x2") or {}
    lh, la = mp.get("lambda_home"), mp.get("lambda_away")
    if lh is None or la is None or not p:
        return False, None
    try:
        lh_f, la_f = float(lh), float(la)
        ph, pa = float(p.get("home", 0.0)), float(p.get("away", 0.0))
    except (TypeError, ValueError):
        return False, None
    sel = str(selection or "").strip().lower()
    pick_side = side or ("home" if sel.startswith("home") else ("away" if sel.startswith("away") else None))
    if pick_side not in ("home", "away"):
        return False, None
    blend_fav = "home" if ph >= pa else "away"
    # Equal lambdas carry no direction -- only a STRICT reversal is a conflict.
    if abs(lh_f - la_f) < 1e-9:
        return False, None
    lam_fav = "home" if lh_f > la_f else "away"
    if pick_side != blend_fav or lam_fav == blend_fav:
        return False, None
    # Wording stays free of engine internals (the primary card bans "λ").
    return True, (
        f"gol ekspektasi {lh_f:.2f} vs {la_f:.2f} memihak {lam_fav} padahal 1X2 model "
        f"favorit {blend_fav} — dua bagian model tidak sepakat arah, confidence dibatasi "
        "MEDIUM (pola Coventry/Cercle 2026-08-29)"
    )


def elo_integrity_gate(
    model_probs: dict[str, Any] | None,
    *,
    lo: float = DEFAULT_ELO_MIN,
    hi: float = DEFAULT_ELO_MAX,
    require_seeded: bool = True,
    collision_eps: float = DEFAULT_ELO_COLLISION_EPS,
) -> tuple[bool, list[str]]:
    """G5: Elo ratings must be seeded, in range, and distinct.

    ``elo_home`` / ``elo_away`` are optional in ``model_probs``; when absent
    only the ``elo_seeded`` flag is checked (never fabricate a veto).
    """
    mp = model_probs or {}
    reasons: list[str] = []
    if require_seeded and mp.get("elo_seeded") is False:
        reasons.append(
            "Elo belum ter-seed untuk kedua tim — lambda hanya prior + form window "
            "(pola Al-Faisaly/Neom 2026-08-21: streak W-W-W-W-W dibaca sebagai kekuatan)"
        )
    eh, ea = mp.get("elo_home"), mp.get("elo_away")
    if eh is not None and ea is not None:
        eh, ea = float(eh), float(ea)
        for label, val in (("home", eh), ("away", ea)):
            if not (lo <= val <= hi):
                reasons.append(
                    f"Elo {label} {val:.1f} di luar band [{lo:.0f}, {hi:.0f}] — "
                    "rating tidak kredibel (pola Real Sociedad 2361 pada 2026-08-21)"
                )
        if abs(eh - ea) <= collision_eps:
            reasons.append(
                f"Elo home dan away identik ({eh:.1f}) — tabrakan lookup, bukan penilaian kekuatan"
            )
    return (not reasons), reasons


def entity_integrity_gate(
    home: str | None,
    away: str | None,
    *,
    veto_asymmetric: bool = True,
) -> tuple[bool, list[str]]:
    """G6: reject fabricated pairings (asymmetric Women / youth / reserve).

    Returns ``(passed, reasons)``. Symmetric markers (both sides women, both
    sides U19) pass but emit a note, because the fixture is real -- it just
    needs the matching data source.
    """
    mh, ma = _markers(home), _markers(away)
    reasons: list[str] = []

    asymmetric = mh.symmetric_difference(ma)
    if asymmetric:
        kinds = ", ".join(sorted(asymmetric))
        side = home if (mh - ma) else away
        reasons.append(
            f"pasangan tidak konsisten ({kinds}): '{side}' bertanding melawan tim "
            "level/gender berbeda — fixture fabrikasi, Elo/form yang dipakai bukan milik "
            "tim ini (pola Braga v Austria Wien Women 2026-08-20)"
        )
        if veto_asymmetric:
            return False, reasons

    both = mh & ma
    if both:
        reasons.append(
            f"kedua tim bertanda {', '.join(sorted(both))} — fixture sah, "
            "tapi butuh sumber data yang sesuai"
        )
    for name in (home, away):
        for art in _artifacts(name):
            reasons.append(
                f"artefak penamaan sumber pada '{name}' ({art}) — perlu resolusi entitas, "
                "tidak diveto (pola Hapoel Tel Aviv BC 2026-08-20 adalah fixture nyata)"
            )
    return True, reasons


def total_favor_gate(
    market: str | None,
    selection: str | None,
    model_prob: float | None,
    edge_pp: float | None,
) -> tuple[bool, list[str]]:
    """G11: Total/BTTS must favor the picked side (model_prob >= 0.5) when edge is negative.

    Lincoln 01-Sep Over 2.5 46% (<50% Under favored) with edge -2.4pp and
    Atalanta 31-Aug Over 49% (<50%) with edge -0.9pp both lost 1-0/0-0 while
    the model itself said Under. Picking a side the model thinks is the
    underdog AND the market also prices as underdog (edge<0) is a
    contradiction — not value. A contrarian Over with edge>0 (e.g. model 46%
    vs market 30% edge +16pp) still passes, so winners like that are kept.
    """
    if market not in ("Total", "BTTS"):
        return True, []
    if model_prob is None or edge_pp is None:
        return True, []
    try:
        mp = float(model_prob)
        ep = float(edge_pp)
    except (TypeError, ValueError):
        return True, []
    if mp < 0.5 and ep < 0:
        sel = selection or market
        return False, [
            f"{sel} model {mp:.0%} (<50% underdog) + edge {ep:+.1f}pp negatif — "
            "pick lawan model dan lawan pasar, bukan value (pola Lincoln/Atalanta 0-0/1-0)"
        ]
    return True, []


def price_gate(
    market_odds: float | None,
    *,
    bookmakers_count: int | None = None,
    min_bookmakers: int = 3,
) -> tuple[bool, list[str]]:
    """G7: a pick without a tradeable price is not a pick.

    Verified failure: Braga v Austria Wien Women 2026-08-20 was published as a
    BEST PICK with ``market_odds: null``.
    """
    reasons: list[str] = []
    if market_odds is None or float(market_odds) <= 1.0:
        reasons.append("tidak ada harga pasar (market_odds kosong/invalid) — pick tanpa harga bukan pick")
        return False, reasons
    if bookmakers_count is not None and int(bookmakers_count) < min_bookmakers:
        reasons.append(
            f"hanya {int(bookmakers_count)} bookmaker (< {min_bookmakers}) — konsensus harga tidak dapat dipercaya"
        )
        return False, reasons
    return True, reasons


# ==========================================================================
# Aggregate helper
# ==========================================================================

# Markets whose probability is inherited from the DIRECTION of the lambda
# matrix. A lambda-vs-1X2 contradiction corrupts these; it does NOT corrupt
# Totals/BTTS, which depend only on lambda_total (a direction-free sum).
#
# This scoping is load-bearing. Arsenal v Coventry 2026-08-21 carried a real
# contradiction (1X2 home 66.0% but lam_h 1.441 < lam_a 1.524) and its Over 2.5
# pick WON +0.57u. A card-wide veto on that contradiction would have thrown the
# winner away; a direction-scoped veto keeps it while still killing the
# Erzurumspor AH Home +1 (lost 0-4) and the Al Riyadh AH Home +1.75 (lost 0-4).
DIRECTIONAL_MARKETS = ("Asian Handicap", "1X2")


def is_directional_selection(market: str | None, selection: str | None) -> bool:
    """True for picks that bet on WHICH side is stronger.

    ``Draw`` is a 1X2 selection but carries no direction (it bets on
    closeness, not on a side), so an unseeded-Elo veto must not touch it --
    verified 2026-08-27 Internacional v Gremio Draw @3.38 WON on a card where
    both Elo were the 1500 prior.
    """
    if market == "Asian Handicap":
        return True
    if market == "1X2":
        return str(selection or "").strip().lower() != "draw"
    return False


def is_low_scoring_selection(market: str | None, selection: str | None) -> bool:
    """True for picks that need FEW goals (``Under x.5``, ``BTTS No``)."""
    sel = str(selection or "").strip().lower()
    if market == "Total":
        return sel.startswith("under")
    if market == "BTTS":
        return sel in ("no", "btts no")
    return False


# --------------------------------------------------------------------------
# K1 -- "know when you know nothing": Elo evidence scope
# --------------------------------------------------------------------------
# Post-mortem 2026-08-28 (25-27 Aug live): every BEST PICK whose BOTH Elo
# ratings were the 1500 prior went 3-3 (Al Shabab Home Win, Anyang Away Win,
# Vietnam Home Win all LOST; the three wins were Under / BTTS / Draw -- none
# of them a directional bet). Elo carries 60% of the 1X2 ensemble weight, so
# with two priors the direction is home-advantage noise. Picks with ONE side
# seeded went 5-1 (the seeded side still informs the direction).

def elo_evidence_scope(model_probs: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return ``(scope, note)`` describing how much of the card Elo can support.

    ``scope`` is ``"all"`` when BOTH sides are on the prior (directional
    markets must be vetoed, the rest capped), ``"one"`` when exactly one side
    is on the prior (directional confidence capped), ``None`` when both are
    seeded. Per-side flags are preferred; when only the legacy combined
    ``elo_seeded`` flag exists, ``False`` is treated as "one side" (the
    conservative reading that never fabricates a card-wide veto).
    """
    mp = model_probs or {}
    hs, as_ = mp.get("elo_home_seeded"), mp.get("elo_away_seeded")
    if hs is None or as_ is None:
        if mp.get("elo_seeded") is False:
            return "one", (
                "Elo salah satu tim belum ter-seed — arah 1X2/AH hanya dari sisi "
                "yang punya rating; confidence directional dibatasi"
            )
        return None, None
    if not hs and not as_:
        return "all", (
            "kedua tim tanpa Elo (prior 1500) — arah 1X2/AH murni noise "
            "(pola Al Shabab / Anyang / Vietnam 2026-08-25/26: 0-3); "
            "pick directional diveto, pick non-directional dibatasi LOW"
        )
    if not hs or not as_:
        side = "home" if not hs else "away"
        return "one", (
            f"Elo {side} belum ter-seed — arah hanya dari sisi lawan; "
            "confidence directional dibatasi MEDIUM"
        )
    return None, None


# --------------------------------------------------------------------------
# K2 -- source consistency: our team data vs the market's view of the team
# --------------------------------------------------------------------------
# Verified 2026-08-27 FC Copenhagen v Inter Turku: market home 1.25 (~80%),
# our features for "Copenhagen" said form D-L-L-L-L, 5.2 goals conceded per
# game, lambda 0.81 vs 1.53 (underdog). Those are not FC Kobenhavn's numbers
# -- the form/xG providers resolved a different entity. The engine emitted
# BTTS No @1.76 (LOW) on that lambda; FT 4-1. A card whose OWN data
# contradicts an 80% market favourite this violently is an entity-resolution
# failure, not a value spot: no market on that card is trustworthy.
DEFAULT_CONSISTENCY_FAV_IMPLIED = 0.70
DEFAULT_CONSISTENCY_MAX_WINS = 1
DEFAULT_CONSISTENCY_MIN_FORM = 4
DEFAULT_CONSISTENCY_MIN_GA = 3.0


def _devig_1x2(odds_1x2: dict[str, Any] | None) -> dict[str, float] | None:
    o = odds_1x2 or {}
    try:
        inv = {k: 1.0 / float(o[k]) for k in ("home", "draw", "away") if o.get(k)}
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if len(inv) < 2 or any(v <= 0 for v in inv.values()):
        return None
    tot = sum(inv.values())
    return {k: v / tot for k, v in inv.items()}


def _form_wins(sequence: str | None) -> tuple[int, int]:
    """(wins, matches) parsed from a ``W-D-L`` style sequence."""
    seq = [c for c in str(sequence or "").upper() if c in "WDL"]
    return seq.count("W"), len(seq)


def source_consistency_gate(
    odds_1x2: dict[str, Any] | None,
    team_form: dict[str, Any] | None,
    model_probs: dict[str, Any] | None,
    *,
    fav_implied_min: float = DEFAULT_CONSISTENCY_FAV_IMPLIED,
    max_wins: int = DEFAULT_CONSISTENCY_MAX_WINS,
    min_form_len: int = DEFAULT_CONSISTENCY_MIN_FORM,
    min_ga_avg: float = DEFAULT_CONSISTENCY_MIN_GA,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """K2: veto the whole card when our data for the market favourite is
    bottom-tier while the market prices it as a heavy favourite.

    ``team_form`` = ``{"home": {"sequence", "ga_avg"}, "away": {...}}``.
    Fires only when BOTH signals agree the entity is wrong: (form has
    <= ``max_wins`` wins over >= ``min_form_len`` matches AND concedes
    >= ``min_ga_avg`` per game) AND the lambda matrix also makes the
    market favourite the underdog. Missing inputs never veto. Returns
    ``(passed, reasons, detail)`` where ``detail`` describes the mismatch
    for the entity-resolution log.
    """
    fair = _devig_1x2(odds_1x2)
    if not fair:
        return True, [], None
    fav = max(("home", "away"), key=lambda k: fair.get(k, 0.0))
    fav_p = fair.get(fav, 0.0)
    if fav_p < fav_implied_min:
        return True, [], None
    tf = (team_form or {}).get(fav) or {}
    wins, n = _form_wins(tf.get("sequence"))
    try:
        ga = float(tf.get("ga_avg")) if tf.get("ga_avg") is not None else None
    except (TypeError, ValueError):
        ga = None
    mp = model_probs or {}
    lh, la = mp.get("lambda_home"), mp.get("lambda_away")
    lam_says_underdog = False
    if lh is not None and la is not None:
        lam_fav = float(lh) if fav == "home" else float(la)
        lam_other = float(la) if fav == "home" else float(lh)
        lam_says_underdog = lam_fav < lam_other
    form_bottom = n >= min_form_len and wins <= max_wins and ga is not None and ga >= min_ga_avg
    if form_bottom and lam_says_underdog:
        detail = {
            "favourite": fav,
            "market_implied": round(fav_p, 3),
            "form": tf.get("sequence"),
            "ga_avg": ga,
            "lambda_home": lh,
            "lambda_away": la,
        }
        return False, [
            f"data tim tidak konsisten dengan market: favorit {fav} dihargai "
            f"{fav_p:.0%} tapi form kita {tf.get('sequence')} ({wins}W/{n}), "
            f"kebobolan {ga:.1f}/laga, lambda menempatkannya sebagai underdog — "
            "kemungkinan entitas tim salah (pola Copenhagen 2026-08-27), "
            "seluruh kartu tidak dapat dipercaya"
        ], detail
    return True, [], None
