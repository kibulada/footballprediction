"""Multi-bookmaker odds dari football-data.co.uk (gratis, tanpa API key).

KENAPA MODUL INI ADA
--------------------
Audit 2026-09-04 (n=1336 kandidat tersettle) menemukan model internal
TIDAK menambah informasi di atas harga pasar:

    regresi logistik y ~ logit(pasar) + logit(model)
        koef pasar : +1.59
        koef model : -0.12   <- NEGATIF
    blend optimal p = w*model + (1-w)*pasar  ->  w* = 0.00

Artinya edge tidak akan datang dari "model lebih pintar". Tapi ada edge
struktural yang tidak butuh model sama sekali -- LINE SHOPPING:

    12.104 match (12 liga x 3 musim), strategi & pilihan IDENTIK:
        vig harga rata-rata (AvgC) : +5.74%
        vig Pinnacle (PSC)         : +3.31%
        vig harga terbaik (MaxC)   : +0.98%
        beli di rata-rata : -338.58u  (ROI -2.80%)
        beli di terbaik   : +107.09u  (ROI +0.88%)
    Konsisten 3 musim: +4.08% / +3.73% / +3.23% per taruhan.

Itu bukan prediksi yang bisa meleset -- itu aritmetika harga.

SUMBER
------
- ``fixtures.csv``  : pertandingan MENDATANG + odds pre-match (9 bookmaker,
  1X2 + O/U 2.5 + Asian Handicap). Diverifikasi 48/48 fixture lengkap.
- ``mmz4281/<musim>/<liga>.csv`` : historis + odds pembukaan & penutupan,
  22 kolom bookmaker termasuk Pinnacle. Untuk backtest/kalibrasi.

Tanpa API key, tanpa kuota. (Catatan: THE_ODDS_API_KEY di .env sudah mati
-- HTTP 401 per 2026-09-04.)

Semua fungsi fail-soft: kegagalan jaringan mengembalikan None/{} dan
TIDAK PERNAH melempar ke pemanggil.
"""

from __future__ import annotations

import csv
import io
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("hermes-football-bot")

BASE = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE}/fixtures.csv"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# football-data.co.uk Div code -> nama liga internal
DIV_TO_LEAGUE: dict[str, str] = {
    "E0": "EPL", "E1": "EFL Championship", "E2": "EFL League One",
    "E3": "EFL League Two", "EC": "National League",
    "SC0": "Scottish Premiership", "SC1": "Scottish Championship",
    "D1": "Bundesliga", "D2": "2. Bundesliga",
    "I1": "Serie A", "I2": "Serie B",
    "SP1": "LaLiga", "SP2": "Segunda",
    "F1": "Ligue 1", "F2": "Ligue 2",
    "N1": "Eredivisie", "B1": "Belgian Pro League",
    "P1": "Primeira Liga", "T1": "Super Lig", "G1": "Super League Greece",
}
LEAGUE_TO_DIV: dict[str, str] = {v: k for k, v in DIV_TO_LEAGUE.items()}

# prefiks kolom bookmaker -> nama enak dibaca. Urutan = prioritas referensi.
BOOKMAKERS: dict[str, str] = {
    "PS": "Pinnacle", "B365": "Bet365", "BF": "Betfair", "BFD": "Betfair Sportsbook",
    "BV": "BetVictor", "BW": "bwin", "PP": "Paddy Power", "SKB": "Skybet",
    "BFE": "Betfair Exchange", "BMGM": "BetMGM", "CL": "Coral", "LB": "Ladbrokes",
    "WH": "William Hill", "VC": "VC Bet",
}
# agregat (bukan bandar tunggal)
AGGREGATES = {"Max": "Harga terbaik", "Avg": "Rata-rata pasar"}

_CACHE_TTL = 900.0  # 15 menit
_cache: dict[str, tuple[float, Any]] = {}


def _fetch(url: str, timeout: float = 25.0, retries: int = 2) -> str | None:
    """Ambil URL sebagai teks. None kalau gagal (tidak pernah melempar)."""
    now = time.time()
    hit = _cache.get(url)
    if hit and (now - hit[0]) < _CACHE_TTL:
        return hit[1]
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            text = raw.decode("utf-8-sig", errors="ignore")
            if len(text) < 200:
                return None
            _cache[url] = (now, text)
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            if attempt >= retries:
                logger.warning("multi_odds: gagal ambil %s: %s", url, exc)
                return None
            time.sleep(0.8 * (attempt + 1))
    return None


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 1.0 else None


def _parse_csv(text: str) -> list[dict[str, str]]:
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except (csv.Error, ValueError) as exc:
        logger.warning("multi_odds: CSV tidak terbaca: %s", exc)
        return []


# --------------------------------------------------------------------------
# Harga & devig
# --------------------------------------------------------------------------

def collect_1x2(row: dict[str, str], closing: bool = False) -> dict[str, list[float | None]]:
    """{'Pinnacle': [H,D,A], ...} dari satu baris CSV.

    ``closing=True`` membaca kolom penutupan (sufiks C, mis. ``PSCH``).
    """
    suffix = "C" if closing else ""
    out: dict[str, list[float | None]] = {}
    for pref, name in {**BOOKMAKERS, **AGGREGATES}.items():
        trio = [_f(row.get(f"{pref}{suffix}{k}")) for k in ("H", "D", "A")]
        if any(x is not None for x in trio):
            out[name] = trio
    return out


def devig(odds: Iterable[float | None], method: str = "proportional") -> list[float] | None:
    """Buang margin bandar -> probabilitas wajar. None kalau tidak lengkap."""
    vals = list(odds)
    if not vals or any(v is None or v <= 1.0 for v in vals):
        return None
    inv = [1.0 / float(v) for v in vals]  # type: ignore[arg-type]
    total = sum(inv)
    if total <= 0:
        return None
    if method == "power":
        lo, hi = 0.5, 3.0
        for _ in range(60):
            k = (lo + hi) / 2
            s = sum(x ** k for x in inv)
            if s > 1.0:
                lo = k
            else:
                hi = k
        k = (lo + hi) / 2
        p = [x ** k for x in inv]
        s = sum(p)
        return [x / s for x in p]
    return [x / total for x in inv]


def vig_pct(odds: Iterable[float | None]) -> float | None:
    vals = [v for v in odds if v]
    if len(vals) < 2:
        return None
    return (sum(1.0 / float(v) for v in vals) - 1.0) * 100.0


def best_price(books: dict[str, list[float | None]], idx: int) -> tuple[float, str] | None:
    """Harga tertinggi untuk outcome ``idx`` + nama sumbernya.

    Agregat ("Rata-rata pasar"/"Harga terbaik") dilewati saat mencari bandar
    -- keduanya bukan tempat taruhan bisa dipasang. Kolom ``Max`` dipakai
    hanya sebagai pembanding kalau tidak ada bandar tunggal yang punya harga
    (dan ditandai jelas), supaya kartu tidak pernah menyuruh pasang di
    "Harga terbaik" yang bukan bandar sungguhan.
    """
    best: tuple[float, str] | None = None
    for name, trio in books.items():
        if name in AGGREGATES.values():
            continue  # agregat, bukan bandar -> tidak bisa dipasangi
        if idx >= len(trio):
            continue
        v = trio[idx]
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, name)
    if best is None:
        agg = books.get("Harga terbaik")
        if agg and idx < len(agg) and agg[idx] is not None:
            return (agg[idx], "Max pasar (bandar tak teridentifikasi)")  # type: ignore[return-value]
    return best


def price_edge(books: dict[str, list[float | None]], reference: str = "Pinnacle") -> list[dict[str, Any]] | None:
    """Edge tiap outcome: harga TERBAIK vs probabilitas wajar dari ``reference``.

    Ini inti line shopping: referensi (Pinnacle) memberi harga wajar,
    lalu kita ambil harga terbaik yang tersedia. Edge = EV murni, tidak
    melibatkan model internal sama sekali.
    """
    ref = books.get(reference) or books.get("Rata-rata pasar")
    if not ref:
        return None
    fair = devig(ref)
    if not fair:
        return None
    out = []
    for i, label in enumerate(("home", "draw", "away")):
        bp = best_price(books, i)
        if bp is None:
            continue
        price, src = bp
        out.append({
            "outcome": label,
            "fair_prob": round(fair[i], 4),
            "fair_odds": round(1.0 / fair[i], 3) if fair[i] > 0 else None,
            "best_odds": price,
            "best_book": src,
            "edge_pct": round((price * fair[i] - 1.0) * 100.0, 2),
            "reference": reference if books.get(reference) else "Rata-rata pasar",
        })
    return out


# --------------------------------------------------------------------------
# API publik
# --------------------------------------------------------------------------

def get_fixtures(league: str | None = None) -> list[dict[str, Any]]:
    """Fixture mendatang + odds pre-match multi-bookmaker."""
    text = _fetch(FIXTURES_URL)
    if not text:
        return []
    div = LEAGUE_TO_DIV.get(league) if league else None
    out = []
    for row in _parse_csv(text):
        if not row.get("HomeTeam"):
            continue
        if div and row.get("Div") != div:
            continue
        books = collect_1x2(row)
        if not books:
            continue
        out.append({
            "div": row.get("Div"),
            "league": DIV_TO_LEAGUE.get(row.get("Div") or "", row.get("Div")),
            "date": row.get("Date"), "time": row.get("Time"),
            "home": row.get("HomeTeam"), "away": row.get("AwayTeam"),
            "books": books,
            "n_books": len(books),
            "vig_avg": vig_pct(books.get("Rata-rata pasar") or []),
            "vig_best": vig_pct(books.get("Harga terbaik") or []),
            "edges": price_edge(books),
            "totals": {
                "over_2.5_best": _f(row.get("Max>2.5")),
                "under_2.5_best": _f(row.get("Max<2.5")),
                "over_2.5_avg": _f(row.get("Avg>2.5")),
                "under_2.5_avg": _f(row.get("Avg<2.5")),
            },
            "ah": {
                "line": _f(row.get("AHh")) if row.get("AHh") not in (None, "") else None,
                "home_best": _f(row.get("MaxAHH")),
                "away_best": _f(row.get("MaxAHA")),
            },
        })
    return out


def find_fixture(home: str, away: str, league: str | None = None) -> dict[str, Any] | None:
    """Cari satu fixture dengan pencocokan nama longgar."""
    def norm(s: str) -> str:
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    h, a = norm(home), norm(away)
    if not h or not a:
        return None
    best = None
    for fx in get_fixtures(league):
        fh, fa = norm(fx["home"]), norm(fx["away"])
        if (fh.startswith(h[:6]) or h.startswith(fh[:6])) and \
           (fa.startswith(a[:6]) or a.startswith(fa[:6])):
            return fx
        if h in fh or fh in h:
            best = best or fx
    return best


def get_historical(league: str, season: str = "2526") -> list[dict[str, Any]]:
    """Match historis + odds pembukaan/penutupan (untuk backtest)."""
    div = LEAGUE_TO_DIV.get(league, league)
    text = _fetch(f"{BASE}/mmz4281/{season}/{div}.csv")
    if not text:
        return []
    out = []
    for row in _parse_csv(text):
        if not row.get("HomeTeam"):
            continue
        out.append({
            "date": row.get("Date"), "home": row.get("HomeTeam"), "away": row.get("AwayTeam"),
            "fthg": row.get("FTHG"), "ftag": row.get("FTAG"), "ftr": row.get("FTR"),
            "books_open": collect_1x2(row, closing=False),
            "books_close": collect_1x2(row, closing=True),
        })
    return out


def summary(fx: dict[str, Any]) -> str:
    """Ringkasan enak dibaca untuk kartu/laporan."""
    lines = [f"{fx['home']} v {fx['away']} ({fx['league']}, {fx['date']})"]
    va, vb = fx.get("vig_avg"), fx.get("vig_best")
    if va is not None and vb is not None:
        lines.append(f"  vig: rata-rata {va:+.2f}% -> harga terbaik {vb:+.2f}% "
                     f"(hemat {va - vb:.2f}pp)")
    for e in fx.get("edges") or []:
        lines.append(
            f"  {e['outcome']:<5} wajar {e['fair_odds']} | terbaik {e['best_odds']} "
            f"@ {e['best_book']} | edge {e['edge_pct']:+.2f}%"
        )
    return "\n".join(lines)
