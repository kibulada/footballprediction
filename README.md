# Hermes Football

Agent advisory sepak bola terpisah dari [Hermes-QA](../Hermes-QA) (QA engineer agent Salsabila). Bot Discord baru, repo berdiri sendiri, audience hanya Kibul.

## Fitur

- **`!football today besok`** — top 5 match hari H+1, lintas liga (EPL, La Liga, Serie A, Bundesliga, Ligue 1, Liga 1 ID).
- **`!football today 2026-08-12`** — tanggal spesifik.
- **`!football today besok --leagues "EPL,Liga 1"`** — filter liga.
- **`!football today besok --top-n 10`** — ambil lebih banyak.
- **`!football compare MCN ARS`** — banding Manchester City vs Arsenal di EPL.
- **`!livescore <liga> <home> vs <away>`** — cari match via LiveScore (hari ini/besok), lalu pipeline analisa penuh (odds NowGoal + prediksi).
- **`!flashscore <liga> <home> vs <away>`** — sama, tapi sumber match-nya Flashscore.

## Stack

- Python 3.11+
- OddsPapi (primary, butuh `ODDSPAPI_KEY`) → NowGoal (kedua, tanpa key, via Tor/proxy) → The Odds API (terakhir; free tier, 500 req/bulan)
- API-Football via RapidAPI (free tier, 100 req/hari)
- discord.py
- httpx async
- Cache file `cache/football/`

## Setup

1. `cd D:\Hermes-Football`
2. `python -m venv .venv && .venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` → `.env`, isi:
    - `DISCORD_FOOTBALL_TOKEN` — bot baru dari Discord Developer Portal
    - `THE_ODDS_API_KEY` — https://the-odds-api.com
    - `API_FOOTBALL_KEY` — https://rapidapi.com/api-sports/api/api-football
    - `FOOTBALL_ALLOWED_USER_ID` — Discord user ID Kibul (numerik)
    - `FOOTBALL_DEFAULT_CHANNEL` — nama channel target (default `football-picks`)
5. Di Discord, invite bot baru ke channel `#football-picks`.

## Jalankan

```bash
# standalone test (no Discord)
python -m agents.football.runner top --date 2026-08-11 --top-n 3

# bot Discord
python bot.py
```

## Tests

```bash
python tests/test_football.py
```

## Batasan

- Tidak ada scheduled/cron. Request on-demand.
- Free tier — kuota rendah. Output selalu tunjuk `odds quota: X/500`.
- Read-only advisory. Tidak ada auto-bet.
- Audience Kibul only (user_id hard filter).
- Channel filter — bot abaikan message dari channel lain.
- Tidak ada HTTP service port. Runner di-spawn subprocess per request, timeout 90s.

## ML (Fase integrasi ProphitBet)

Feature engine rolling-stat + model ML terlatih (sklearn, kalibrasi isotonic,
sampling imbalanced). Sumber: port `statistics.py` ProphitBet (MIT) →
`agents/football/ml_features.py`. Model hidup offline, probabilitas jadi
sinyal tambahan (hybrid) di decision engine — tidak menggantikan Elo+Poisson.

```bash
# training + walk-forward eval (kronologis, anti-leakage), artifact ke cache/football/models/
python -m agents.football.runner train-model [--target result|over-under] [--model lr|rf|xgb|auto] [--folds 5] [--tune N]

# prediksi fixture 1X2 + O/U 2.5 dari model (butuh FOOTBALL_DATA_KEY)
python -m agents.football.runner predict-model [--date YYYY-MM-DD] [--leagues EPL,...]

# diagnostik fitur sekali-off: boruta / correlation / variance / coefficients
python -m agents.football.runner ml-analysis [--league EPL] [--metric all]
```

- **`train-model`** — fitur = rolling window intra-season (23 kolom ProphitBet)
  + fitur Elo (replay kronologis per-liga, leak-free). Evaluasi walk-forward
  per-season; baseline & metrik tersimpan di `metrics.json`.
- **`predict-model`** — `status: "unavailable"` bila tim belum punya ≥ window
  match di season berjalan (jujur, tidak di-fabrikasi). Per-match juga dipakai
  `!best`/`analisa` sebagai komponen `ml_agreement` di Decision Score (bobot
  via `models.decision.weights.ml_agreement`).
- **`ml-analysis`** — pilih fitur untuk retrain berikutnya.

Batasan jujur (hasil walk-forward 5 liga 2022-26): model 1X2 logloss ~1.06
≈ ensemble Elo+Poisson (~0.99 di EPL-only), di bawah market implied. O/U 2.5
~0.69 ≈ base rate. ML menambah redundancy/dekorrelasi, bukan lompatan akurasi.

## Alias Tim

Lihat `agents/football/teams.json`. Tambah alias dengan format `KODE: "Nama Asli"`.
