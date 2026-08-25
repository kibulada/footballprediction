# BEST PICK v3 — BLIND REPLAY LOG (2026-08-24)

Replay model v3 (F1 anchor + F2 kalibrasi total + F4-lite + F14 kandidat 1X2)
atas seluruh snapshot analisa >= 23 Agu 16:00 WIB dari predictions.jsonl.
**Ditulis SEBELUM melihat hasil pertandingan** — settle & skor ROI menyusul
agar evaluasi objektif (baseline pengukuran fase settlement-loop).

Catatan: bot production belum restart saat replay ini; ini simulasi offline
dari snapshot yang sudah ada. Match 24 Agu belum ada di log.
| Analisa (UTC) | Kick WIB | Liga | Match | Pick LAMA | Pick v3 (BLIND) |
|---|---|---|---|---|---|
| 08-23T09:56 | 23 17:15 | Eredivisie | G.A. Eagles v Den Haag | Over 2.5 @ 1.52 [62 MEDIUM] | **Over 2.5 @ 1.52 [75 HIGH]** |
| 08-23T12:19 | 23 20:00 | EPL | Manchester City v Bournemouth | - | **NO BET** |
| 08-23T12:23 | 23 20:00 | Ligue 1 | Angers v Lille | BTTS Yes @ 1.99 [75 HIGH] | **NO BET** |
| 08-23T12:27 | 23 19:30 | Eredivisie | PSV v Groningen | BTTS Yes @ 1.587 [64 MEDIUM] | **NO BET** |
| 08-23T12:31 | 23 20:00 | EPL | Brighton v Aston Villa | - | **Over 2.5 @ 1.98 [53 MEDIUM]** |
| 08-23T13:37 | 23 22:00 | La Liga | Atl. Madrid v Villarreal | Over 2.5 @ 1.65 [62 MEDIUM] | **Over 2.5 @ 1.65 [71 HIGH]** |
| 08-23T14:14 | 23 21:45 | Eredivisie | Cambuur v Feyenoord | BTTS Yes @ 1.75 [73 HIGH] | **Over 2.5 @ 1.37 [83 VERY HIGH]** |
| 08-23T14:18 | ? | La Liga | Club Atlético de Madrid v Villarreal | - | **NO BET** |
| 08-23T14:23 | 23 21:30 | Primeira Liga | Vitoria Guimaraes v Nacional | BTTS No @ 2.02 [47 LOW] | **BTTS No @ 2.02 [62 MEDIUM]** |
| 08-23T14:27 | 23 22:15 | Ligue 1 | Le Havre v Monaco | Under 2.5 @ 2.12 [51 LOW] | **NO BET** |
| 08-23T15:05 | 23 22:30 | EPL | Newcastle v Liverpool | BTTS No @ 2.88 [47 LOW] | **Over 2.5 @ 1.52 [76 HIGH]** |
| 08-23T16:02 | 23 23:00 | Süper Lig | Trabzonspor v Basaksehir | - | **Home Win @ 1.88 [62 MEDIUM]** |
| 08-23T16:08 | 23 23:30 | Serie A | Frosinone v Juventus | BTTS Yes @ 1.833 [54 MEDIUM] | **Over 2.5 @ 1.657 [72 HIGH]** |
| 08-23T16:10 | 23 23:30 | Belgian Pro League | Club Brugge KV v Cercle Brugge KSV | - | **NO BET** |
| 08-23T16:12 | 23 23:30 | Serie A | Venezia v Lecce | - | **Away Win @ 5.01 [50 LOW]** |
| 08-23T16:39 | 24 00:30 | La Liga | Getafe v Racing Santander | - | **NO BET** |
| 08-23T17:37 | 24 01:30 | Süper Lig | Alanyaspor v Besiktas | - | **Away Win @ 1.714 [48 LOW]** |
| 08-23T17:39 | 24 01:30 | Süper Lig | Goztepe v Genclerbirligi | - | **NO BET** |
| 08-23T17:50 | 24 01:45 | Serie A | Atalanta v Sassuolo | BTTS No @ 2.05 [51 LOW] | **Home Win @ 1.561 [56 MEDIUM]** |
| 08-23T17:54 | ? | Ligue 1 | Rennes v PSG | - | **NO BET** |
| 08-23T17:57 | 24 01:45 | Serie A | Torino v AC Milan | BTTS Yes @ 2.03 [48 LOW] | **NO BET** |
| 08-23T18:00 | 24 01:45 | Ligue 1 | Rennes v PSG | - | **NO BET** |
| 08-23T18:05 | 24 02:30 | Primeira Liga | FC Porto v Arouca | - | **NO BET** |
| 08-23T18:13 | 24 02:00 | HNL | Hajduk Split v Osijek | BTTS Yes @ 1.617 [47 LOW] | **BTTS Yes @ 1.617 [73 HIGH]** |
| 08-23T18:17 | 24 01:30 | Liga 1 | CFR Cluj v FCSB | Over 2.5 @ 1.75 [82 VERY HIGH] | **BTTS Yes @ 1.662 [64 MEDIUM]** |
| 08-23T18:17 | 24 02:00 | Serie B | Palermo v Juve Stabia | - | **NO BET** |
| 08-23T18:21 | 24 02:30 | La Liga | Elche v Barcelona | BTTS Yes @ 1.781 [57 MEDIUM] | **Over 2.5 @ 1.512 [77 HIGH]** |
| 08-23T18:25 | 24 02:00 | Serie A | RB Bragantino v Gremio | - | **Over 2.5 @ 1.9300000000000002 [50 LOW]** |
| 08-23T18:25 | 24 02:00 | Serie A | Palmeiras v Vasco da Gama | - | **NO BET** |

### Rekap pick v3

```
NO BET                       14
Total:Over 2.5               8
1X2:Home Win                 2
1X2:Away Win                 2
BTTS:BTTS Yes                2
BTTS:BTTS No                 1
```

Total match: 29


---

## HASIL SETTLE (diisi 2026-08-24 setelah pertandingan)

Sumber FT: LiveScore feed resmi bot (etch_finished_livescore_results, sama
dengan jalur unner settle auto). 22/27 fixture unik ketemu; n/a = nama feed
tidak cocok (G.A. Eagles/Den Haag, Rennes-PSG) — tidak dipaksakan.

| Buku | Stake | W-L | ROI flat |
|---|---|---|---|
| OLD (pick terpublikasi lama) | 13u | 6-7 | **−14.2%** |
| NEW v3 (blind replay) | 14u | 9-5 | **+32.2%** |

Detail per-match (OLD → NEW, hasil settle):

| Match | FT | OLD | NEW v3 |
|---|---|---|---|
| Man City v Bournemouth | 2-1 | – | NO BET (G4 band 3.79) |
| Angers v Lille | 0-2 | BTTS Yes LOSS | NO BET ✅ hindari |
| PSV v Groningen | 5-1 | BTTS Yes WIN | NO BET ❌ miss |
| Brighton v Aston Villa | 4-0 | – | Over 2.5 @1.98 WIN ✅ |
| Atl. Madrid v Villarreal | 2-2 | Over 2.5 WIN | Over 2.5 WIN |
| Cambuur v Feyenoord | 2-5 | BTTS Yes WIN @1.75 | Over 2.5 @1.37 WIN (odds turun) |
| Vitoria v Nacional | 1-0 | BTTS No WIN | BTTS No WIN |
| Le Havre v Monaco | 0-1 | Under 2.5 WIN | NO BET ❌ miss |
| Newcastle v Liverpool | 2-2 | BTTS No LOSS | Over 2.5 @1.52 WIN ✅ |
| Trabzonspor v Basaksehir | 2-1 | – | Home Win @1.88 WIN ✅ |
| Frosinone v Juventus | 0-1 | BTTS Yes LOSS | Over 2.5 @1.66 LOSS |
| Venezia v Lecce | 0-2 | – | Away Win @5.01 WIN ✅ |
| Alanyaspor v Besiktas | 1-0 | – | Away Win @1.71 LOSS |
| Goztepe v Genclerbirligi | 0-1 | (leak Over) LOSS | NO BET ✅ hindari |
| Atalanta v Sassuolo | 2-1 | BTTS No LOSS | Home Win @1.56 WIN ✅ |
| Torino v AC Milan | 1-2 | BTTS Yes WIN | NO BET ❌ miss |
| FC Porto v Arouca | 2-0 | – | NO BET (dev Home 10.3pp) |
| Hajduk Split v Osijek | 4-0 | BTTS Yes LOSS | BTTS Yes LOSS |
| CFR Cluj v FCSB | 1-0 | Over 2.5 LOSS | BTTS Yes LOSS (varian) |
| Palermo v Juve Stabia | 1-0 | – | NO BET (dev 11.5pp) |
| Elche v Barcelona | 0-5 | BTTS Yes LOSS | **Over 2.5 @1.51 WIN ✅** |
| RB Bragantino v Gremio | 1-0 | – | Over 2.5 @1.93 LOSS |
| Palmeiras v Vasco | 4-1 | – | NO BET |

Catatan jujur:
- NO BET set: 2 loss berhasil dihindari (Angers, Goztepe-leak), 3 winner ketinggalan
  (PSV +1.59u, Le Havre +2.12u, Torino +2.03u potensi). Net disiplin tetap positif
  pada buku ini, tapi biaya skip NYATA dan harus dipantau mingguan.
- G4 ceiling 3.6 mem-blok kartu penuh di Man City (3.79; Over akan MENANG) dan
  Club Brugge (3.80; Over akan KALAH) — bukti campuran, ceiling dipertahankan,
  masuk daftar ukur (league-aware ceiling) fase berikutnya.
- Sampel 1 hari (13-14 stake) — arah konsisten dengan desain & buku historis
  (−7.9%), belum cukup untuk klaim statistik final.
