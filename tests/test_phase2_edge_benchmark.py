"""Phase 2 tests: edge-benchmark labeling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.edge_benchmark import (
    SELF_CLOSE,
    SOFT_CONSENSUS,
    edge_benchmark,
)
from agents.football.prediction_log import append_snapshot


def test_default_benchmark_is_soft_consensus():
    b = edge_benchmark(None)
    assert b["key"] == SOFT_CONSENSUS
    assert b["beats_market_claim"] is False
    assert "closing line" in b["label"]


def test_self_close_configured():
    b = edge_benchmark({"models": {"decision": {"edge_benchmark": SELF_CLOSE}}})
    assert b["key"] == SELF_CLOSE
    assert b["beats_market_claim"] is False


def test_label_from_config():
    b = edge_benchmark(
        {"models": {"decision": {"edge_benchmark_label": "custom close"}}}
    )
    assert b["label"] == "custom close"


def test_snapshot_stores_benchmark(tmp_path):
    path = tmp_path / "pred.jsonl"
    append_snapshot(
        path,
        match_id="X",
        league="EPL", home="A", away="B", kickoff="2026-01-01T00:00:00Z",
        prob={"home": 0.5, "draw": 0.3, "away": 0.2},
        odds=None, edge=None, confidence=None, signal=None, calibration=None,
        model_version=None, input_hash=None, best_pick=None, sources=[],
        edge_benchmark=edge_benchmark(None),
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["edge_benchmark"]["key"] == SOFT_CONSENSUS
    assert row["edge_benchmark"]["beats_market_claim"] is False
