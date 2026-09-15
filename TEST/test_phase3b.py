"""Dependency-light contract tests for the Phase 3B driver."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import phase3b_runner as r


class StubChat:
    def __init__(self, reply='{"result":"normal","reason":"object appears intact"}'):
        self.reply = reply
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.reply


def test_result_parser_recovery():
    parsed, errors = r.parse_result('prefix\n```json\n{"result":"ANOMALOUS","reason":"crack visible"}\n```')
    assert not errors
    assert parsed == {"result": "anomalous", "reason": "crack visible"}


def test_prompt_contract_and_image_count():
    query = "/data/pfy/dataset/MVTec-AD/bottle/test/good/000.png"
    patch = "/data/pfy/AgentIAD/results/phase2/crops/bottle/good_000__rank1.png"
    direct = r.messages_for(query, None, "direct")
    global_patch = r.messages_for(query, patch, "global_patch")
    assert sum(x.get("type") == "image_url" for x in direct[1]["content"]) == 1
    assert sum(x.get("type") == "image_url" for x in global_patch[1]["content"]) == 2
    assert all(token.lower() not in direct[0]["content"].lower() for token in r.FORBIDDEN_DIRECT)
    assert "anomaly score" not in global_patch[0]["content"].lower()
    assert "PatchCore" not in global_patch[0]["content"]


def test_cached_top1_provenance():
    query = "/data/pfy/dataset/MVTec-AD/bottle/test/good/000.png"
    meta = r.load_patch("bottle", query)
    assert meta["patch_rank"] == 1
    assert Path(meta["patch_path"]).is_file()
    assert len(meta["patch_bbox"]) == 4


def test_stub_call_uses_single_deterministic_generation():
    chat = StubChat()
    result = asyncio.run(r.run_one(chat, "/data/pfy/dataset/MVTec-AD/bottle/test/good/000.png", None, "direct", 32))
    assert result["parse_ok"] and result["parsed_output"]["result"] == "normal"
    assert result["vlm_calls"] == 1
    assert chat.calls[0][1] == {"max_new_tokens": 32, "do_sample": False}


def test_metrics_include_required_binary_fields():
    rows = [
        {"parse_ok": True, "ground_truth": "normal", "prediction": "normal", "correct": True},
        {"parse_ok": True, "ground_truth": "normal", "prediction": "anomalous", "correct": False},
        {"parse_ok": True, "ground_truth": "anomalous", "prediction": "anomalous", "correct": True},
        {"parse_ok": True, "ground_truth": "anomalous", "prediction": "normal", "correct": False},
    ]
    got = r.metric(rows)
    assert all(k in got for k in ("accuracy", "precision", "recall", "specificity", "f1", "TP", "TN", "FP", "FN"))
    assert (got["TP"], got["TN"], got["FP"], got["FN"]) == (1, 1, 1, 1)
