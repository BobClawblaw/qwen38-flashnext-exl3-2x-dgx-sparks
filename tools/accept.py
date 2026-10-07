#!/usr/bin/env python3
"""MTP acceptance between two TensorFold /health snapshots.

  python3 tools/accept.py health-before.json health-after.json

TensorFold counts verify rounds (rounds_total), draft tokens verified (drafted_total) and drafts
kept (accepted_total). Tokens per round = 1 + accepted / rounds; draft hit rate = accepted / drafted.
vLLM's spec_decode counters have no rounds_total, so bench_decode.py prints no acceptance here.
"""
import json
import sys

KEYS = ("requests_total", "completion_tokens_total", "rounds_total", "drafted_total", "accepted_total",
        "decode_seconds_total", "prefill_seconds_total", "prompt_tokens_total", "cached_tokens_total")


def main() -> int:
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    d = {k: b.get(k, 0) - a.get(k, 0) for k in KEYS}
    out = dict(d)
    if d["rounds_total"]:
        out["tokens_per_round"] = round(1 + d["accepted_total"] / d["rounds_total"], 3)
    if d["drafted_total"]:
        out["draft_hit_rate"] = round(d["accepted_total"] / d["drafted_total"], 4)
    if d["decode_seconds_total"]:
        out["decode_tok_s_server"] = round(d["completion_tokens_total"] / d["decode_seconds_total"], 2)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
