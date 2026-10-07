#!/usr/bin/env python3
"""Tabulate gate directories against a baseline (markdown).

  python3 tools/compare.py BASE_DIR CAND_DIR [CAND_DIR ...]

Reads bench-frozen.out (bench_decode.py SUMMARY), bench-cells.out (tools/bench_cells.py SUMMARY) and
accept-frozen.txt from each tools/session_gate.sh directory. Rows: per-stream decode tok/s of every
frozen cell (c=2 rows show aggregate too) and every longer cell; columns: base, then each candidate
with its delta against base. Reply sha256s of greedy cells are compared to base (= same text).
"""
import json
import sys
from pathlib import Path


def summary(path: Path):
    if not path.is_file():
        return None
    t = path.read_text()
    i = t.find("SUMMARY ")
    return json.loads(t[i + 8:]) if i >= 0 else None


def load(d: Path) -> dict:
    out = {}
    for r in summary(d / "bench-frozen.out") or []:
        key = f"{r['phase']} c{r['concurrency']}"
        out[key] = (r["median_decode_tok_s"], None)
        if r["concurrency"] > 1:
            out[key + " agg"] = (r["median_agg_tok_s"], None)
        out[key + " ttft"] = (r["median_ttft_s"], None)
    for c in summary(d / "bench-cells.out") or []:
        out[c["cell"]] = (c["median_decode_tok_s"], c["shas"][0] if len(set(c["shas"])) == 1 else "mixed")
    acc = d / "accept-frozen.txt"
    if acc.is_file():
        try:
            out["frozen tokens/round"] = (json.loads(acc.read_text()).get("tokens_per_round"), None)
        except ValueError:
            pass
    return out


def main() -> int:
    dirs = [Path(p) for p in sys.argv[1:]]
    if len(dirs) < 2:
        print(__doc__)
        return 2
    data = [load(d) for d in dirs]
    keys = list(data[0])
    print("| metric | " + " | ".join(d.name for d in dirs) + " |")
    print("|---|" + "---:|" * len(dirs))
    for k in keys:
        base = data[0].get(k, (None, None))
        cells = [f"{base[0]:.3g}" if isinstance(base[0], float) else str(base[0])]
        for dd in data[1:]:
            v = dd.get(k, (None, None))
            if v[0] is None or base[0] in (None, 0):
                cells.append("-")
                continue
            delta = (v[0] / base[0] - 1) * 100
            same = "" if v[1] is None or base[1] is None else (" =" if v[1] == base[1] != "mixed" else " ≠")
            cells.append(f"{v[0]:.3g} ({delta:+.1f}%){same}")
        print(f"| {k} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
