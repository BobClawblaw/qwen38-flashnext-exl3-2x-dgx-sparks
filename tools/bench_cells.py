#!/usr/bin/env python3
"""Longer single-stream decode cells against an OpenAI-compatible server (stdlib only).

bench_decode.py is the frozen ruler shared with the sibling vLLM recipe: 100-token prose and a
200-token count. These cells add longer replies of other kinds, so a lever that only helps the
count cannot pass for a general win. Each cell runs REPS times at c=1, streamed, thinking off.
Per cell: median decode tok/s ((completion - 1) / time after the first content token), median TTFT,
median completion tokens, the reply sha256 of each run (greedy cells must repeat exactly), and the
MTP acceptance of the cell from /health deltas when --base points at a TensorFold server.
Prints one `CELL {json}` line per cell and a final `SUMMARY [...]`.
"""
import argparse
import hashlib
import json
import statistics
import time
import urllib.request

CELLS = {
    "code": ("Write a Python module that parses ISO-8601 durations such as P3DT4H5M6S into a "
             "datetime.timedelta. Include type hints, a docstring, input validation with clear errors, "
             "and five unittest test cases. Output only the code.", 512, None),
    "prose_long": ("Explain to a new engineer how a hash map handles collisions. Compare separate chaining "
                   "and open addressing, cover load factor and resizing, and finish with practical advice. "
                   "Write about 400 words in plain paragraphs.", 512, None),
    "json": ("Return a JSON array of 15 fictional employees. Each object has id (int), name, email, "
             "department (one of Sales, Engineering, Finance, Support), start_date (YYYY-MM-DD) and "
             "skills (array of 3 strings). Output only the JSON.", 768, None),
    "multilingual": ("Schreibe eine kurze Geschichte (etwa 250 Wörter) über einen Leuchtturmwärter, der "
                     "einen unerwarteten Besucher bekommt. Danach fasse sie in drei Sätzen auf Französisch "
                     "zusammen.", 512, None),
    "chat_sampled": ("Give me five creative ideas for a weekend project that teaches a teenager basic "
                     "electronics, with one paragraph each.", 512,
                     {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 1234}),
}


def health(base):
    if not base:
        return None
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=10) as r:
            return json.load(r)
    except Exception:
        return None


def one(url, model, prompt, max_tokens, sampling):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(sampling or {"temperature": 0})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_end = None
    parts, usage = [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    parts.append(piece)
                    t_end = time.perf_counter()
    text = "".join(parts)
    n = (usage or {}).get("completion_tokens")
    if n is None or t_first is None:
        raise RuntimeError("stream had no usage or no content")
    dec = (n - 1) / (t_end - t_first) if n > 1 and t_end > t_first else float("nan")
    return {"ttft_s": t_first - t0, "decode_tok_s": dec, "completion_tokens": n,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:12]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    p.add_argument("--base", default="", help="server root for /health acceptance deltas (TensorFold)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cells", nargs="+", default=list(CELLS))
    a = p.parse_args()
    summary = []
    for name in a.cells:
        prompt, mt, sampling = CELLS[name]
        h0 = health(a.base)
        runs = [one(a.url, a.model, prompt, mt, sampling) for _ in range(a.reps)]
        h1 = health(a.base)
        cell = {"cell": name, "n": len(runs),
                "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in runs),
                "median_ttft_s": statistics.median(r["ttft_s"] for r in runs),
                "median_completion_tokens": statistics.median(r["completion_tokens"] for r in runs),
                "shas": [r["sha"] for r in runs], "runs": runs}
        # Greedy replies must repeat byte for byte; a sampled cell carries its seed, so it should too.
        cell["repeat_ok"] = len(set(cell["shas"])) == 1
        if h0 and h1 and h1.get("rounds_total", 0) > h0.get("rounds_total", 0):
            dr = h1["rounds_total"] - h0["rounds_total"]
            cell["tokens_per_round"] = round(1 + (h1["accepted_total"] - h0["accepted_total"]) / dr, 3)
        print("CELL " + json.dumps(cell), flush=True)
        summary.append({k: v for k, v in cell.items() if k != "runs"})
    print("SUMMARY " + json.dumps(summary, indent=1))
    bad = [c["cell"] for c in summary if not c["repeat_ok"]]
    if bad:
        raise SystemExit(f"replies did not repeat across runs: {bad}")


if __name__ == "__main__":
    main()
