#!/usr/bin/env python3
"""N concurrent streams, each on its own long filler, decoding together (stdlib only).
Each stream's system message is a distinct seeded filler of --prompt-tokens (bench_prefill's generator), the user
asks for a short story, greedy, thinking off, --tokens reply tokens, streamed. All N start together. Reports per
stream: prompt tokens (usage), TTFT, decode tok/s; and aggregates: total reply tokens over the wall from the first
content token to the last, and steady tok/s: chunks that arrived while every stream was decoding (after the last
first token, before the first finish), so no prompt was filling inside those rounds. A second run with the same
--seed finds the prompts cached and measures decode-together alone.
  python3 tools/bench_longctx_concurrent.py --streams 4 --prompt-tokens 250000 --tokens 256
Prints `STREAM {json}` lines and `SUMMARY {json}`."""
import argparse, json, sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_prefill import filler  # noqa: E402

ASKS = {"story": "Write a 400-word story about a lighthouse keeper.",
        "code": "Write a Python module implementing an LRU cache class with get, put and a capacity limit, plus pytest tests for it."}


def stream(url, model, system, tokens, out, idx, ask="story"):
    body = {"model": model, "stream": True, "max_tokens": tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": ASKS[ask]}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; n = 0; usage = None; times = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                c = json.loads(line[6:])
                if c.get("usage"):
                    usage = c["usage"]
                for ch in c.get("choices") or []:
                    if ch.get("delta", {}).get("content"):
                        now = time.perf_counter(); n += 1; times.append(now)
                        first = first or now; last = now
    except Exception as e:  # noqa: BLE001
        out[idx] = {"stream": idx, "error": str(e)[:200]}; return
    out[idx] = {"stream": idx, "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens", n), "ttft_s": round(first - t0, 2) if first else None,
                # chunks carry several tokens (MTP), so scale by the reply's tokens per chunk
                "decode_tok_s": round((n - 1) * ((usage or {}).get("completion_tokens", n) / n) / (last - first), 1)
                if n > 1 and last > first else None,
                "t_first": first, "t_last": last, "t_chunks": times, "chunks": n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    ap.add_argument("--streams", type=int, default=4); ap.add_argument("--prompt-tokens", type=int, default=250000)
    ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task", choices=sorted(ASKS), default="story")
    a = ap.parse_args()
    systems = [filler(a.seed * 100 + i, a.prompt_tokens) for i in range(a.streams)]
    out = [None] * a.streams
    ths = [threading.Thread(target=stream, args=(a.url, a.model, systems[i], a.tokens, out, i, a.task)) for i in range(a.streams)]
    wall0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - wall0
    ok = [s for s in out if s and "error" not in s and s["t_first"]]
    for s in out: print("STREAM", json.dumps({k: v for k, v in s.items() if not k.startswith("t_")}))
    total = sum(s["completion_tokens"] for s in ok)
    span = max(s["t_last"] for s in ok) - min(s["t_first"] for s in ok) if ok else 0
    lo, hi = max(s["t_first"] for s in ok), min(s["t_last"] for s in ok)
    steady = None
    if ok and hi > lo:
        # chunks that arrived while every stream was decoding (no prompt still filling), scaled by each
        # stream's tokens per chunk: the decode-together rate, not an average over prefill-shared time
        got = sum(sum(1 for t in s["t_chunks"] if lo <= t <= hi) * s["completion_tokens"] / s["chunks"] for s in ok if s["chunks"])
        steady = round(got / (hi - lo), 1)
    print("SUMMARY", json.dumps({"task": a.task, "streams": a.streams, "prompt_tokens_target": a.prompt_tokens, "tokens": a.tokens,
                                 "ok": len(ok), "wall_s": round(wall, 1), "prompt_tokens_total": sum(s["prompt_tokens"] or 0 for s in ok),
                                 "aggregate_tok_s_first_to_last": round(total / span, 1) if span else None,
                                 "steady_tok_s_all_decoding": steady, "overlap_s": round(hi - lo, 1) if ok else None}))
main()
