#!/usr/bin/env python3
"""Lone requests with a different prompt each time (stdlib only).

bench_decode.py and tools/bench_cells.py repeat each prompt, so a server that keeps per-prompt state
(prompt states, CUDA graphs bound to a cache slot) is measured mostly warm. This sends N distinct prompts one
after another at c=1, greedy, thinking off, --tokens each (ignore_eos off), and reports the median decode
tok/s, the median TTFT and every request's numbers. Prints `SUMMARY {json}`.

  python3 tools/bench_distinct.py --n 12 --tokens 256
"""
import argparse
import json
import statistics
import time
import urllib.request

TOPICS = ["a lighthouse keeper", "a robot gardener", "the first day of a new job", "a lost umbrella", "a chess match",
          "a city without cars", "an old bakery", "a mountain rescue", "a time traveller's mistake", "a desert well",
          "a jazz musician", "a library at night", "a ship's cat", "a winter market", "a broken compass", "a beekeeper"]


def one(url, model, prompt, tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    last = time.perf_counter()
                    first = first or last
    n = (usage or {}).get("completion_tokens", 0)
    return {"ttft_s": first - t0, "decode_tok_s": (n - 1) / (last - first) if n > 1 and last > first else None,
            "completion_tokens": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--tokens", type=int, default=256)
    a = p.parse_args()
    rows = []
    for i in range(a.n):
        r = one(a.url, a.model, f"Write a 250-word story about {TOPICS[i % len(TOPICS)]}. (#{i})", a.tokens)
        print("REQ " + json.dumps(r), flush=True)
        rows.append(r)
    dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    print("SUMMARY " + json.dumps({"n": len(rows), "median_decode_tok_s": statistics.median(dec),
                                   "median_ttft_s": statistics.median(r["ttft_s"] for r in rows)}))


if __name__ == "__main__":
    main()
