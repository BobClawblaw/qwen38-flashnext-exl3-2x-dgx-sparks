#!/usr/bin/env python3
"""Failure-mode probes for two-rank `--parallel` (the PR-141 profile). Stdlib only.

1. leave: 4 greedy streams start together; one client closes its connection after its first content
   chunks. The other 3 must finish with the same reply as when they run alone.
2. grammar: a `response_format` json_schema request must be refused with HTTP 400 (two-rank
   concurrency serves no grammars), not hang a rank.
3. after: a fresh greedy request must still succeed and equal its earlier solo reply.
Exit 0 only if all three hold. Check the rank logs for OutOfStep / Traceback afterwards
(tools/session_gate.sh writes engine-needles.txt).

  python3 tools/probe_parallel.py --url http://127.0.0.1:8000
"""
import argparse
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request

PROMPTS = ["Write a haiku about rivers.", "List five prime numbers above 100, comma separated.",
           "Explain recursion to a child in three sentences.", "Name three moons of Jupiter and one fact each."]


def post(base, body, stream, leave_after=None):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r)["choices"][0]["message"].get("content") or ""
        parts, n = [], 0
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    parts.append(piece)
                    n += 1
                    if leave_after and n >= leave_after:
                        return None  # closing the response drops the connection mid-stream
        return "".join(parts)


def body(model, prompt, stream=True, **extra):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 160, "temperature": 0,
         "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(extra)
    return b


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    a = p.parse_args()
    base = a.url.rstrip("/")
    ok = True

    solo = {i: sha(post(base, body(a.model, pr), True)) for i, pr in enumerate(PROMPTS)}
    print("solo", json.dumps(solo))

    got = {}

    def run(i):
        try:
            text = post(base, body(a.model, PROMPTS[i]), True, leave_after=3 if i == 0 else None)
            got[i] = None if text is None else sha(text)
        except Exception as exc:  # noqa: BLE001 - reported below
            got[i] = f"error {type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Client 0 must really have left (None), and the other three must equal their solo replies.
    leave_ok = got.get(0) is None and all(got.get(i) == solo[i] for i in range(1, len(PROMPTS)))
    print("leave", json.dumps(got), "PASS" if leave_ok else "FAIL")
    ok &= leave_ok

    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    try:
        post(base, body(a.model, "Return n=3 as JSON.", False,
                        response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": schema}}), False)
        code = 200
    except urllib.error.HTTPError as exc:
        code = exc.code
    grammar_ok = code == 400
    print("grammar", code, "PASS" if grammar_ok else "FAIL (want 400 under two-rank --parallel)")
    ok &= grammar_ok

    after = sha(post(base, body(a.model, PROMPTS[2]), True))
    after_ok = after == solo[2]
    print("after", after, "PASS" if after_ok else "FAIL")
    ok &= after_ok
    print("PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
