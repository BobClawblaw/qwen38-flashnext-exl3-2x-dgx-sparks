#!/usr/bin/env python3
"""Image replies on two ranks are exact: drafted equals one-token decoding, concurrent equals alone, text unmoved.

Cells (greedy, thinking off, reply token sha from the engine's ``tensorfold`` block):
  A. each image request drafted against the same request with "draft": false
  B. four image requests sent together against each sent alone
  C. a text request sent beside the four image requests against the same text request alone
Usage: tools/vision_exact.py --photo FILE [--url ...] [--model ...] [--tokens 160]
"""
import argparse
import base64
import concurrent.futures as cf
import json
import mimetypes
import sys
import time
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__file__))
from visioncheck import png  # noqa: E402


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    tf = out.get("tensorfold", {})
    return {"sha": tf.get("token_sha"), "tokens": out["usage"]["completion_tokens"], "wall": time.time() - t0,
            "prefill_s": tf.get("prefill_s"), "decode_s": tf.get("decode_s"), "rounds": tf.get("rounds"),
            "text": (out["choices"][0]["message"].get("content") or "")[:70], "keys": sorted(tf)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    p.add_argument("--photo", required=True)
    p.add_argument("--tokens", type=int, default=160)
    a = p.parse_args()
    mime = mimetypes.guess_type(a.photo)[0] or "image/jpeg"
    photo = f"data:{mime};base64," + base64.b64encode(open(a.photo, "rb").read()).decode()
    shapes = "data:image/png;base64," + base64.b64encode(png()).decode()
    questions = [(shapes, "Which shapes are in this image, and what colour is each? Then explain how you can tell."),
                 (photo, "Describe this picture in detail: the sky, the water, and anything standing in it."),
                 (photo, "Write a short poem about this scene."),
                 (photo, "List five concrete details you can see, one per line.")]

    def body(url, text, draft=True):
        b = {"model": a.model, "max_tokens": a.tokens, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False},
             "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                       {"type": "text", "text": text}]}]}
        if not draft:
            b["draft"] = False
        return b
    text_body = {"model": a.model, "max_tokens": a.tokens, "temperature": 0,
                 "chat_template_kwargs": {"enable_thinking": False},
                 "messages": [{"role": "user", "content": "Explain in a paragraph why the sky is blue."}]}

    ok = True
    print("A. drafted against one token a round")
    alone = []
    for i, (url, text) in enumerate(questions):
        d, s = post(a.url, body(url, text)), post(a.url, body(url, text, draft=False))
        alone.append(d)
        same = d["sha"] == s["sha"] and d["tokens"] == s["tokens"]
        ok &= same
        print(f"  q{i}: drafted {d['tokens']} tok in {d['wall']:.1f}s (prefill {d['prefill_s']}s, {d['rounds']} rounds) "
              f"sha {d['sha']} | plain {s['tokens']} tok in {s['wall']:.1f}s sha {s['sha']} | {'EQUAL' if same else 'DIFFERENT'}")
    print(f"  stats keys: {alone[0]['keys']}")
    print("B. four image requests together against alone, C. text beside them against text alone")
    text_alone = post(a.url, text_body)
    with cf.ThreadPoolExecutor(5) as pool:
        futures = [pool.submit(post, a.url, body(url, text)) for url, text in questions]
        futures.append(pool.submit(post, a.url, text_body))
        together = [f.result() for f in futures]
    for i, (t, al) in enumerate(zip(together[:4], alone)):
        same = t["sha"] == al["sha"] and t["tokens"] == al["tokens"]
        ok &= same
        print(f"  q{i}: together {t['tokens']} tok in {t['wall']:.1f}s (prefill {t['prefill_s']}s) sha {t['sha']} "
              f"| alone {al['wall']:.1f}s | {'EQUAL' if same else 'DIFFERENT'}")
    t = together[4]
    same = t["sha"] == text_alone["sha"] and t["tokens"] == text_alone["tokens"]
    ok &= same
    print(f"  text: beside images {t['tokens']} tok in {t['wall']:.1f}s sha {t['sha']} | alone {text_alone['wall']:.1f}s "
          f"sha {text_alone['sha']} | {'EQUAL' if same else 'DIFFERENT'}")
    print("vision_exact:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
