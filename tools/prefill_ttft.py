#!/usr/bin/env python3
"""Cold prefill speed: one fresh random-word prompt per size, streamed, time to the first content token.

Prompt tokens come from the usage block, so the prefill rate is prompt_tokens / TTFT; the reply is 4 tokens with thinking off.
Usage: tools/prefill_ttft.py [--url URL] [--model ID] [--words 6000 24000 96000]
"""
import argparse, json, os, random, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
p.add_argument("--words", type=int, nargs="+", default=[6000, 24000, 96000])
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
random.seed(a.seed)
words = open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else [f"w{i}" for i in range(5000)]
for nw in a.words:
    prompt = "Read this list and then reply with the single word done.\n\n" + " ".join(random.choice(words) for _ in range(nw))
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if ttft is None and d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.time() - t0
            if d.get("usage"):
                usage = d["usage"]
    pt = usage["prompt_tokens"]
    print(f"{nw} words: prompt_tokens={pt} ttft={ttft:.2f}s prefill={pt/ttft:.0f} tok/s", flush=True)
