#!/usr/bin/env python3
"""Cold and warm prompt fill at long context against an OpenAI-compatible server (stdlib only).

For each target length the filler (seeded per length and per --run) is the system message, and three
requests follow, streamed, thinking off, greedy:
  cold:   filler + a one-sentence question, max_tokens 16 (no earlier request shares the filler)
  resend: the identical request again (a server that keeps prompt states resumes it)
  story:  filler + a request for a 300-word story, max_tokens --decode-tokens: resumes at the message
          boundary on a server that keeps one there, and times natural-prose decode at that context.
Reports prompt tokens (from usage), cold TTFT, cold prefill tok/s = prompt tokens / TTFT, resend TTFT,
story TTFT and story decode tok/s = (completion - 1) / (last - first content token).
A throwaway prompt of the shortest length runs first and is not reported.
Prints `ROW {json}` per length and `SUMMARY [...]`.

  python3 tools/bench_prefill.py --lengths 8192 32768 65536 131072
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("amber brittle copper dusky eager feral gilded hollow ivory jagged knotted languid mossy narrow ochre "
         "pallid quiet russet silent tawny umber velvet wary woven lantern orchard ledger kettle bridge harbor "
         "quarry meadow chimney anvil cellar compass lighthouse granary loom mill parcel quill ridge saddle "
         "mended carried weighed painted counted guarded sealed traded repaired measured polished copied").split()
CHARS_PER_TOKEN = 3.95  # this filler under the Qwen3.8 tokenizer (measured); usage reports the real count


def filler(seed: int, tokens: int) -> str:
    rng = random.Random(seed)
    out, n = [], 0
    while n < tokens * CHARS_PER_TOKEN:
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))).capitalize() + f" ({rng.randint(0, 99999)})."
        out.append(s)
        n += len(s) + 1
    return " ".join(out)


COLD_Q = "In one sentence, what is the text above about?"
WARM_Q = "Now set the text aside and write a 300-word story about a lighthouse keeper who finds a message in a bottle."


def one(url, model, prompt, question, max_tokens):
    body = {"model": model, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, last, usage = None, None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    last = time.perf_counter() - t0
                    if ttft is None:
                        ttft = last
    details = (usage or {}).get("prompt_tokens_details") or {}
    n = (usage or {}).get("completion_tokens") or 0
    dec = (n - 1) / (last - ttft) if n > 1 and last and ttft and last > ttft else None
    return {"prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": details.get("cached_tokens"),
            "ttft_s": ttft, "decode_tok_s": dec, "completion_tokens": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    p.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536, 131072])
    p.add_argument("--run", type=int, default=0, help="salt for the filler: change it to get a cold prompt on a server that already saw this one; keep it fixed across boots so configs compare the same text")
    p.add_argument("--decode-tokens", type=int, default=256)
    a = p.parse_args()
    rows = []
    # One throwaway long prompt first, sharing no filler with the measured ones, so a server's first-use
    # cost (kernel autotune, graph capture for long shapes) does not land in the first cold row.
    one(a.url, a.model, f"Warm-up {a.run}.\n" + filler(-1 - a.run, min(a.lengths)), COLD_Q, 1)
    for n in a.lengths:
        prompt = f"Run {a.run}, length {n}.\n" + filler(n * 1000 + a.run, n)
        cold = one(a.url, a.model, prompt, COLD_Q, 16)
        resend = one(a.url, a.model, prompt, COLD_Q, 16)
        story = one(a.url, a.model, prompt, WARM_Q, a.decode_tokens)
        row = {"target": n, "prompt_tokens": cold["prompt_tokens"], "cold_ttft_s": cold["ttft_s"],
               "cold_prefill_tok_s": (cold["prompt_tokens"] / cold["ttft_s"]) if cold["ttft_s"] else None,
               "resend_ttft_s": resend["ttft_s"], "resend_cached_tokens": resend["cached_tokens"],
               "story_ttft_s": story["ttft_s"], "story_cached_tokens": story["cached_tokens"],
               "story_decode_tok_s": story["decode_tok_s"], "story_completion_tokens": story["completion_tokens"]}
        print("ROW " + json.dumps(row), flush=True)
        rows.append(row)
    print("SUMMARY " + json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
