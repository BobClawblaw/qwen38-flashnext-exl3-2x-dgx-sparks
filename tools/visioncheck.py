#!/usr/bin/env python3
"""Image and video input on the pair (VISION=1): three probes with known answers, greedy, thinking off.

1. A drawn PNG (red circle left, blue square right; stdlib only): the reply must name both shapes and colours.
2. A photo (--photo PATH, JPEG/PNG): the reply is printed for the reader; --expect WORD... makes it a check.
3. A video (--video PATH, anything FFmpeg decodes; needs PyAV in the image): printed, or checked with --expect-video.

Usage: tools/visioncheck.py [--url http://127.0.0.1:8888/v1/chat/completions] [--model ID]
                            [--photo FILE --expect WORD ...] [--video FILE --expect-video WORD ...]
Exit code 1 when a checked probe fails (a server without VISION=1 answers 400 to all of them).
"""
import argparse
import base64
import json
import mimetypes
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

W, H = 480, 240


def png() -> bytes:
    """A white 480 x 240 RGB image: a red circle on the left, a blue square on the right."""

    rows = []
    for y in range(H):
        row = bytearray([0])
        for x in range(W):
            if (x - 120) ** 2 + (y - 120) ** 2 <= 80 ** 2:
                row += bytes((220, 30, 30))
            elif 280 <= x < 440 and 40 <= y < 200:
                row += bytes((30, 60, 220))
            else:
                row += bytes((255, 255, 255))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b""))


def ask(url: str, model: str, part: dict, question: str, max_tokens: int = 200) -> tuple[str, dict, float]:
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [part, {"type": "text", "text": question}]}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        out = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}")
    return (out["choices"][0]["message"].get("content") or "").strip(), out, time.time() - t0


def data_url(path: str, kind: str) -> str:
    mime = mimetypes.guess_type(path)[0] or ("video/mp4" if kind == "video" else "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def report(name: str, text: str, out: dict, seconds: float, expect: list[str] | None) -> bool:
    usage, tf = out.get("usage", {}), out.get("tensorfold", {})
    n, decode_s = usage.get("completion_tokens") or 0, tf.get("decode_s") or 0
    rate = f"{n / decode_s:.0f} tok/s" if n and decode_s else "? tok/s"
    print(f"{name}: {usage.get('prompt_tokens')} prompt tokens, {n} reply tokens in {seconds:.1f}s "
          f"(prefill {tf.get('prefill_s', '?')}s incl. the tower, {rate}, sha {tf.get('token_sha', '?')})")
    print(f"  reply: {text}")
    if expect is None:
        return True
    low = text.lower()
    missing = [w for w in expect if w.lower() not in low]
    print(f"  {'OK' if not missing else 'FAILED: missing ' + ', '.join(missing)}")
    return not missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    p.add_argument("--model", default="turboderp/Qwen3.8-Flash-Next-exl3")
    p.add_argument("--photo")
    p.add_argument("--expect", nargs="*")
    p.add_argument("--video")
    p.add_argument("--expect-video", nargs="*")
    a = p.parse_args()
    ok = True
    shapes = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png()).decode()}}
    ok &= report("shapes", *ask(a.url, a.model, shapes,
                                "Which shapes are in this image, and what colour is each? One sentence."),
                 ["red", "circle", "blue", "square"])
    if a.photo:
        part = {"type": "image_url", "image_url": {"url": data_url(a.photo, "image")}}
        ok &= report("photo", *ask(a.url, a.model, part, "Describe this picture in one sentence."), a.expect)
    if a.video:
        part = {"type": "video_url", "video_url": {"url": data_url(a.video, "video")}}
        ok &= report("video", *ask(a.url, a.model, part, "What happens in this video? One sentence."), a.expect_video)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
