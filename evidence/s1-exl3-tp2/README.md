# s1 · EXL3 4.05 bpw on two ranks, 2026-10-07

The development build (TensorFold 0.6.6 worktree with the patch, mounted over the image; port 8889) and the shipped image started by `run.sh` (port 8888) are the same code; files say which.

- `cmp-two-ranks.txt`, `cmp-one-gpu.txt`, `texts-*.json`, `two-ranks-vs-one-gpu.txt`: six greedy prompts, drafted and `"draft": false`, on two ranks (serial profile, 262k) and on one Spark (65k window); shas, tokens, tok/s and the texts.
- `bench-serial-c1.out`, `bench-serial-c1-image.out`: the frozen ruler, serial profile, development build and shipped image.
- `bench-concurrent-c1-c8-before-part8.out`: concurrent profile with vision, 1 / 4 / 8 users; that boot stopped at 16 users on the 128-row cap part 8 lifts (the traceback is in the file).
- `bench-concurrent-c1-c16.out`, `bench-concurrent-c1-c16-image.out`: 1 and 16 users after part 8, development build and shipped image (the README's generated table).
- `visioncheck.txt`, `visioncheck-image.txt`: the shapes probe on two ranks with the converted tower.
- `needle.log`: the 1M profile (YaRN x4): short prompts (drafted equals undrafted), needles at 7.5k, 286k and 959k prompt tokens, all found, with the prefill rates.
- `quality/`: `t2/` (GSM8K, IFEval, tools, rep4, effort; concurrent profile, four workers), `t2-json/` (serial profile), `score-vs-mlx.txt` (item by item against the MLX pair's run of the same day).
- `run-concurrent-vision.log`, `run-serial.log`: `run.sh` on the shipped image, both profiles, start to ready (the one-time vision tower conversion in the first). `run-serial-first-attempt-reg-mr.log`: the restart that followed a stop within seconds and failed NCCL's first memory registration on the worker; `run.sh` now waits 30 s after stopping a rank.
- `measure-concurrent.log`, `measure-serial.log`: the measurement sessions' consoles.
