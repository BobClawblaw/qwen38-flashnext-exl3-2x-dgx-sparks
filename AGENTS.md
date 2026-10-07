# AGENTS.md — Qwen3.8-Flash-Next · EXL3 · TensorFold · 2× DGX Spark

Serve `turboderp/Qwen3.8-Flash-Next-exl3` (branch `4.05bpw_h6_ng6`, snapshot `55a732e`) at TP=2 with the TensorFold engine (`cb2ebf0`, v0.6.6, plus `docker/patches/flashnext-exl3-0.6.6.patch`) inside `nvcr.io/nvidia/pytorch:26.07-py3` (digest-pinned in `docker/Dockerfile`). Not vLLM, not ExLlamaV3's own server. The MLX 4-bit recipe for the same pair is a separate repo (`BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks`); one repo per checkpoint format.

Humans read [README.md](README.md).

## Standing orders

1. **PR only.** Commit on `agent/**` branches. Open a PR against `main`. Never push `main`. Never merge. Never force-push.
2. **Evidence.** Every number in `README.md` has a file under `evidence/` that proves it. No file, no number. Never gitignore `evidence/`. `recipe.yaml` names the file per measured row; `python3 kit/render.py --check` lists the rows without one.
3. **Exclusive GPUs.** One serve per pair. Before a boot, stop any other GPU serve on both Sparks with its own stop script (gracefully: `docker stop`, not `docker rm -f`) and restore it afterwards with its smokes. `run.sh` refuses to start next to a foreign GPU container and never removes it.
4. **Validate first.** `VALIDATE_ONLY=1 ./run.sh` before any serve. It needs no Docker.
5. **Source of truth.** Values live in `recipe.yaml`. Edit it and run `python3 kit/render.py`. Never edit inside the generated markers by hand.
6. **Voice.** Short sentences. Numbers first. No marketing language. No emojis.
7. **Secrets.** Never commit tokens, Tailscale IPs, or auth headers.

## Working rules

- Change one knob at a time against the gate (`tools/session_gate.sh EVDIR`, or `tools/sweep.sh SPEC EVROOT` for a list). Compare with `tools/compare.py BASE CAND`. Session-to-session noise on this pair is about ±2.5% per cell: a lever that does not beat it, or that regresses another cell, is reverted and recorded.
- `bench_decode.py` is byte-identical to the vLLM sibling's frozen ruler (sha256 `6a9c64bd…`, enforced by `tests/`). Do not edit it. TensorFold-specific measurement lives in `tools/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. The default lists both live ones (`rocep1s0f1,roceP2p1s0f1`), as TensorFold's runbook recommends; this cluster's `.env.cluster` pins one (`rocep1s0f1`) after a rank-1 boot failure with both.
- Both ranks must get identical engine flags (`--context`, `--kv-dtype`, `--mtp-drafts`, `--mtp-confidence`, `--parallel`); TensorFold all-gathers them at startup and refuses a mismatch. `run.sh` builds both argvs from one `serve_args()`; keep it that way.
- Default thinking is off (`--no-thinking`), matching the vLLM sibling. Requests override it with `chat_template_kwargs.enable_thinking`.
- The image is built locally from `docker/Dockerfile`; `run.sh` refuses an image whose `tensorfold.sha` label differs from `TF_SHA`, and copies the head's image to the worker when the worker's image ID differs.

## Known engine limits (TensorFold 0.6.6 with the patch, two ranks)

- Two ranks read an EXL3 pack through patch part 6: dense projections cut on 128-wide Hadamard blocks, experts held whole (256 a rank, the shared expert alternating by layer), partial sums gathered. The grouped EXL3 kernel takes one expert width a layer, so the experts are never width-split; one stream pays the per-token imbalance (up to 10 of 10 experts on one rank), sixteen users do not. NVFP4 checkpoints still run on one GPU.
- `PROFILE=long` is YaRN x4 from the profile folder's `config.json` (part 7): a static factor that also shifts short prompts. Keep ordinary work on `serial` or `concurrent`. The EXL3 prompt path runs at about 0.4x the MLX path's rate and slows as the prompt grows; a 1M prompt is a long wait, measured in `evidence/s1-exl3-tp2/needle.log`.
- `PROFILE=concurrent` sets `PARALLEL=16` (part 8 lets the EXL3 path take a 256-row decode window). Grammars (`response_format`, `guided_*`) and logprobs are refused on two ranks with `--parallel`; the serial default serves `response_format`. `VISION=1` works on either profile after the one-time tower conversion.
- The only patch is `docker/patches/flashnext-exl3-0.6.6.patch`. Never edit it by hand: edit a clean v0.6.6 checkout, run `tests/cuda/test_qwen4_exp_exl3.py` and `tests/test_flashnext_yarn_host.py` inside the image with the pack, regenerate the diff, update both pins (`run.sh`, `recipe.yaml`).
- After a large download or image load on a node, NCCL's first memory registration can fail (`ibv_reg_mr: Cannot allocate memory`); `run.sh` drops caches and compacts memory before each start, which clears it. A `docker run` outside `run.sh` must do the same.
