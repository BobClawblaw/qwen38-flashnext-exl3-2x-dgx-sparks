# Qwen3.8-Flash-Next · EXL3 4.05 bpw · TensorFold 0.6.6 · 2× DGX Spark

Serve [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (the `4.05bpw_h6_ng6` pack) across two NVIDIA DGX Spark (GB10, 128 GB) nodes at tensor-parallel 2 with the [TensorFold](https://github.com/ashhart/TensorFold) engine: the full 262,144-token window on both ranks, a 1M-token profile, sixteen concurrent streams, MTP and copy drafts, image and video input, tool calls, an OpenAI-compatible API, and a systemd unit that brings the pair up at boot.

- **Why this pack.** TensorFold's two-rank Flash Next engine read one checkpoint: the uniform MLX 4-bit, whose quality cost is documented (TensorFold issue #85: 10 to 12 points on a 92-question suite against a mixed 4/8-bit pack; +20.6% perplexity against +1.3%). turboderp's EXL3 packs are the best-measured 4-bit weights for this model (KL 0.0177 at 3.05 bpw; the 4.05 pack lower still), and upstream's own check found the 3.05 pack ahead of the MLX 4-bit on next-token loss (0.851 against 1.057 on the same tokens). TensorFold served them on one GPU only.
- **What this recipe adds.** One patch, built into the image ([`docker/patches/flashnext-exl3-0.6.6.patch`](docker/patches/flashnext-exl3-0.6.6.patch)): EXL3 packs on two ranks (the dense projections cut on whole 128-wide Hadamard blocks, the 640-wide experts held whole, 256 a rank, their picks remapped and partial sums gathered), YaRN from `config.json` for the 1M profile, and a 16-stream decode window the EXL3 path lacked. Plus the MLX recipe's parts, byte for byte: `--tool-system`, declared-parameter tool calls, copy drafts, images and video on two ranks. [`docker/patches/README.md`](docker/patches/README.md), [`docs/two-rank-exl3.md`](docs/two-rank-exl3.md), [`docs/one-million-context.md`](docs/one-million-context.md).
- **Where it stands.** The first two-rank serve of an EXL3 Flash Next pack. 41 GiB a rank. Drafted replies equal undrafted ones on every prompt; two ranks equal the same pack on one GPU to the last byte on four of six greedy replies and diverge at a near-tie on the other two. One user decodes 57 prose / 231-239 structured / 105-115 code tok/s on the serial profile, about 1.3x the pack on one Spark; sixteen users 369 prose and 798 structured tok/s aggregate. Slower than the MLX pair at one and eight users (its experts split by width; EXL3's cannot), ahead of it on structured replies at sixteen, and the better weights at every count.
- **Lineage.** The MLX recipe ([BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks](https://github.com/BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks)), itself from [sfxnz's base recipe](https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark); `run.sh`'s guards, the frozen ruler and the quality harness come from there.

Qwen3.8-Flash-Next is a ~180B MoE (512 experts, top-10) with Gated DeltaNet, sparse attention, hyper-connections, hashed n-gram (PLE) tables and one MTP layer. The pack is EXL3 (ExLlamaV3's trellis format, `mul1` codebook) at 4.05 bits a weight on every linear, 6 bits on the head and the n-gram table, 4 on the MTP head; 108 GB on disk, the 33 GB n-gram table memory-mapped and locked on each node. Pinned snapshot: `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`.

## Measured on this pair (TensorFold 0.6.6 with the patch, two ranks)

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: streamed greedy, thinking off, max_tokens 200 (prose ends at EOS near 100), 3-run median; TensorFold 0.6.6 with the patch at TP=2, PROFILE=concurrent (PARALLEL=16) with VISION=1, context 262144, kv int8, MTP up to 15 drafts at confidence 0.70, one RoCE HCA; the shipped image, started by run.sh.

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 52.9 | 52.9 | 0.06 s |
| prose | 16 | 22.9 | 366.7 | 0.18 s |
| structured | 1 | 225.4 | 225.2 | 0.09 s |
| structured | 16 | 50.1 | 801.2 | 0.36 s |
<!-- END generated measured -->

Receipts: [`evidence/s1-exl3-tp2/`](evidence/s1-exl3-tp2/). `bench_decode.py` is the sibling recipes' frozen ruler, byte for byte, so the rows compare with theirs.

### One to sixteen users, and the serial profile

| Cell | Serial profile (`PARALLEL=1`) | Concurrent profile with vision (`PARALLEL=16`, `VISION=1`) |
|---|---:|---:|
| prose, 1 user | 56.7-57.3 | 52.9-54.6 |
| structured, 1 user | 231.2-239.3 | 225.4-234.1 |
| prose, 4 users, aggregate | | 161 (40 a stream) |
| prose, 8 users, aggregate | | 264 (33 a stream) |
| prose, 16 users, aggregate | | 367-369 (23 a stream) |
| structured, 16 users, aggregate | | 798-801 (50 a stream) |

tok/s. The concurrent column's ranges are two boots of the same patch: the shipped image started by `run.sh` (the generated table above, `bench-concurrent-c1-c16-image.out`) and the development build (`bench-concurrent-c1-c16.out`); 4 and 8 users come from `bench-concurrent-c1-c8-before-part8.out` (that boot stopped at 16 users on the cap part 8 lifts); the serial column is `bench-serial-c1-image.out` (the shipped image) and `bench-serial-c1.out` (the development build).

### Drafted replies, six prompts, one user

| Prompt | Two ranks, drafted | Two ranks, `"draft": false` | One GPU, drafted | One GPU, `"draft": false` |
|---|---:|---:|---:|---:|
| prose (why the sky is blue) | 74.4 | 40.8 | 54.1 | 31.7 |
| Python function with docstring | 107.2 | 41.1 | 81.1 | 32.5 |
| TCP against UDP, a list | 69.2 | 41.2 | 56.4 | 32.6 |
| a French translation | 60.0 | 36.1 | 44.9 | 27.5 |
| a bash script | 115.1 | 41.1 | 84.9 | 32.5 |
| a short arithmetic problem | 102.7 | 40.7 | 85.9 | 32.4 |

tok/s, greedy, max_tokens 300, the serial profile at 262k on two ranks and the same pack on one Spark at a 65k window (`cmp-two-ranks.txt`, `cmp-one-gpu.txt`). On every row the drafted reply has the undrafted reply's text and token count.

### Exactness

- Drafted against `"draft": false`: identical on 6 of 6 prompts on the serial profile, 3 of 3 on the 1M profile, and on the one-GPU reference (`two-ranks-vs-one-gpu.txt`).
- Two ranks against one GPU, the same pack, greedy: 4 of 6 replies identical to the last byte; the TCP/UDP list and the bash script diverge at character 447 and 158 ("but" against "whereas"; the names of three counters), both continuations as coherent as the other (`texts-two-ranks.json`, `texts-one-gpu.json`). That is the tensor-parallel sum order, the same kind of difference the MLX pair has against one Spark.
- The patch's unit tests, inside the image with the pack: the per-rank cuts of q, in_proj_qkv, o_proj and out_proj equal the whole layer's columns or add up to its outputs; the two ranks' expert shares add up to the whole table through one scratch over two passes; the fp16 partials take a 256-row window; the YaRN frequencies and attention factor equal transformers' (`tests/cuda/test_qwen4_exp_exl3.py`, 21 tests; `tests/test_flashnext_yarn_host.py`, 6).

### Images and video

`VISION=1` on either profile. The pack's vision tower is EXL3; `run.sh` converts it to fp16 once with TensorFold's converter (a few CPU minutes, `$TF_CACHE/<TF_SHA>/vision-f16-v3.safetensors`), rank 0 loads it (0.84 GiB plus a 4 GiB encode workspace) and sends each request's features to rank 1. The shapes probe (a drawn red circle and blue square) answers "The image contains a red circle and a blue square." with reply hash `a7588cf7f668`, the MLX pair's hash too; the tower adds 1.0 s to a 150-token prompt (`visioncheck.txt`, `visioncheck-image.txt`).

### 1M context

`PROFILE=long` serves 1,048,576 tokens on both ranks (58 GiB a rank at startup, int8 KV) with Qwen's static YaRN (factor 4 over the 262,144 trained window) read from a profile folder's `config.json`. Needles (a secret code in a generated document, the question at the end; `needle.log`):

| Prompt tokens | Depth | Found | Time to the reply | Prefill |
|---:|---:|---|---:|---:|
| 7,556 | 50% | yes | 7 s | 1,055 tok/s |
| 286,217 | 50% | yes | 320 s | 894 tok/s |
| 958,947 | 50% | yes | 1,468 s | 653 tok/s |

The 1M prompt is a 24-minute wait on this pair: the EXL3 prompt path runs at about 0.4x the MLX path's rate and the sparse attention's indexer scores every earlier position for each chunk, so the rate falls from 1,055 to 653 tok/s between 8k and 1M. Short prompts on the long profile stay coherent and drafted equals undrafted, but they are not the plain profile's replies (a static factor shifts every position; Qwen's card says so). Serve `serial` or `concurrent` for ordinary work.

### Quality

What is measured for these weights, and by whom:

- turboderp's card: KL divergence against bf16 0.0177 at 3.05 bpw (0.0684 at 2.05; the 4.05 pack's value is in the card's plot, lower). A separate held-out test found 4.05 and 3.05 equal in next-token loss within noise.
- TensorFold upstream, teacher-forced on 2,048 tokens: the 3.05 pack NLL 0.851 and top-1 77.4%, the MLX 4-bit checkpoint 1.057 and 72.6% on the same tokens.
- This recipe, the sibling recipes' T2 gates on the shipped image (concurrent profile with vision, `quality/t2.py`, four workers; `evidence/s1-exl3-tp2/quality/`), beside the MLX pair's run of the same gates on the same day:

| Gate | EXL3 4.05 bpw, this recipe | MLX 4-bit, the MLX recipe |
|---|---:|---:|
| GSM8K, 250 | **239 / 250** (95.6%) | 239 / 250 (95.6%) |
| IFEval, 120 | **103 / 120** (85.8%) | 102 / 120 (85.0%) |
| Tool calls, 60 (exact arguments) | **60 / 60** | 60 / 60 |
| Repeated 4-grams, 64 prompts | **64 / 64** (mean 0.05%) | 64 / 64 (0.04%) |
| `reasoning_effort`, 5 | **5 / 5** | 5 / 5 |
| JSON schema, 30 (serial profile) | **30 / 30** | 30 / 30 |

`score.py t2` against the MLX run, item by item: PASS; GSM8K 2 lost and 2 gained, IFEval 6 lost and 7 gained, McNemar p = 1.0 on both (`quality/score-vs-mlx.txt`). These gates are coarse; they say the two-rank EXL3 serve is not broken anywhere they look, not that it is better. The KL and NLL figures above are where the weights differ.

### Against the MLX pair and the other published recipes

Every public recipe for this model on DGX Spark that reports a number, as of 2026-10-07; their figures are their own, ours the frozen ruler on this pair. Quantizations differ, so this compares served recipes, not engines on equal weights.

| Recipe | Sparks | Weights | Engine | One user, prose tok/s | Concurrency, aggregate tok/s |
|---|---:|---|---|---:|---|
| **this recipe**, `PROFILE=serial` | 2 | EXL3 4.05 bpw | TensorFold 0.6.6 | **57** (structured 239, code 105-115) | one stream |
| **this recipe**, `PROFILE=concurrent` with vision | 2 | EXL3 4.05 bpw | TensorFold 0.6.6 | **55** (structured 234) | 4 / 8 / 16 users: 161 / 264 / 369; structured at 16: 798 |
| [MLX recipe, same pair](https://github.com/BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks) | 2 | MLX 4-bit g32 | TensorFold 0.6.6 | 67-73 (structured 242-250) | 4 / 8 / 16 users: 280-330 / 460-520 / 650-715 |
| [sfxnz base recipe](https://github.com/sfxnz/Qwen3.8-Flash-Next-TensorFold-2x-DGX-Spark) | 2 | MLX 4-bit g32 | TensorFold 0.6.2 | 69.6 (structured 249.5) | 4 / 8 users: 162-323 / 228-520 |
| [Mia's AI Lab, TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) | 1 | MLX 4-bit g32 | TensorFold 0.6.1 | 63.6 (code 96.9) | 4 users: 114-166 |
| [EXL3 4.05 bpw, native ExLlamaV3 / vllm-exl3](https://github.com/evanwtf/local-llm/issues/434) | 1 | EXL3 4.05 bpw | ExLlamaV3, vLLM | 58.8 | 4 / 8 users: 156 / 158 |
| [ykmran AutoRound int4](https://forums.developer.nvidia.com/t/which-single-spark-qwen3-8-flash-next-thread-is-the-best/382522) | 1 | int4 | vLLM | 47.5 code, 60 JSON | not reported |
| [getrefined, RadixArk NVFP4](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark) | 2 | NVFP4 | vLLM | 55.8 greedy (36-56 by content) | 4 / 8 users: 76 / 126 |
| official NVFP4, [blazux](https://github.com/blazux/qwen3.8-Flash-DGX), SGLang | 1 | NVFP4 | vLLM, SGLang | 43-44 | not reported |
| [pocharlies](https://huggingface.co/pocharlies/Qwen3.8-Flash-Next), weschera, NVFP4 | 2 | NVFP4 | SGLang | 41-42 | 8 users: 153 |
| [FP8 on two Sparks](https://forums.developer.nvidia.com/t/fp8-qwen3-8-flash-next-on-2x-dgx-spark-via-sglang-37-40-tok-s/382435) | 2 | FP8 | SGLang | 36-41 | 4 users: 88-99 |
| [EXL3 3.05 bpw, SGLang](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-SGLang-DGX-Spark-recipe) | 1 | EXL3 3.05 bpw | SGLang | 22-31 | not reported |

Where this recipe does not lead: the MLX pair is 20-25% faster at one user and about 1.8x at eight (its experts split by width across the ranks, EXL3's are held whole, and the EXL3 expert kernel decodes a tile again for every 16 rows); the MLX path prefills about 2.5x faster. Where it does: the weights, the 1M window on a pair, and the only two-rank EXL3 serve of this model.

## Requirements

- Two DGX Spark (GB10, 128 GB unified memory each) on the ConnectX-7 link, passwordless ssh from the head to the worker, Docker with the NVIDIA runtime on both, passwordless `sudo` for `drop_caches` (optional; `run.sh` says when it skips it).
- 110 GB of disk on each node for the pack, plus the image (about 25 GB) and the extension cache.
- `hf` (huggingface_hub) on the head for the download; the worker needs the same snapshot (`run.sh` downloads on each node, or copy `~/.cache/huggingface/hub/models--turboderp--Qwen3.8-Flash-Next-exl3` over).

## Quick start

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-exl3-2x-dgx-sparks
cd qwen38-flashnext-exl3-2x-dgx-sparks
$EDITOR .env.cluster        # HEAD_IP, WORKER_HOST=user@worker, PORT, HCA
VALIDATE_ONLY=1 ./run.sh    # the settings, no Docker needed
./run.sh                    # builds the image on first use, downloads the pack, starts rank 1 then rank 0
```

Then:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
    "model": "turboderp/Qwen3.8-Flash-Next-exl3",
    "messages": [{"role": "user", "content": "Explain why the sky is blue in three sentences."}],
    "max_tokens": 200, "temperature": 0}'
python3 smoke_thinking.py --model turboderp/Qwen3.8-Flash-Next-exl3 --url http://127.0.0.1:8000/v1/chat/completions
python3 smoke_tools.py    --model turboderp/Qwen3.8-Flash-Next-exl3 --url http://127.0.0.1:8000/v1/chat/completions
python3 bench_decode.py   --model turboderp/Qwen3.8-Flash-Next-exl3 --url http://127.0.0.1:8000/v1/chat/completions
./stop.sh
```

Profiles:

```bash
PROFILE=concurrent ./run.sh            # sixteen streams on two ranks
VISION=1 PROFILE=concurrent ./run.sh   # plus images and video (one-time tower conversion on first use)
PROFILE=long ./run.sh                  # 1,048,576 tokens, YaRN x4; for prompts beyond 262k
```

At boot: copy `qwen38-exl3-tensorfold.service` to `/etc/systemd/system/`, replace `USER` and the checkout path, `sudo systemctl daemon-reload && sudo systemctl enable --now qwen38-exl3-tensorfold`. The unit's `PROFILE` is `concurrent`; a drop-in overrides it.

## Defaults

`recipe.yaml` is the source of truth for the shipped defaults. Edit it, then run `python3 kit/render.py`. CI fails when this table or the `run.sh` block drifts from it. `.env.cluster` overrides them per cluster.

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Engine | TensorFold `cb2ebf0540f42604e2759b2ddef497861e928248` (v0.6.6), built into `tf-qwen38-flashnext-exl3:0.6.6` from `docker/Dockerfile` |
| Patch | `patches/flashnext-exl3-0.6.6.patch`: EXL3 packs on two ranks (whole experts a rank, head and projection cuts on Hadamard blocks), YaRN from config.json, a 16-stream EXL3 decode window, plus the MLX recipe's tool, copy-draft and two-rank vision parts (`docker/patches/README.md`) |
| Model | `turboderp/Qwen3.8-Flash-Next-exl3` branch `4.05bpw_h6_ng6` at `55a732e0c4c3d4614bc42b68493bb930d9b02c0a` (108 GB; both nodes) |
| Profiles | `PROFILE=serial` (default, one stream on CUDA graphs), `concurrent` (`PARALLEL=16`), `long` (`CONTEXT=1048576` through a YaRN profile folder; short prompts belong on the other two) |
| Images | `VISION=0`: off. `VISION=1` converts the pack's EXL3 vision tower to fp16 once (`$TF_CACHE/<TF_SHA>/vision-f16-v3.safetensors`, a few CPU minutes) and serves `image_url` and `video_url` parts on either profile; rank 0 runs the tower, rank 1 receives each request's features. `VISION_URLS=1` also fetches public https URLs; `VISION_MAX_IMAGES` caps images per request |
| Tool system message | `Tool calls: include only the arguments the user explicitly provided or clearly implied. Never fill in optional arguments with default or guessed values.` (added by the server to a chat request that offers tools and has no system message; `TOOL_SYSTEM=` serves none) |
| Ranks | `--tp 2`: rank 1 on `spark2` first, then rank 0 (HTTP) on the head; rendezvous `10.100.8.1:29551`. Each rank holds 256 of the 512 experts, half the attention and DeltaNet heads and half the vocabulary: about 41 GiB |
| `--context` | 262144 (the trained window, on both ranks); `PROFILE=long` 1048576 |
| `--kv-dtype` | `int8` |
| MTP drafts | `--mtp-drafts 15 --mtp-confidence 0.70`, plus copy drafts |
| `--parallel` | 1 by default; `PROFILE=concurrent` (or `PARALLEL=16`) serves sixteen streams on two ranks |
| Default thinking | off (`THINKING=0`); requests override with `chat_template_kwargs.enable_thinking` |
| `--max-tokens` | 32768 (the reply cap when a request sets none) |
| NCCL | `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1` |
| API | `http://<head>:8000/v1`, served as `turboderp/Qwen3.8-Flash-Next-exl3` |
| Container | `tf-qwen38-exl3` |
<!-- END generated defaults -->

`run.sh` refuses, before anything starts:

- non-decimal or zero-padded integers, `MTP_CONFIDENCE` outside [0, 1], a `TF_SHA` / `SNAPSHOT_SHA` that is not 40 hex
- `TP` other than 1 or 2, `CONTEXT` above the trained 262144 on the plain profiles or above 1048576 on `long` (and at or below 262144 on `long`: that window belongs on the plain rotary), `MTP_DRAFTS` above the engine cap 15, `KV_DTYPE` other than bf16 / int8 / int4
- a patch file that differs from its pin, an image whose labels do not match `TF_SHA` / the patch sha, an incomplete snapshot when `SKIP_DOWNLOAD=1`
- `EXTRA_ARGS` that re-sets a flag `run.sh` builds, and `--prefill-fp8` / `--drafter` / `--ple-on-ssd` / `--ssd-experts` (refused or unused for an EXL3 pack)

At start it also refuses another running GPU container on either node and a busy port, drops caches and compacts memory on each node (NCCL's first memory registration failed after large downloads without it), and copies the head's image to the worker when the worker's differs. `BENCH_ONLY=1` binds the API to 127.0.0.1. `MTP_DRAFTS=0` serves without drafts.

## Not supported (TensorFold 0.6.6 with the patch, two ranks)

- Grammars (`response_format`, `guided_*`) and logprobs with `--parallel` on two ranks; the serial default serves `response_format`.
- Prefix reuse for image prompts; grammars and logprobs with images.
- Other EXL3 branches of the pack are not pinned here. The loader reads any width (2.05 to 6.05 bpw packs exist; 6.05 is 139 GB and would fit two ranks at about 53 GiB each); a different branch needs its own `SNAPSHOT_SHA` and its own numbers.
- The Metal engine does not read the long profile's YaRN fields.

## Environment

Everything `run.sh` reads is in the Defaults table; `.env.cluster` next to it sets a cluster's values (`KEY=VALUE` lines; the environment wins). `EXTRA_ENV` forwards `KEY=VALUE` words to both containers; `EXTRA_ARGS` appends engine flags `run.sh` does not own.

## Logs

`docker logs -f tf-qwen38-exl3` on either node. The startup lines name the profile's window, the rotary (YaRN on `long`), the KV dtype, the draft rule and, with `VISION=1`, the tower. Each finished request logs its prompt and reply tokens, the reply's sha, tok/s and the drafts accepted.

## Evidence

[`evidence/s1-exl3-tp2/`](evidence/s1-exl3-tp2/): the rulers, the six-prompt comparisons (texts included), the two-ranks-against-one-GPU receipt, the vision probe, the needles. Every number above has a file there.

## Gotchas

- After a large download or image load, a node's first NCCL start can fail with `ibv_reg_mr: Cannot allocate memory`. `run.sh` drops caches and compacts memory before every start, which clears it; a container started by hand needs `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory` first.
- A 1M prompt is a long wait: the EXL3 prompt path runs at about 0.4x the MLX path's rate and the sparse attention's indexer scores every earlier position for each chunk. The needle table has the rates.
- One user pays the expert imbalance of whole-expert sharing (five of ten experts a rank on average, up to ten). Sixteen users do not.
- The long profile shifts short prompts. Keep it for long ones.

## Credits

- Weights: [turboderp](https://huggingface.co/turboderp) (the EXL3 pack and the format, [ExLlamaV3](https://github.com/turboderp-org/exllamav3)); base model Qwen3.8-Flash-Next by the Qwen team.
- Engine: [TensorFold](https://github.com/ashhart/TensorFold) by ashhart; its EXL3 reader, grouped expert kernel and two-rank engine are what the patch builds on. The vision port descends from MiaAI-Lab's patch 0008 (`LICENSES/`).
- Recipe lineage: the MLX recipe (BobClawblaw), sfxnz's base recipe and the vLLM sibling, whose `run.sh` guards, frozen ruler and harness this repo keeps.

## License

Apache-2.0 for this repository's files, as the base recipes. TensorFold is Apache-2.0; the pack carries the Qwen Community 1.0 license; see `LICENSE`, `NOTICE` and `LICENSES/`.
