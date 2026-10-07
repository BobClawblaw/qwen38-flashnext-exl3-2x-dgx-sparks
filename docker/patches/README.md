# docker/patches/

Patches the image build applies on top of TensorFold at `TF_SHA` (`docker build --build-arg TF_PATCH=patches/<file> --build-arg TF_PATCH_SHA=<sha256>`). Each shipped patch's sha256 is pinned in `run.sh` (`PATCH_PINS`) and `recipe.yaml` (`engine.patches`), and `tests/` checks the files against the pin. `run.sh` refuses a patch file that differs from its pin. The build checks the copied file against the sha it is given, and `run.sh` refuses an image whose `tensorfold.patch_sha` label differs. A regenerated patch therefore needs a new pin, and new evidence for the numbers it affects.

## flashnext-exl3-0.6.6.patch

The default image carries it. Eight changes to TensorFold v0.6.6 (`cb2ebf0540f42604e2759b2ddef497861e928248`), Apache-2.0 as TensorFold. Parts 1 to 5 are the MLX recipe's patch, byte for byte (`BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks`, `docker/patches/flashnext-tools-0.6.6.patch`, pin `023db641`); parts 3 to 5 are in TensorFold main since 2026-10-07 (#468, #473 and upstream's follow-ups). Parts 6 to 8 are this recipe's, and are what makes an EXL3 pack serve on two ranks, at 1M tokens, to sixteen users. The Python engine is frozen upstream at 0.6.6, so these live here while the image builds from v0.6.6.

### 1 to 5. Tools, copy drafts, image and video on two ranks

`--tool-system TEXT` on the CUDA server; a tool call keeps only the parameters the offered tool declares; copy drafts (a reply that repeats earlier text drafts its continuation whole, byte-identical replies, `TENSORFOLD_COPY_DRAFTS=0` turns it off); image and video input on two ranks for the concurrent decoder and the serial engine (rank 0 runs the tower, rank 1 receives each request's features inside the admission lockstep). The MLX recipe's `docker/patches/README.md` documents each with its receipts; nothing in them is format-specific, so an EXL3 pack gets all of it.

### 6. EXL3 packs on two ranks

Files: `families/qwen4_exp/cuda/exl3.py` (the per-rank loader), `exl3_mm.py` (`x3_part`: a matrix cut to output rows or input columns), `forward.py` (`_out_proj` gathers non-MLX partials too; `_exl3_moe_partial`), `engine.py` (the gate), `cuda/exl3/experts.py` (`remap`, `share`), `cuda/geometry.py` (the admission estimate reads EXL3 tensor names), `tests/cuda/test_qwen4_exp_exl3.py` (`test_two_rank_cuts_equal_the_whole_layer`, `test_two_rank_expert_shares_add_up_to_the_layer`).

- **Why.** TensorFold 0.6.6 serves EXL3 packs of Flash Next on one GPU and refuses `--tp 2`; two ranks read the MLX 4-bit checkpoint only. The MLX checkpoint is uniform affine 4-bit on every linear, and the quality complaints about it are documented (TensorFold issue #85: 10 to 12 points against a mixed 4/8-bit pack; +20.6% perplexity against +1.3%). turboderp's EXL3 packs are the best-measured 4-bit weights for this model (KL 0.0177 at 3.05 bpw), and two ranks are what give the full window and concurrency on a pair.
- **Dense projections.** An EXL3 layer is `y = ((x * suh) @ H_K @ W_q) @ H_N * svh + bias` with `H` a 128-wide block Hadamard, so a cut on whole 16x16 tiles and whole 128-blocks is exact: by outputs with columns of tiles and of `svh` (all of `suh`), by inputs with rows of tiles and of `suh` (all of `svh`), the ranks' partial outputs added. Each rank takes its heads' rows of q (with its gates), k and v, the DeltaNet q, k, v and z projections, and its half of the head's vocabulary; o_proj and out_proj are cut by inputs and their fp32 partials gathered, as the MLX loader does. Flash Next's dimensions all land on 128: 12 of 24 heads x 256, 1 of 2 KV heads, 24 of 48 DeltaNet value heads x 128, 8 of 16 key heads, 124,160 of 248,320 vocabulary rows.
- **Experts.** A routed expert is 640 wide, five Hadamard blocks, which two ranks cannot halve. So each rank holds whole experts: 256 of the 512, and, layer by layer in turn, the shared expert. Both ranks route every token; a rank's picks go through the layer's `remap` (an expert held elsewhere lands past the local table, which every grouped kernel skips) and that slot's weight is zeroed (`experts.share`), because the fused down-and-combine kernel takes a skipped slot's output from its scratch as the caller left it, and the engine keeps one scratch for every layer. The ranks' weighted partial sums are gathered and added like the MLX path's. Per token a rank runs the experts it holds: 5 of 10 on average, up to all 10; at 16 users the imbalance averages out. The draft head's rows come out of the whole head, read once at load and dropped.
- **Memory.** 41.4 GiB a rank for the 4.05 bpw pack (the 33 GB n-gram table stays memory-mapped and locked on each node), so the full 262,144 window fits on both ranks with room, and the 1M profile fits too (58 GiB a rank).
- **Exactness, measured on the pair.** Drafted replies equal `"draft": false` ones, 6 of 6 prompts on the serial profile and 3 of 3 on the 1M profile; against the same pack on one GPU, 4 of 6 greedy replies are identical to the last byte and the other two diverge at a near-tie 158 and 447 characters in (the tensor-parallel sum order), the two-rank reply as coherent as the one-GPU one. The unit tests check the cuts against the whole layer and the expert shares against the whole table, with one scratch across tables and two passes. `evidence/s1-exl3-tp2/`.
- **What it is not.** Not expert width splitting (the kernel takes one width a layer); not a change to the one-GPU path, whose tests pass unchanged; NVFP4 checkpoints still run on one GPU.

### 7. YaRN from config.json (1M context)

Files: `families/qwen4_exp/cuda/weight_types.py` (`Config.rope_scaling`, `rotary_inv_freq`, `scale_rotary`), `weights.py` and `exl3.py` (both loaders use them), `engine.py` (a startup line), `tests/test_flashnext_yarn_host.py`.

- **What.** `config.json`'s `rope_parameters` with `rope_type: "yarn"`, `factor` and `original_max_position_embeddings` (optional `beta_fast`, `beta_slow`, `attention_factor`, `mscale`, `mscale_all_dim`, `truncate`) switches the rotary to YaRN, exactly as Qwen's card describes for 1M and as transformers computes it (`_compute_yarn_parameters`; the host test checks the frequencies and the attention factor against it). The window comes from `max_position_embeddings` as before. The kernels build cos and sin from the frequencies and the position, so the attention factor is folded into the rotated dims of the q, k and indexer norm scales instead: rotation is linear, so scaling the rotated part before rotation equals scaling cos and sin.
- **How the recipe uses it.** `PROFILE=long` builds a folder of links to the pack's files with its own `config.json` (`run.sh ensure_long_profile`); the pack's bytes are untouched. A static factor shifts short prompts too (Qwen's card says so), so the long profile is a profile, not the default.
- **Scope.** CUDA only; the Metal engine ignores the fields.

### 8. A sixteen-stream EXL3 decode window

Files: `families/qwen4_exp/cuda/exl3_mm.py` (`Scratch.allocate` sizes the fp16 partials for the prompt rows; `F16.partials` takes a window up to that), `tests/cuda/test_qwen4_exp_exl3.py` (`test_f16_partials_take_a_full_concurrent_window`).

The hyper-connections' fp16 down projection returned fp32 K-slices for at most 128 rows; sixteen streams x up to 16 verify rows a round is 256, and the engine stopped at the first such round (`F16.partials takes at most 128 rows`) on one GPU and on two. The scratch now spans the prompt rows (2,048) and the check is against it. Row-invariant as before: a row's slices do not depend on the window.

### Validation

Inside the image, with the pack: `tests/cuda/test_qwen4_exp_exl3.py` (21 tests: upstream's 19, the one-GPU EXL3 path unchanged, plus the two-rank cuts and shares and the 256-row window) and `tests/test_flashnext_yarn_host.py` (6). On the pair: `evidence/s1-exl3-tp2/` (drafted against undrafted, two ranks against one GPU, the vision probe, the rulers, the needles).
