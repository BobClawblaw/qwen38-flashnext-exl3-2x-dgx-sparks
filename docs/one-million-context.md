# The 1M profile

`PROFILE=long` serves 1,048,576 tokens on both ranks. It is a separate profile because of how it works.

## What changes

Qwen3.8 Flash Next is trained to 262,144 tokens. Qwen's model card reaches 1M with static YaRN at factor 4: the low-frequency rotary pairs are interpolated by 4, the high-frequency pairs kept, a ramp between, and the attention logits scaled by `0.1 ln 4 + 1 = 1.139`. TensorFold 0.6.6 has no rope scaling; the patch's part 7 reads it from `config.json` (`rope_parameters` with `rope_type: "yarn"`, `factor`, `original_max_position_embeddings`), computes the frequencies as transformers does, and folds the attention factor into the rotated dims of the norm scales, since the kernels build cos and sin from the frequencies and the position.

`run.sh` does not edit the pack. It builds `$TF_CACHE/<TF_SHA>/long-1048576/`: links to every file of the pinned snapshot (container paths) plus a `config.json` with the YaRN block and `max_position_embeddings` at 1,048,576, and serves that folder. The folder is rebuilt when the pinned snapshot changes.

## Why a profile

A static factor applies to every position. Qwen's card: "Static YaRN implementation means the scaling factor remains constant regardless of input length, potentially impacting performance on shorter texts." On this pair the short-prompt replies on the long profile are coherent and drafted equals undrafted, but they are not the plain profile's replies (different greedy text on the same prompts). Serve `serial` or `concurrent` for ordinary work and `long` when prompts exceed 262k.

## Cost

Memory is not the constraint: 58 GiB a rank at startup with the 1M window admitted on both ranks (int8 KV; 12 full-attention layers x 1 KV head a rank x 256 x 1M x 2 bytes is 6 GiB, the DeltaNet state is constant). Time is: the EXL3 prompt path decodes its tiles again for every 16 rows and runs at about 0.4x the MLX path's rate, and the sparse-attention indexer scores every earlier position for each new chunk, so a prompt's prefill slows as it grows. The needle runs in `evidence/s1-exl3-tp2/needle.log` give the rates at each length.

## Checks

- `tests/test_flashnext_yarn_host.py`: the frequencies and the attention factor against transformers' `_compute_yarn_parameters`; refusals of incomplete parameters; the fold touches the rotated dims only, main layers and the MTP layer.
- Needles on the pair: a 7.5k-token document (control), 300k and 1M, the secret at mid-depth and at 10%.
