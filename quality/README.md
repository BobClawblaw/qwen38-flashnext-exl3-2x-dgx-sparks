# Quality gates

Host-side scripts, Python stdlib (plus numpy for `score.py`). They talk to a running serve over its OpenAI API and never start, stop or touch a container. Each one writes to `--out DIR`, including a `manifest.json` with the server's `/v1/models`, the git rev and dirty flag, argv, and the sha256 of every quality script and input file. Keep runs under `~/projects/data/qwen38-evals/runs/` and copy only the small JSON verdicts into `evidence/`.

These files come from the vLLM sibling recipe (`sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark`, `quality/`). Changes here are limited to `qlib.py`:
- `DEFAULT_MODEL` is this recipe's served name.
- `running_requests()` also reads `tensorfold:num_requests_running`.
- `tokenize_count()` uses the checkpoint's `tokenizer.json` when `QLIB_TOKENIZER` names it, because TensorFold has no `/tokenize` endpoint.

| Gate | Command | Runs on TensorFold |
|---|---|---|
| T2 | `python3 quality/t2.py --out D`, then `python3 quality/score.py t2 REF D` | Yes. GSM8K-250, IFEval-120, tools 60 (exact arguments), JSON schema 30 (needs `xgrammar`, which the image has), repeat-4gram, `reasoning_effort` |
| T3 | `tools/run_t3.sh D [--lengths 250000]` (runs `t3_needles.py` inside the recipe image with `QLIB_TOKENIZER`) | Yes. 4k / 16k / 64k / 128k × 3 depths × 2 needles; `--lengths 250000` for the near-full window |
| T1 (KL / top-1 against a reference) | not shipped | No: needs `prompt_logprobs`, and TensorFold has no logprobs at two ranks |

`score.py t2` compares item by item (McNemar). Across different checkpoints (this recipe's MLX 4-bit against the sibling's NVFP4) the comparison is a report, not a gate: `score.py` exits 1 when a rule like "tools 60/60" fails against the reference.

Datasets are read from `$QWEN38_EVALS_DIR/datasets` (default `~/projects/data/qwen38-evals/datasets`): `gsm8k/test.jsonl` (openai/grade-school-math @3101c7d) and `ifeval/input_data.jsonl` (google-research @26d8ccd), both sha256-checked. `--limit N` gives a smoke run.
