#!/usr/bin/env bash
# Qwen3.8-Flash-Next, turboderp's EXL3 pack (4.05 bpw), on 2x DGX Spark (GB10) with TensorFold at TP=2.
# Guards and orchestration follow sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark run.sh @ d66a95e and the
# MLX recipe (BobClawblaw/qwen38-flashnext-tensorfold-2x-dgx-sparks). The head starts rank 1 on WORKER_HOST
# over ssh, then rank 0, which serves HTTP.
set -euo pipefail

# PROFILE=serial (one stream, graphs), concurrent (sixteen streams) or long (the 1M YaRN profile, README
# "1M context"). A profile only fills the variables below when they are unset, so an explicit PARALLEL or
# CONTEXT still wins.
# Cluster settings live in .env.cluster next to this script (KEY=VALUE lines; a variable already in the
# environment wins, so systemd's EnvironmentFile and a one-off `PORT=8001 ./run.sh` both override it).
# ENV_CLUSTER names another file; ENV_CLUSTER=/dev/null reads none (tests/ and CI use that).
_env_cluster="${ENV_CLUSTER:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env.cluster}"
if [[ -f "$_env_cluster" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    [[ -z "${!BASH_REMATCH[1]+x}" ]] && declare "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
  done <"$_env_cluster"
fi
unset _env_cluster _line

PROFILE="${PROFILE:-serial}"
case "$PROFILE" in
  serial) ;;
  concurrent)
    # Sixteen streams on two ranks (measured: 1 / 4 / 8 / 16 users, README "Measured"). The patch lets the
    # EXL3 path take a 16 x 16 decode window (part 8).
    PARALLEL="${PARALLEL:-16}"
    ;;
  long)
    # 1,048,576 tokens on both ranks: the pack served through a profile folder whose config.json carries
    # Qwen's static YaRN (factor 4 over the 262,144 trained window; patch part 7). Short prompts see the scaled
    # rotary too, so this is a profile beside the plain one, not the default.
    CONTEXT="${CONTEXT:-1048576}"
    ;;
  *) echo "PROFILE=$PROFILE must be serial, concurrent or long." >&2; exit 1 ;;
esac

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-turboderp/Qwen3.8-Flash-Next-exl3}"
SERVED_NAME="${SERVED_NAME:-turboderp/Qwen3.8-Flash-Next-exl3}"
IMAGE="${IMAGE:-tf-qwen38-flashnext-exl3:0.6.6}"
TF_SHA="${TF_SHA:-cb2ebf0540f42604e2759b2ddef497861e928248}"
TF_PATCH="${TF_PATCH:-patches/flashnext-exl3-0.6.6.patch}"
CONTAINER_NAME="${CONTAINER_NAME:-tf-qwen38-exl3}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29551}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1,roceP2p1s0f1}"
TP="${TP:-2}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
MTP_DRAFTS="${MTP_DRAFTS:-15}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.70}"
PARALLEL="${PARALLEL:-1}"
THINKING="${THINKING:-0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TOOL_SYSTEM="${TOOL_SYSTEM:-Tool calls: include only the arguments the user explicitly provided or clearly implied. Never fill in optional arguments with default or guessed values.}"
VISION="${VISION:-0}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-}"
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_HOME_IN_CONTAINER="/cache/huggingface"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-55a732e0c4c3d4614bc42b68493bb930d9b02c0a}"
SNAPSHOT="${HF_CACHE}/hub/models--turboderp--Qwen3.8-Flash-Next-exl3/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--turboderp--Qwen3.8-Flash-Next-exl3/snapshots/${SNAPSHOT_SHA}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
TF_CACHE="${TF_CACHE:-$HOME/.cache/tensorfold-qwen38-exl3}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-1000}"
MEMGUARD="${MEMGUARD:-1}"
MEMGUARD_MIN_AVAIL_MB="${MEMGUARD_MIN_AVAIL_MB:-3072}"
MEMGUARD_MIN_SWAP_FREE_MB="${MEMGUARD_MIN_SWAP_FREE_MB:-2048}"
BENCH_ONLY="${BENCH_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
# END generated
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.run-state"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# The trained window (config.json max_position_embeddings); TensorFold refuses more. PROFILE=long serves
# YARN_FACTOR times it through the profile folder's config.json (patch part 7).
NATIVE_CONTEXT=262144
YARN_FACTOR=4
LONG_CONTEXT=$(( NATIVE_CONTEXT * YARN_FACTOR ))
# The engine's draft cap (families/qwen4_exp/cuda/engine.py MAX_DEPTH).
MAX_MTP_DRAFTS=15
# sha256 of every shipped patch, the bytes the published numbers were measured with (recipe.yaml
# engine.patches; tests/ checks the files). A regenerated patch needs a new pin and new evidence.
declare -A PATCH_PINS=(
  [patches/flashnext-exl3-0.6.6.patch]=6868816a004c5acb8199648a97af90c5d189c0b316914f9c49fda4937bce91d1
)

die() {
  echo "$*" >&2
  exit 1
}

# --- integers and flags. Leading zeros are refused: bash reads 010 as octal 8.
for name in PORT MASTER_PORT TP CONTEXT PARALLEL MAX_TOKENS MEMGUARD_MIN_AVAIL_MB MEMGUARD_MIN_SWAP_FREE_MB; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name=${!name} is not a positive decimal integer."
done
[[ "$MTP_DRAFTS" =~ ^(0|[1-9][0-9]*)$ ]] || die "MTP_DRAFTS=$MTP_DRAFTS is not a non-negative decimal integer."
for name in THINKING SKIP_DOWNLOAD HF_HUB_DISABLE_XET MEMGUARD BENCH_ONLY VISION VISION_URLS; do
  [[ "${!name}" =~ ^[01]$ ]] || die "$name=${!name} must be 0 or 1."
done
[[ -z "$VISION_MAX_IMAGES" || "$VISION_MAX_IMAGES" =~ ^[1-9][0-9]*$ ]] || die "VISION_MAX_IMAGES=$VISION_MAX_IMAGES is not empty or a positive decimal integer."
# Images on either profile (patch parts 4 and 5): rank 0 encodes, rank 1 receives the features. On the serial
# profile an image request decodes eagerly (the captured graphs are text-only); text requests keep their graphs.
[[ "$OOM_SCORE_ADJ" =~ ^(-?[1-9][0-9]*|0)$ ]] && (( OOM_SCORE_ADJ >= -1000 && OOM_SCORE_ADJ <= 1000 )) || die "OOM_SCORE_ADJ=$OOM_SCORE_ADJ must be an integer in [-1000, 1000]."
[[ "$MTP_CONFIDENCE" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || die "MTP_CONFIDENCE=$MTP_CONFIDENCE must be a decimal in [0, 1], e.g. 0.70."
[[ -z "$MEMORY_RESERVE_GIB" || "$MEMORY_RESERVE_GIB" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]] || die "MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB is not empty or a positive decimal."
[[ "$TF_SHA" =~ ^[0-9a-f]{40}$ ]] || die "TF_SHA=$TF_SHA is not a 40-hex TensorFold commit."
[[ "$SNAPSHOT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "SNAPSHOT_SHA=$SNAPSHOT_SHA is not a 40-hex snapshot revision."
[[ "$TF_CACHE" == /* ]] || die "TF_CACHE=$TF_CACHE must be an absolute path."
# The head hashes the patch file; the worker's /tmp copy of this script has no docker/ dir and gets the
# head's hash forwarded instead (its image must carry the same patch_sha label either way).
if [[ "$TF_PATCH" == none ]]; then
  TF_PATCH_SHA=none
else
  [[ "$TF_PATCH" =~ ^patches/[A-Za-z0-9._-]+\.patch$ ]] || die "TF_PATCH=$TF_PATCH must be none or a file under docker/patches/."
  if [[ -f "$SCRIPT_DIR/docker/$TF_PATCH" ]]; then
    TF_PATCH_SHA="$(sha256sum "$SCRIPT_DIR/docker/$TF_PATCH" | cut -d' ' -f1)"
  elif [[ "${ROLE:-}" == worker && "${TF_PATCH_SHA:-}" =~ ^[0-9a-f]{64}$ ]]; then
    :
  else
    die "TF_PATCH=$TF_PATCH must be none or a file under docker/patches/."
  fi
  [[ -n "${PATCH_PINS[$TF_PATCH]:-}" ]] || die "TF_PATCH=$TF_PATCH has no pin in run.sh PATCH_PINS (docker/patches/README.md)."
  [[ "$TF_PATCH_SHA" == "${PATCH_PINS[$TF_PATCH]}" ]] || die "TF_PATCH=$TF_PATCH has sha256 $TF_PATCH_SHA, not the pinned ${PATCH_PINS[$TF_PATCH]}. Never edit a patch by hand (docker/patches/README.md)."
fi

# --- topology. TensorFold runs one rank per machine, at most two (cli_args.py --tp choices 1/2).
case "$TP" in
  1 | 2) ;;
  *) die "TP=$TP: TensorFold serves Flash Next on one or two ranks only." ;;
esac
case "$KV_DTYPE" in
  bf16 | int8 | int4) ;;
  *) die "KV_DTYPE=$KV_DTYPE must be bf16, int8 or int4." ;;
esac
if [[ "$PROFILE" == long ]]; then
  (( CONTEXT <= LONG_CONTEXT )) || die "CONTEXT=$CONTEXT exceeds the long profile's window $LONG_CONTEXT (YaRN x$YARN_FACTOR over $NATIVE_CONTEXT)."
  (( CONTEXT > NATIVE_CONTEXT )) || die "CONTEXT=$CONTEXT fits the trained window $NATIVE_CONTEXT: serve PROFILE=serial or concurrent, which keep the plain rotary."
else
  (( CONTEXT <= NATIVE_CONTEXT )) || die "CONTEXT=$CONTEXT exceeds the trained window $NATIVE_CONTEXT; PROFILE=long serves up to $LONG_CONTEXT with YaRN."
fi
(( MTP_DRAFTS <= MAX_MTP_DRAFTS )) || die "MTP_DRAFTS=$MTP_DRAFTS exceeds the engine cap $MAX_MTP_DRAFTS."
# --parallel N at --tp 2: TensorFold 0.6.4+ serves it (release notes, #141). Grammars (response_format,
# guided_*) are still refused at two ranks with --parallel (families/qwen4_exp/cuda/engine.py).

# --- EXTRA_ENV: KEY=VALUE words, no spaces inside a value.
EXTRA_ENV_ARGS=()
for kv in $EXTRA_ENV; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "EXTRA_ENV entry '$kv' is not KEY=VALUE."
  EXTRA_ENV_ARGS+=(-e "$kv")
done

# --- EXTRA_ARGS must not re-set a flag run.sh builds; argparse keeps the last value, so a duplicate
# would bypass the guard on its variable or desynchronise the two ranks. TensorFold's parser expands
# unambiguous prefixes (--paral means --parallel), so any prefix of a guarded flag is refused too.
OWNED_FLAGS="--tp --rank --master --master-port --host --port --name --context --kv-dtype --mtp-drafts --mtp-confidence --parallel --thinking --no-thinking --max-tokens --no-update-check --no-drafts --tool-system --vision --vision-urls --vision-max-images"
UNUSED_FLAGS="--prefill-fp8 --drafter --ple-on-ssd --ssd-experts --vision-offload --vision-image-tokens"
for w in $EXTRA_ARGS; do
  f="${w%%=*}"
  [[ "$f" == --?* ]] || continue
  for g in $OWNED_FLAGS; do
    [[ "$g" == "$f"* ]] && die "EXTRA_ARGS sets $w (argparse reads it as $g), which run.sh passes itself. Use TP, PORT, MASTER_PORT, SERVED_NAME, CONTEXT, KV_DTYPE, MTP_DRAFTS (0 = no drafts), MTP_CONFIDENCE, PARALLEL, THINKING, MAX_TOKENS, VISION, VISION_URLS or VISION_MAX_IMAGES instead."
  done
  for g in $UNUSED_FLAGS; do
    [[ "$g" == "$f"* ]] && die "EXTRA_ARGS sets $w ($g): refused or unused for an EXL3 pack on CUDA (README 'Not supported')."
  done
done

API_HOST=0.0.0.0
[[ "$BENCH_ONLY" == 1 ]] && API_HOST=127.0.0.1

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only profile=%s image=%s tf=%s patch=%s snapshot=%s tp=%s ctx=%s kv=%s mtp=%s@%s parallel=%s thinking=%s max_tokens=%s vision=%s hca=%s host=%s\n' \
    "$PROFILE" "$IMAGE" "$TF_SHA" "$TF_PATCH" "$SNAPSHOT_SHA" "$TP" "$CONTEXT" "$KV_DTYPE" "$MTP_DRAFTS" "$MTP_CONFIDENCE" "$PARALLEL" "$THINKING" "$MAX_TOKENS" "$VISION" "$HCA" "$API_HOST"
  exit 0
fi

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    # Compact too: after a large image load the worker refused NCCL's first memory registration
    # (ibv_reg_mr: Cannot allocate memory) on seven starts in a row and came up right after one
    # compaction; the NIC driver wants higher-order pages for a region's translation tables.
    echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
    log "drop_caches: ran (and compact_memory)"
  else
    log "drop_caches: skipped (no passwordless sudo)"
  fi
}

image_sha() {
  # "<tensorfold sha> <patch sha>" from the image labels, empty when the image is missing.
  docker image inspect -f '{{ index .Config.Labels "tensorfold.sha" }} {{ index .Config.Labels "tensorfold.patch_sha" }}' "$IMAGE" 2>/dev/null || true
}

ensure_image() {
  local have
  have="$(image_sha)"
  if [[ -z "$have" ]]; then
    [[ -f "$SCRIPT_DIR/docker/Dockerfile" ]] || die "Image $IMAGE is missing and $SCRIPT_DIR/docker/Dockerfile is not here. Build it on this node (README 'Image')."
    log "Building $IMAGE (TensorFold $TF_SHA, patch $TF_PATCH) from docker/Dockerfile"
    docker build --build-arg "TF_SHA=$TF_SHA" --build-arg "TF_PATCH=$TF_PATCH" --build-arg "TF_PATCH_SHA=$TF_PATCH_SHA" \
      -t "$IMAGE" "$SCRIPT_DIR/docker"
    have="$(image_sha)"
  fi
  [[ "$have" == "$TF_SHA $TF_PATCH_SHA" ]] || die "Image $IMAGE carries TensorFold/patch '${have:-unknown}', not '$TF_SHA $TF_PATCH_SHA' (TF_SHA, sha256 of TF_PATCH). Rebuild it or set IMAGE to the matching tag."
  log "Image $IMAGE (TensorFold $TF_SHA, patch $TF_PATCH sha256 $TF_PATCH_SHA)"
}

snapshot_complete() {
  # Every shard in the index, config.json and the tokenizer exist, and each shard's safetensors
  # header ends exactly at its file size (a truncated download fails here, not mid-load).
  python3 - "$SNAPSHOT" <<'PY'
import json
import struct
import sys
from pathlib import Path

snap = Path(sys.argv[1])
bad = []
for name in ("config.json", "quantization_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
             "generation_config.json", "ngram_embedding.safetensors", "vision_k6.safetensors"):
    p = snap / name
    if not p.is_file() or p.stat().st_size == 0:
        bad.append(name)
index = snap / "model.safetensors.index.json"
try:
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"snapshot {snap} incomplete: {index.name} ({exc})")
for shard in shards:
    p = snap / shard
    try:
        with p.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if 8 + n + end != p.stat().st_size:
            bad.append(f"{shard} (truncated)")
    except (OSError, ValueError, struct.error) as exc:
        bad.append(f"{shard} ({exc.__class__.__name__})")
if bad:
    more = " ..." if len(bad) > 8 else ""
    sys.exit(f"snapshot {snap} incomplete: {', '.join(bad[:8])}{more}")
PY
}

ensure_weights() {
  if snapshot_complete 2>/dev/null; then
    log "Using pinned snapshot $SNAPSHOT (index shards, headers and tokenizer checked)"
    return
  fi
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    snapshot_complete || true
    die "SKIP_DOWNLOAD=1 and the pinned snapshot is incomplete."
  fi
  local HF=""
  HF="$(hf_bin || true)"
  [[ -n "$HF" ]] || die "No hf CLI on PATH and snapshot $SNAPSHOT is incomplete."
  export HF_HUB_DISABLE_XET
  log "Downloading $MODEL revision $SNAPSHOT_SHA (the 4.05bpw_h6_ng6 branch; resumes under $HF_CACHE; about 108 GB)"
  "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA" --cache-dir "$HF_CACHE/hub"
  snapshot_complete || die "Snapshot still incomplete after download."
}

refuse_foreign_serve() {
  local name devices
  while IFS= read -r name; do
    [[ -z "$name" || "$name" == "$CONTAINER_NAME" || "$name" == buildx_buildkit_dgx-bundle-builder0 ]] && continue
    devices="$(docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}} {{json .Config.Env}}' "$name" 2>/dev/null || true)"
    if printf '%s' "$devices" | grep -Eqi 'gpu|nvidia|infiniband'; then
      die "$name is using GPUs or InfiniBand. This recipe needs exclusive GPUs on both Sparks. Stop that serve first (its own stop script). Do not docker rm it from this script."
    fi
  done < <(docker ps --format '{{.Names}}')
}

refuse_busy_port() {
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    die "Port $PORT is already in use"
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Stopping existing container $CONTAINER_NAME"
    docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER_NAME" >/dev/null
    # The pages a 40 GiB rank frees take a while to come back; a start right behind them failed NCCL's first
    # memory registration on the worker (ibv_reg_mr_iova2: Cannot allocate memory) where a start 40 s later
    # came up first time. SETTLE_SECONDS=0 skips the wait.
    log "Waiting ${SETTLE_SECONDS:-30}s for the stopped rank's memory to settle"
    sleep "${SETTLE_SECONDS:-30}"
  fi
}

# Host memory watchdog. GB10 is unified memory: when the serve plus host load exhausts RAM and swap,
# the host thrashes for minutes before the kernel OOM killer acts, and it then picks small services
# first. Kill the serve instead once available RAM and free swap are both low for 3 samples (6 s).
# Exits when the container is gone.
memguard_loop() {
  local name="$1" min_avail_kb=$(( $2 * 1024 )) min_swap_kb=$(( $3 * 1024 )) hits=0 n=0 avail swapfree
  while :; do
    avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swapfree="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    if (( avail < min_avail_kb && swapfree < min_swap_kb )); then
      hits=$(( hits + 1 ))
    else
      hits=0
    fi
    if (( hits >= 3 )); then
      logger -t tf-qwen38-exl3-memguard "MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      echo "$(date -Is) MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
      docker kill "$name" >/dev/null 2>&1 || true
      return 0
    fi
    n=$(( n + 1 ))
    if (( n % 8 == 0 )) && [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]]; then
      return 0
    fi
    sleep 2
  done
}

start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || { log "memguard off (MEMGUARD=$MEMGUARD)"; return 0; }
  local logf="$STATE_DIR/memguard.log"
  mkdir -p "$STATE_DIR"
  setsid bash -c "$(declare -f memguard_loop); memguard_loop \"\$@\"" memguard \
    "$CONTAINER_NAME" "$MEMGUARD_MIN_AVAIL_MB" "$MEMGUARD_MIN_SWAP_FREE_MB" >>"$logf" 2>&1 </dev/null &
  disown || true
  log "memguard pid=$! min_avail=${MEMGUARD_MIN_AVAIL_MB}MB min_swap_free=${MEMGUARD_MIN_SWAP_FREE_MB}MB log=$logf"
}

serve_args() {
  # The tensorfold serve argv after the model directory, for one rank.
  local rank="$1"
  local args=(--context "$CONTEXT" --kv-dtype "$KV_DTYPE" --no-update-check)
  if [[ "$MTP_DRAFTS" == 0 ]]; then
    args+=(--no-drafts)
  else
    args+=(--mtp-drafts "$MTP_DRAFTS" --mtp-confidence "$MTP_CONFIDENCE")
  fi
  (( PARALLEL > 1 )) && args+=(--parallel "$PARALLEL")
  # --vision on both ranks (patch, part 4): the tower loads on rank 0 only; rank 1 reserves no room for it
  # and attaches the features rank 0 sends with each image admission. Both ranks must agree on the flag.
  [[ "$VISION" == 1 ]] && args+=(--vision)
  if [[ "$TP" == 2 ]]; then
    args+=(--tp 2 --rank "$rank" --master "$HEAD_IP" --master-port "$MASTER_PORT")
  fi
  if [[ "$rank" == 0 ]]; then
    args+=(--name "$SERVED_NAME" --host "$API_HOST" --port "$PORT" --max-tokens "$MAX_TOKENS")
    if [[ "$THINKING" == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
    # --tool-system (patch): the server's instruction for tool requests that carry no system message; empty = none
    [[ -n "$TOOL_SYSTEM" ]] && args+=(--tool-system "$TOOL_SYSTEM")
    # the HTTP server's image options (rank 0 decodes the request's images): public https URLs, images a request may carry
    [[ "$VISION" == 1 && "$VISION_URLS" == 1 ]] && args+=(--vision-urls)
    [[ "$VISION" == 1 && -n "$VISION_MAX_IMAGES" ]] && args+=(--vision-max-images "$VISION_MAX_IMAGES")
  fi
  printf '%s\n' "${args[@]}"
}

settle_after_stop() {
  # stop.sh stamps the time it stopped the pair. A rank started within SETTLE_SECONDS of that refused NCCL's first
  # memory registration on the worker (ibv_reg_mr_iova2: Cannot allocate memory; seen on `systemctl restart`, whose
  # ExecStop runs stop.sh right before ExecStart). Wait out the remainder; SETTLE_SECONDS=0 skips it.
  local stamp="$STATE_DIR/stopped_at" settle="${SETTLE_SECONDS:-30}" since
  [[ -f "$stamp" ]] || return 0
  since=$(( $(date +%s) - $(cat "$stamp" 2>/dev/null || echo 0) ))
  (( since >= settle )) && return 0
  log "Waiting $(( settle - since ))s more after the last stop for the ranks' memory to settle"
  sleep $(( settle - since ))
}

ensure_long_profile() {
  # PROFILE=long: a folder of links to the snapshot's files (container paths) with its own config.json:
  # Qwen's YaRN block in rope_parameters and max_position_embeddings at the served window. The pack's bytes are
  # untouched; TensorFold reads the folder as a checkpoint. Rebuilt whenever the pinned snapshot changes.
  local dir="$TF_CACHE/$TF_SHA/long-$LONG_CONTEXT" f b
  if [[ -f "$dir/.snapshot" && "$(cat "$dir/.snapshot")" == "$SNAPSHOT_SHA" ]]; then
    log "Long profile folder $dir (YaRN x$YARN_FACTOR, $LONG_CONTEXT tokens)"
    return
  fi
  rm -rf "$dir"
  mkdir -p "$dir"
  for f in "$SNAPSHOT"/*; do
    b="$(basename "$f")"
    [[ "$b" == config.json ]] && continue
    ln -s "$SNAPSHOT_IN_CONTAINER/$b" "$dir/$b"
  done
  python3 - "$SNAPSHOT/config.json" "$dir/config.json" "$YARN_FACTOR" "$NATIVE_CONTEXT" "$LONG_CONTEXT" <<'PY'
import json
import sys

src, dst, factor, native, window = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = json.load(open(src))
t = c.get("text_config", c)
if int(t.get("max_position_embeddings", 0)) != native:
    sys.exit(f"{src}: max_position_embeddings is {t.get('max_position_embeddings')}, not {native}")
rope = dict(t.get("rope_parameters") or {})
rope.pop("type", None)
rope.update({"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": native})
t["rope_parameters"] = rope
t["max_position_embeddings"] = window
if "text_config" in c:
    c["max_position_embeddings"] = window
json.dump(c, open(dst, "w"), indent=1)
PY
  printf '%s\n' "$SNAPSHOT_SHA" >"$dir/.snapshot"
  log "Built long profile folder $dir (YaRN x$YARN_FACTOR over $NATIVE_CONTEXT, window $LONG_CONTEXT)"
}

ensure_vision_sidecar() {
  # The pack's vision tower is EXL3 (vision_k6.safetensors); TensorFold's CUDA tower reads a one-time fp16
  # conversion (tensorfold.vision.exl3_convert), done here on the CPU inside the image. Rank 0 loads it; rank 1
  # reserves no tower but takes the same flag.
  local out="$TF_CACHE/$TF_SHA/vision-f16-v3.safetensors"
  if [[ -s "$out" ]]; then
    log "Vision tower sidecar $out"
    return
  fi
  log "Converting the pack's EXL3 vision tower to fp16 (once; a few minutes on the CPU)"
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}:ro" -v "${TF_CACHE}/${TF_SHA}:/cache/tf" \
    -e "HF_HOME=$HF_HOME_IN_CONTAINER" -e HF_HUB_OFFLINE=1 \
    --entrypoint python3 "$IMAGE" -m tensorfold.vision.exl3_convert \
    "$SNAPSHOT_IN_CONTAINER/vision_k6.safetensors" /cache/tf/vision-f16-v3.safetensors >/dev/null
  [[ -s "$out" ]] || die "The vision tower conversion left no $out."
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE" "$TF_CACHE/$TF_SHA"
  command -v docker >/dev/null 2>&1 || die "docker not found"
  refuse_foreign_serve
  stop_local
  maybe_drop_caches
  ensure_image
  ensure_weights
  local model_in_container="$SNAPSHOT_IN_CONTAINER"
  if [[ "$PROFILE" == long ]]; then
    ensure_long_profile
    model_in_container="/cache/tf/long-$LONG_CONTEXT"
  fi
  [[ "$VISION" == 1 ]] && ensure_vision_sidecar

  # The snapshot is complete (ensure_weights), so the container never needs the Hub or a token.
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "HF_HUB_OFFLINE=1"
    -e "TENSORFOLD_NO_UPDATE_CHECK=1"
    -e "TORCH_EXTENSIONS_DIR=/cache/tf/torch_extensions"
    -e "TRITON_CACHE_DIR=/cache/tf/triton"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_DEBUG=WARN"
  )
  [[ -n "$MEMORY_RESERVE_GIB" ]] && env_args+=(-e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB")
  [[ "$VISION" == 1 ]] && env_args+=(-e "TENSORFOLD_VISION_WEIGHTS=/cache/tf/vision-f16-v3.safetensors")
  env_args+=("${EXTRA_ENV_ARGS[@]}")

  local args=()
  mapfile -t args < <(serve_args "$rank")

  # --init: rank 1 installs no SIGTERM handler and would ignore it as PID 1, so `docker stop` would wait
  # out its timeout. --ulimit memlock + IPC_LOCK: the 29.8 GiB n-gram tables are mlocked; without them
  # the lock fails silently and lookups can page. --ulimit core=1: no multi-GiB core dumps in host RAM.
  log "Starting $CONTAINER_NAME rank=$rank profile=$PROFILE tp=$TP ctx=$CONTEXT kv=$KV_DTYPE mtp=$MTP_DRAFTS@$MTP_CONFIDENCE parallel=$PARALLEL vision=$VISION hca=$HCA"
  # shellcheck disable=SC2086 # EXTRA_ARGS is word-split on purpose
  docker run -d \
    --name "$CONTAINER_NAME" \
    --init \
    --restart no \
    --oom-score-adj "$OOM_SCORE_ADJ" \
    --ulimit core=1 \
    --gpus all \
    --network host \
    --ipc host \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    -v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}:ro" \
    -v "${TF_CACHE}/${TF_SHA}:/cache/tf" \
    "${env_args[@]}" \
    "$IMAGE" \
    tensorfold serve "$model_in_container" "${args[@]}" $EXTRA_ARGS >/dev/null
  start_memguard
}

# Variables the worker rank needs, forwarded shell-quoted over ssh.
FORWARD_VARS=(
  PROFILE MODEL SERVED_NAME IMAGE TF_SHA TF_PATCH CONTAINER_NAME PORT MASTER_PORT HEAD_IP IFACE HCA TP CONTEXT KV_DTYPE
  TF_PATCH_SHA MTP_DRAFTS MTP_CONFIDENCE PARALLEL THINKING MAX_TOKENS MEMORY_RESERVE_GIB HF_CACHE SNAPSHOT_SHA
  SKIP_DOWNLOAD HF_HUB_DISABLE_XET TF_CACHE OOM_SCORE_ADJ MEMGUARD MEMGUARD_MIN_AVAIL_MB
  MEMGUARD_MIN_SWAP_FREE_MB BENCH_ONLY EXTRA_ARGS EXTRA_ENV TOOL_SYSTEM VISION VISION_URLS VISION_MAX_IMAGES
)

worker_env() {
  local v out="ROLE=worker ORCHESTRATE=0"
  for v in "${FORWARD_VARS[@]}"; do
    out+=" $v=$(printf '%q' "${!v}")"
  done
  printf '%s\n' "$out"
}

worker_state() {
  # Prints true, false or missing; prints nothing when ssh itself fails. docker inspect prints an
  # empty line before failing on a missing container, so strip whitespace.
  { ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
    "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null || echo missing" 2>/dev/null || true; } |
    tr -d '[:space:]'
}

abort_worker_dead() {
  echo "Worker $CONTAINER_NAME on $WORKER_HOST is not running ($1). Worker logs:" >&2
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker logs --tail 120 '$CONTAINER_NAME'" >&2 2>&1 || true
  # The head alone is useless, and left running it poisons the next start: a rank 1 started before
  # it is stopped connects to its rendezvous store and dies when it goes (systemd's retries looped
  # on exactly that). Stop it here, so a retry begins from nothing on both nodes.
  stop_local
  echo "Head container stopped; rank 0 does not serve without rank 1." >&2
  exit 1
}

sync_worker_image() {
  # The worker runs the head's exact image: same image ID, or the head's copy is sent over the link
  # (no second nvcr.io pull, and no rank pair built from two different builds).
  local local_id remote_id
  ensure_image
  local_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
  remote_id="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "docker image inspect -f '{{.Id}}' '$IMAGE' 2>/dev/null || true" | tr -d '[:space:]')" ||
    die "Lost SSH to $WORKER_HOST while comparing image IDs."
  [[ "$remote_id" == "$local_id" ]] && return 0
  log "Copying $IMAGE to $WORKER_HOST (worker has ${remote_id:-none}, head has $local_id)"
  docker save "$IMAGE" | ssh "$WORKER_HOST" docker load >/dev/null
}

wait_ready() {
  local watch_worker="${1:-0}"
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models"
  local i health body state
  for i in $(seq 1 720); do
    health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
    if [[ "$health" == "200" && -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
      log "Ready → http://${API_HOST}:${PORT}/v1  (context=$CONTEXT)"
      printf '%s\n' "$body"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    # Every ~10 s, so a dead worker aborts the head within a minute instead of a 600 s rendezvous wait.
    if [[ "$watch_worker" == 1 ]] && (( i % 2 == 0 )); then
      state="$(worker_state)"
      [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  stop_local                     # a head left from a failed start, before rank 1 could find its store
  refuse_foreign_serve
  refuse_busy_port
  watch_worker=0
  if [[ "$TP" == 2 ]]; then
    if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
      die "Cannot SSH to $WORKER_HOST. Refusing to start a TP=2 head rank alone."
    fi
    mkdir -p "$STATE_DIR"
    printf '%s\n' "$WORKER_HOST" >"$STATE_DIR/worker_host"
    ensure_weights                 # the head's own snapshot, before rank 1 starts and waits on it
    sync_worker_image
    settle_after_stop
    log "Starting rank 1 on $WORKER_HOST first"
    scp -q "$0" "${WORKER_HOST}:/tmp/${CONTAINER_NAME}-run.sh"
    ssh "$WORKER_HOST" "$(worker_env) bash /tmp/${CONTAINER_NAME}-run.sh"
    # Rank 1 waits up to 600 s on rank 0's rendezvous store; give its container a moment to come up.
    sleep 5
    state="$(worker_state)"
    [[ "$state" == false || "$state" == missing ]] && abort_worker_dead "$state"
    watch_worker=1
  fi
  start_local 0
  wait_ready "$watch_worker"
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Rank 1 is up and waits for rank 0 at $HEAD_IP:$MASTER_PORT."
else
  settle_after_stop
  start_local 0
  wait_ready 0
  log "Stop with: ./stop.sh"
fi
