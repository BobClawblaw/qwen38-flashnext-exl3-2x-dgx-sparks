#!/usr/bin/env bash
# Gate pack for one session on a live TensorFold serve.
#
#   tools/session_gate.sh EVDIR
#
# Order: receipts and free -h (before) -> /health snapshot -> smokes -> frozen bench_decode.py
# (the sibling vLLM recipe's ruler, byte-identical) -> tools/bench_cells.py (longer c=1 cells)
# -> /health snapshot -> receipts (free -h both nodes, docker logs of both ranks, per-request
# "done" lines, MTP acceptance from /health deltas, harness sha256) -> gate.txt.
# Exits nonzero if any step fails.
#
# Env:
#   RUN_FROZEN=0     skip the frozen bench_decode.py run.
#   RUN_CELLS=0      skip tools/bench_cells.py.
#   CELLS_ARGS       extra args for tools/bench_cells.py.
#   URL, MODEL, WORKER_HOST, CONTAINER_NAME, PORT as in run.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
EV="$(mkdir -p "$1" && cd "$1" && pwd)"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
URL="${URL:-${BASE}/v1/chat/completions}"
MODEL="${MODEL:-turboderp/Qwen3.8-Flash-Next-exl3}"
WORKER_HOST="${WORKER_HOST:-spark2}"
CONTAINER_NAME="${CONTAINER_NAME:-tf-qwen38-flashnext}"
RUN_FROZEN="${RUN_FROZEN:-1}"
RUN_CELLS="${RUN_CELLS:-1}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
cd "$ROOT" || exit 2

declare -a RESULTS=()
FAILED=0
stamp() { date -u +%FT%T.%3NZ; }
log() { echo "$(stamp) session_gate: $*" | tee -a "$EV/gate.log"; }

# step NAME CMD... : run with stdout/stderr/exit captured under EVDIR.
step() {
  local name="$1"; shift
  log "start $name"
  "$@" >"$EV/$name.out" 2>"$EV/$name.err"
  local rc=$?
  echo "$rc" >"$EV/$name.exit"
  RESULTS+=("$name=$rc")
  [[ $rc -eq 0 ]] || FAILED=1
  log "end $name rc=$rc"
  return 0
}

free_both() {
  free -h >"$EV/free-$1.txt" 2>&1
  "${SSH[@]}" "$WORKER_HOST" 'free -h' >"$EV/free-$1-${WORKER_HOST}.txt" 2>&1 || echo "ssh failed" >>"$EV/free-$1-${WORKER_HOST}.txt"
}

logs_both() {
  docker logs "$CONTAINER_NAME" 2>&1 | grep -v 'GET /v1/models\|GET /health\|GET /metrics' >"$EV/docker-head.log" || true
  "${SSH[@]}" "$WORKER_HOST" "docker logs '$CONTAINER_NAME' 2>&1" >"$EV/docker-${WORKER_HOST}.log" 2>&1 || true
  grep ' done req-' "$EV/docker-head.log" >"$EV/requests.log" || true
  grep -E '^\[tensorfold\] (CUDA rank|Flash Next on CUDA|serving|rank 1 ready)' "$EV/docker-head.log" "$EV/docker-${WORKER_HOST}.log" >"$EV/startup.txt" || true
  grep -E 'Traceback|Error|error:|OutOfStep|NCCL WARN.*(timeout|abort)' "$EV/docker-head.log" "$EV/docker-${WORKER_HOST}.log" >"$EV/engine-needles.txt" || true
}

# --- preflight ---------------------------------------------------------------
log "evidence dir $EV"
code="$(curl -s -o "$EV/health-before.json" -w '%{http_code}' --max-time 5 "$BASE/health" || true)"
echo "$code" >"$EV/health.code"
if [[ "$code" != 200 ]]; then
  log "GET /health = ${code:-none}; refusing to run the gate on a server that is not up"
  exit 1
fi
curl -sS --max-time 5 "$BASE/v1/models" >"$EV/models.json" || true
if ! grep -q "\"$MODEL\"" "$EV/models.json"; then
  log "/v1/models does not list $MODEL"
  exit 1
fi
sha256sum bench_decode.py smoke_*.py tools/bench_cells.py tools/accept.py tools/session_gate.sh >"$EV/harness.sha256"
git rev-parse HEAD >"$EV/git-head.txt" 2>/dev/null || true
docker inspect -f '{{.Config.Image}} {{join .Args " "}}' "$CONTAINER_NAME" >"$EV/serve-argv.txt" 2>&1 || true
free_both before

# --- smokes ------------------------------------------------------------------
step smoke-thinking python3 smoke_thinking.py --url "$URL" --model "$MODEL"
step smoke-tools python3 smoke_tools.py --url "$URL" --model "$MODEL"
step smoke-count python3 smoke_count.py --url "$URL" --model "$MODEL"

# --- benches -----------------------------------------------------------------
if [[ "$RUN_FROZEN" == 1 ]]; then
  curl -sS --max-time 5 "$BASE/health" >"$EV/health-frozen-before.json" || true
  step bench-frozen python3 bench_decode.py --url "$URL" --model "$MODEL" --runs 3 --concurrency 1 2 --max-tokens 200
  curl -sS --max-time 5 "$BASE/health" >"$EV/health-frozen-after.json" || true
  python3 tools/accept.py "$EV/health-frozen-before.json" "$EV/health-frozen-after.json" >"$EV/accept-frozen.txt" 2>&1 || true
fi
if [[ "$RUN_CELLS" == 1 ]]; then
  # shellcheck disable=SC2086
  step bench-cells python3 tools/bench_cells.py --url "$URL" --model "$MODEL" --base "$BASE" ${CELLS_ARGS:-}
fi

# --- receipts ----------------------------------------------------------------
curl -sS --max-time 5 "$BASE/health" >"$EV/health-after.json" || true
python3 tools/accept.py "$EV/health-before.json" "$EV/health-after.json" >"$EV/accept-session.txt" 2>&1 || true
free_both after
logs_both
{
  echo "finished_at=$(stamp)"
  printf '%s\n' "${RESULTS[@]}"
  if [[ $FAILED -eq 0 ]]; then echo "GATE=PASS"; else echo "GATE=FAIL"; fi
} >"$EV/gate.txt"
tee -a "$EV/gate.log" <"$EV/gate.txt"
exit "$FAILED"
