#!/usr/bin/env bash
# Graceful stop of both TensorFold ranks. Rank 0 (HTTP) first: SIGTERM closes its server and its
# rendezvous store, and rank 1 then exits by itself ("rank 0 closed the connection; rank 1 stops").
# A rank stuck in NCCL (its peer died mid-decode) is killed after STOP_TIMEOUT seconds.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-tf-qwen38-exl3}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${WORKER_HOST:-}" && -f "$SCRIPT_DIR/.run-state/worker_host" ]]; then
  WORKER_HOST="$(tr -d '[:space:]' <"$SCRIPT_DIR/.run-state/worker_host")"
fi
WORKER_HOST="${WORKER_HOST:-spark2}"

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Stopping $CONTAINER_NAME"
    docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER_NAME" >/dev/null
    echo "Stopped local $CONTAINER_NAME"
  else
    echo "No local container named $CONTAINER_NAME"
  fi
}

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

stop_local

if [[ "$ORCHESTRATE" == "0" ]]; then
  exit 0
fi

if [[ "$ORCHESTRATE" == "auto" ]]; then
  case "$(host_short)" in
    spark2*) ;;
    *)
      if command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
        echo "Stopping $CONTAINER_NAME on $WORKER_HOST"
        # shellcheck disable=SC2029 # expand locally on purpose
        ssh "$WORKER_HOST" "for i in \$(seq 1 $STOP_TIMEOUT); do [ \"\$(docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null)\" = true ] || break; sleep 1; done; docker stop -t $STOP_TIMEOUT '$CONTAINER_NAME' >/dev/null 2>&1; docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 && echo Stopped remote $CONTAINER_NAME || echo No remote container named $CONTAINER_NAME"
      else
        echo "Cannot SSH to $WORKER_HOST. Remote $CONTAINER_NAME may still be running. Set WORKER_HOST to the rank this head started." >&2
        exit 1
      fi
      ;;
  esac
fi
