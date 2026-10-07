#!/usr/bin/env bash
# One boot + gate per line of a spec file: `ID VAR=value ...` (run.sh variables, '#' comments).
#
#   tools/sweep.sh SPEC EVROOT
#
# Each config: ./stop.sh, then `env VARS ./run.sh`, then tools/session_gate.sh EVROOT/ID. A boot that
# fails is recorded as EVROOT/ID/boot.failed and the sweep moves on. The serve left up at the end is
# the last config's; restart the default with ./stop.sh && ./run.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 2 ]] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
SPEC="$1"; EVROOT="$2"
cd "$ROOT" || exit 2
# The spec is read on fd 3: ssh inside run.sh would otherwise swallow the rest of it from stdin.
while read -r id rest <&3; do
  [[ -z "$id" || "$id" == \#* ]] && continue
  ev="$EVROOT/$id"
  mkdir -p "$ev"
  echo "$(date -u +%FT%TZ) sweep: $id $rest"
  ./stop.sh >"$ev/stop.log" 2>&1 </dev/null
  # shellcheck disable=SC2086 # the spec line is a list of VAR=value words
  if ! env $rest ./run.sh >"$ev/boot.log" 2>&1 </dev/null; then
    echo "$(date -u +%FT%TZ) sweep: $id boot failed" | tee "$ev/boot.failed"
    continue
  fi
  printf '%s\n' "$rest" >"$ev/config.txt"
  tools/session_gate.sh "$ev" >/dev/null 2>&1 </dev/null
  echo "$(date -u +%FT%TZ) sweep: $id $(tail -1 "$ev/gate.txt")"
done 3<"$SPEC"
