#!/bin/bash
# systemd ExecStart wrapper. HOME, PATH and PROFILE come from the unit; run.sh reads .env.cluster next to it.
cd "$(dirname "$(readlink -f "$0")")" && exec ./run.sh
