#!/bin/bash

set -e
set -x

docker-compose build benchmark-ann-gsi

DT=$(date +%s)
mkdir -p "./results/logs"
LOG="./results/logs/$(echo $DT)_LOG_$(echo $RAN | md5sum | head -c 20).txt"

docker-compose up benchmark-ann-gsi 2>&1 | tee "$LOG"

echo "Done. Wrote $LOG"
