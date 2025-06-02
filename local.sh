#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "[INFO] Starting Parameter Server..."
python ps.py &

sleep 2

echo "[INFO] Starting Worker 1..."
python worker.py --rank 1 --world_size 3 &

sleep 1

echo "[INFO] Starting Worker 2..."
python worker.py --rank 2 --world_size 3 &

wait
