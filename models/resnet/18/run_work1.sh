export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Terminal 2 (Worker 1)
python main.py --rank 1 --world_size 3 --role worker
