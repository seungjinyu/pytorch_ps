export MASTER_ADDR=localhost
export MASTER_PORT=29500



# Terminal 3 (Worker 2)
python main.py --rank 2 --world_size 3 --role worker
