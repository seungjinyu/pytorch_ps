export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Terminal 1 (PS)
python main.py --rank 0 --world_size 3 --role ps
