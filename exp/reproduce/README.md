manual_seed.py 
-> gets the same gradient


# Build image 
docker build -t deterministic-resnet-arm .

# Get in docker and exec bin bash
docker run --rm -it \
    -v $PWD:/workspace \
    -v $PWD/data:/workspace/data \
    deterministic-resnet-arm /bin/bash
