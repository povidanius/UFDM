#!/bin/bash

docker run \
--shm-size=8gb \
--gpus all \
-v $(pwd):/UFDM \
-p 5001:5000 \
-v /dev:/dev \
--env-file .env \
-it ufdm


