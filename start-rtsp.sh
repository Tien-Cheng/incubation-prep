#!/bin/sh
docker run --rm -it --network=host --env READ_BUFFER_COUNT=2048 aler9/rtsp-simple-server