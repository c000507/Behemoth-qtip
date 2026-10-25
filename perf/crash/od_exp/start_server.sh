#!/bin/bash
cd /home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime && setsid nohup build-rocm/behemoth_server --model /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output --devices 0,1 --transport zerocopy --code-layout sorted --memory-fraction 0.9 --token-budget 1024 --prefill-chunk 256 --host 0.0.0.0 --port 1337 --threads 4 --profile false > /tmp/beh_server.log 2>&1 < /dev/null &
sleep 1; pgrep -x behemoth_server > /tmp/beh_server.pid; cat /tmp/beh_server.pid
