#!/bin/bash
for i in $(seq 120); do grep -q Listening /tmp/beh_server.log && break; sleep 5; done
~/gpu_vf_cap.sh >/dev/null
cd /home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
timeout 200 build-rocm/behemoth_load_bench ~/stress/prompt.txt 1 1 16 127.0.0.1 1337 | grep per_second
timeout 200 build-rocm/behemoth_load_bench ~/stress/prompt.txt 4 4 32 127.0.0.1 1337 | grep per_second
