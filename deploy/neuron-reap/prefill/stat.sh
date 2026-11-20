#!/bin/bash
# stat.sh: per-expert loads of every 256-token prefill chunk (set ST), code + prose prompts one at a time -> run/st/loads.txt
S=/mnt/scratch; cd $S; echo 2.9 > run/cpt
bash /root/op/srv.sh st sets/ST BUILD BEHEMOTH_NEURON_QWEN35_MOE_LOAD_LOG=1 BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=0 > run/st.srv 2>&1 || { cat run/st.srv; exit 1; }
python3 /root/op/pf.py run/st p4k:4000:1:2 p8k:8000:1:2 q6k:6000:1:2 p32k:32000:1:2 > run/st/pf.txt 2>&1
PF_CORPUS=prose python3 /root/op/pf.py run/st r8k:8000:1:2 r32k:32000:1:2 >> run/st/pf.txt 2>&1
pkill -f neuron_server; sleep 2
/opt/aws_neuronx_venv_pytorch_2_9/bin/python /root/op/loadsw.py run/st/server.log > run/st/loads.txt 2>&1
echo STAT_DONE >> run/st/loads.txt
