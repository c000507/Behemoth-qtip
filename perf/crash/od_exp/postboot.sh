#!/bin/bash
# after BMC cycle: kernel.sh, pollers, server
bash ~/kernel.sh > /dev/null 2>&1
sudo setsid nohup /home/charles/fastpoll > /tmp/fastpoll.log 2>&1 < /dev/null &
sudo setsid nohup /home/charles/gpu_poll2.sh /tmp/gpu_poll2.csv > /tmp/gpu_poll2.log 2>&1 < /dev/null &
~/gpu_vf_cap.sh > /dev/null
~/od_exp/start_server.sh > /dev/null
