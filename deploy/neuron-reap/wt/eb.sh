#!/bin/bash
# eb.sh OUTDIR: build the engine (srv + layer_bench + neff_io) from /mnt/scratch/src into OUTDIR -> /mnt/scratch/log/eb_<name>.log
S=/mnt/scratch; O=$1; L=$S/log/eb_$(basename $O).log
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH
(cd $S/src && PATH=$S/shim:$PATH bash neuron/cpp/build.sh $O srv) > $L 2>&1
$S/shim/g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include $S/src/neuron/cpp/layer_bench.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o $O/layer_bench >> $L 2>&1
$S/shim/g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include $S/src/neuron/cpp/neff_io.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o $O/neff_io >> $L 2>&1
ls -la $O/neuron_server >> $L 2>&1 && echo EB_OK >> $L || echo EB_FAIL >> $L
