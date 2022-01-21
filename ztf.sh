#!/bin/sh

declare -i gpu=1
declare -i epochs=2048

python main.py --dataset ztf_periodic --device cuda:$gpu --e $epochs
python main.py --dataset ztf_stochastic --device cuda:$gpu --e $epochs
python main.py --dataset ztf_transient --device cuda:$gpu --e $epochs