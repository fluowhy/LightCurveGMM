#!/bin/sh

declare -i gpu=1
declare -i epochs=2048

python main.py --dataset asas_sn --device cuda:$gpu --oc 8 --e $epochs