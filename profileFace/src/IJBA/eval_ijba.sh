#!/usr/bin/env sh 
CUDA_VISIBLE_DEVICES=0 python eval_ijba.py 2>&1 | tee eval_res.txt
