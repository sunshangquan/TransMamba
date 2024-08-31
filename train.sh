!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 --use_env basicsr/train.py -opt Deraining/Options/Deraining_transMamba_DDN.yml --launcher pytorch