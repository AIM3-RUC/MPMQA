CFG=$1
GPUS=$2
CUDA_VISIBLE_DEVICES=$GPUS deepspeed train.py --deepspeed \
    --config $CFG