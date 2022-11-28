rlaunch --cpu=20 --gpu=4 --memory=204800 -- \
    python train_det.py \
    --num-gpus 4 --resume \
    --config-file expr/vg-bua/config.yaml