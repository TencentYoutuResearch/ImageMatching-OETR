python3 -m torch.distributed.launch --nproc_per_node=2 train.py --config_path configs/baseline/oetr_config.py --validation --epoch 35 --batch_size 2
