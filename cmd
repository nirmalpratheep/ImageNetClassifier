uv run python main_lightning.py --batch_size 128 --find_lr --lr_start 1e-05 --lr_end 10 --lr_iter 20 --lr_plot ./outputs/lr_finder_imagenet1k.png --data_dir ../data --epochs 10 --use_multi_gpu
