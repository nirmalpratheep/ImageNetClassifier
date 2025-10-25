1. Tiny Imagent
```bash
uv run  lightning_main.py --data_dir /Users/uv/Documents/work/gitrepos/imagenet1k-trainer/artifacts/tinyimagenet --dataset tinyimagenet --batch_size 256 --max_epochs 20 --lr_finder --plot_lr 
#training without lr
uv run  lightning_main.py --data_dir /Users/uv/Documents/work/gitrepos/imagenet1k-trainer/artifacts/tinyimagenet --dataset tinyimagenet --batch_size 256 --max_epochs 20
#tensorboard logs
tensorboard --logdir ./results/logs/tensorboard_logs
```
2. Imagenet
```bash
uv run lightning_main.py --data_dir ./data --dataset imagenet --lr_finder --plot_lr
# with label smoothening and better initialization
uv run  lightning_main.py --data_dir ./data --dataset imagenet --loss_type bce_with_logits --label_smoothing 0.1
# with warmup
uv run lightning_main.py --data_dir ./data --loss_type bce_with_logits --label_smoothing 0.1 --warmup_epochs 5 --warmup_start_lr 1e-6 --init_bce_bias

# with augmentation
uv run lightning_main.py --data_dir ./data --loss_type bce_with_logits --label_smoothing 0.1 --warmup_epochs 5 --warmup_start_lr 1e-6 --init_bce_bias --random_erasing_p 0.5 --mixup_alpha 0.2 --cutmix_alpha 1.0 --cutmix_prob 0.5


uv run lightning_main.py --data_dir /mnt/data/imagenet-1k/  --dataset imagenet --loss_type bce_with_logits --label_smoothing 0.1 --warmup_epochs 5 --warmup_start_lr 1e-6 --init_bce_bias --random_erasing_p 0.5 --mixup_alpha 0.2 --cutmix_alpha 1.0 --cutmix_prob 0.5 --max_epochs 90 --batch_size 256  --results_dir /mnt/data/results/ --num_workers 8

```