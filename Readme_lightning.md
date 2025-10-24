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
```
