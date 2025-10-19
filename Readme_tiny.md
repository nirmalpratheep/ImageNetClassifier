# Improvements

1. Improvement in the BCE Loss Function
https://www.databricks.com/blog/mosaic-resnet-deep-dive
```bash
reduction="sum" and then divide the loss by the number of samples in the batch.
&
initializing the logit biases to -log(n_classes) such that the initial outputs are roughly equal to 1/n_classes
#
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --lr_plot ./outputs/lr_finder_imagenet1k.png \
    --data_dir ../imagenet1k-trainer/artifacts/tinyimagenet/tiny-val \
    --epochs 1 \
    --no_plots \
    --loss_function bce_with_logits \
    --bce_use_bias_init
```
