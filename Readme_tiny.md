# Dataset Format Supported
```bash
1. ImageNet Subset Format (Your Current Format)
./data/
├── 00500/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 00501/
│   ├── image1.jpg
│   └── ...
└── 00993/
    └── ...

2. Traditional ImageNet Format
```bash
./data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class2/
│       └── ...
└── val/
    ├── class1/
    └── class2/

3. Generic Class Format
```bash
./data/
├── class_a/
│   ├── image1.jpg
│   └── ...
├── class_b/
│   └── ...
└── class_c/
    └── ...

4. HuggingFace Datasets (Online)
Support Streaming Dataset
```

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
