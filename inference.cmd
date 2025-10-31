uv run python inference.py \
    --checkpoint /mnt/data/ImageNetClassifier/checkpoints/last-v2.ckpt \
    --data_dir ../data \
    --batch_size 256 \
    --num_workers 4 \
    --output_dir ./inference_output \
    --plot_name training_curves.png \
    --log_dir ./logs \
    --experiment_name imagenet_resnet50
