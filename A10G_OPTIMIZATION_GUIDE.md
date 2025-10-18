# A10G GPU Optimization Guide

## Commands to launch g5.2xlarge instance on ec2
```bash
# Start instance
aws ec2 run-instances \
  --image-id ami-0ac1f653c5b6af751 \
  --instance-type g5.2xlarge \
  --key-name aws-kp-lvboy \
  --security-group-ids sg-1c81e178 \
  --subnet-id subnet-006eed59 \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "MaxPrice": "0.55",
      "SpotInstanceType": "one-time",
      "InstanceInterruptionBehavior": "terminate"
    }
  }' \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=resnet50-validation},{Key=Project,Value=mosaic-resnet50}]' \
  --region us-east-1

# Get Running Instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=resnet50-validation" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text --region us-east-1 --profile personal)

# Attach volume
aws ec2 attach-volume \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf \
  --region us-east-1 \
  --profile personal
```

## 🚀 **Quick Start for Maximum Performance**

Your **g5.2xlarge** instance with **A10G GPU (24GB VRAM)** is optimized for high-performance ImageNet training.

### **⚡ One-Command Optimization**

```bash
# Get optimization recommendations
python configs_a10g_optimized.py recommendations

# Run optimized LR finder (with Wandb)
python configs_a10g_optimized.py lr_finder --data_dir ./data

# Run optimized training (with Wandb) 
python configs_a10g_optimized.py full_training_aggressive --data_dir ./data

# Disable Wandb if needed
python configs_a10g_optimized.py lr_finder --data_dir ./data --no_wandb
```

---

## 📊 **A10G Optimized Configurations**

### **🔍 LR Finder (Recommended Start)**
```bash
python configs_a10g_optimized.py lr_finder --data_dir ./data
```
- **Batch Size**: 128 (OOM-safe LR finding)
- **Samples**: 200,000 (fast iteration)
- **Workers**: 8 
- **Features**: AMP enabled, 800 LR iterations, **Wandb logging**
- **Output**: `./outputs/lr_finder_clean_imagenet.png` + `suggested_lr.json`
- **Wandb Project**: `imagenet-lr-finder-a10g`

### **🏃 Sample Training (Fast Iteration)**
```bash
# Use auto-detected LR from LR finder
python configs_a10g_optimized.py sample_training --data_dir ./data --auto_lr

# Or manual LR
python configs_a10g_optimized.py sample_training --data_dir ./data --lr 0.001
```
- **Batch Size**: 128 (conservative for OOM avoidance)
- **Samples**: 25,000 (manageable subset)
- **Epochs**: 20
- **Scheduler**: OneCycleLR (fast convergence)
- **Features**: **Auto-LR support**, **Wandb logging**, AMP enabled
- **Wandb Project**: `imagenet-sample-training-a10g`
- **Time**: ~15-30 minutes

### **🎯 Full Training - Conservative (Recommended)**
```bash
# Use auto-detected LR from LR finder
python configs_a10g_optimized.py full_training_conservative --data_dir ./data --auto_lr

# Or manual LR
python configs_a10g_optimized.py full_training_conservative --data_dir ./data --lr 0.001
```
- **Batch Size**: 64 (very safe for full dataset, no OOM)
- **Epochs**: 50
- **Workers**: 16
- **Features**: **Auto-LR support**, AMP, **Wandb logging**, Model checkpoints
- **Wandb Project**: `imagenet-full-training-a10g`
- **Memory**: ~12GB VRAM usage

### **🔥 Full Training - Aggressive (Higher Performance)**
```bash
# Use auto-detected LR from LR finder
python configs_a10g_optimized.py full_training_aggressive --data_dir ./data --auto_lr

# Or manual LR
python configs_a10g_optimized.py full_training_aggressive --data_dir ./data --lr 0.001
```
- **Batch Size**: 128 (higher performance, gradient accumulation = effective 256)
- **Epochs**: 50
- **Workers**: 16
- **Features**: **Auto-LR support**, **Wandb logging**, AMP, Model checkpoints
- **Wandb Project**: `imagenet-full-training-a10g`
- **Memory**: ~16GB VRAM usage

### **🏆 Speed Benchmark**
```bash
python configs_a10g_optimized.py speed_benchmark --data_dir ./data --auto_lr
```
- **Batch Size**: 192 (maximum safe throughput)
- **Purpose**: Test safe maximum throughput
- **Features**: **Wandb logging** for performance metrics
- **Wandb Project**: `imagenet-benchmark-a10g`
- **Duration**: ~5-10 minutes

---

## 🛠️ **Performance Optimizations Applied**

### **Automatic CUDA Optimizations**
When you run any training, these are automatically applied:

```
✅ cuDNN benchmark enabled
✅ TF32 enabled for matmul  
✅ TF32 enabled for cuDNN
✅ Memory fraction set to 95%
✅ CUDA_LAUNCH_BLOCKING = 0
✅ PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:512
```

### **Model Optimizations**
```
✅ torch.compile enabled (max-autotune)  [PyTorch 2.0+]
✅ Mixed precision training (AMP)
✅ Gradient clipping (norm=1.0)
```

### **Data Loading Optimizations**
```
✅ Pin memory enabled
✅ 12-16 workers (optimal for g5.2xlarge 8 vCPUs)
✅ Prefetch factor optimized
```

---

## 📈 **Expected Performance**

### **A10G Specifications**
- **VRAM**: 24GB GDDR6 
- **Compute**: 9,216 CUDA cores + 288 Tensor cores
- **Memory BW**: 600 GB/s
- **TensorFloat-32**: Supported (significant speedup)

### **Throughput Estimates (OOM-Safe)**
| Configuration | Batch Size | Samples/sec | Time/Epoch* | VRAM Usage |
|---------------|------------|-------------|-------------|------------|
| LR Finder     | 256        | ~800        | 75 sec      | ~10GB      |
| Sample (25k)  | 128        | ~600        | 125 sec     | ~8GB       |
| Conservative  | 64         | ~400        | 150 sec     | ~6GB       |
| Aggressive    | 128        | ~600        | 100 sec     | ~12GB      |
| Benchmark     | 192        | ~850        | 70 sec      | ~14GB      |

*Estimated for 60k samples with data loading optimizations. Conservative estimates to avoid OOM.

---

## 🔍 **Monitoring GPU Usage**

### **Real-time Monitoring**
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Memory usage  
nvidia-smi dmon -s um

# Power & temperature
nvidia-smi dmon -s pt
```

### **Target Metrics**
- **GPU Utilization**: 95-100%
- **Memory Usage**: 18-22GB (75-90% of 24GB)
- **Power**: 250-300W
- **Temperature**: <80°C

---

## ⚠️ **Troubleshooting**

### **Out of Memory (OOM)**
```bash
# Reduce batch size by 25%
python configs_a10g_optimized.py full_training_conservative --data_dir ./data

# Or use gradient accumulation
python main.py --batch_size 256 --gradient_accumulation_steps 2 --data_dir ./data
```

### **Low GPU Utilization (<80%)**
1. **Increase batch size**: Try aggressive config
2. **Increase workers**: Add `--num_workers 20`
3. **Check data bottleneck**: Monitor CPU usage

### **Slow Data Loading**
```bash
# Increase workers
python main.py --num_workers 16 --batch_size 512 --data_dir ./data

# Enable pin memory (auto-enabled)
# Reduce data augmentation if needed
```

---

## 🎯 **Recommended Workflow**

### **1. Find Optimal LR (2 minutes)**
```bash
python configs_a10g_optimized.py lr_finder --data_dir ./data
```

### **2. Quick Validation (20 minutes)**
```bash
python configs_a10g_optimized.py sample_training --data_dir ./data
```

### **3. Full Training (2-4 hours)**
```bash
# Conservative (recommended)
python configs_a10g_optimized.py full_training_conservative --data_dir ./data

# Or aggressive (if conservative works well)
python configs_a10g_optimized.py full_training_aggressive --data_dir ./data
```

### **4. Monitor Progress**
```bash
# Terminal 1: Training
python configs_a10g_optimized.py full_training_conservative --data_dir ./data

# Terminal 2: Monitoring  
watch -n 1 nvidia-smi
```

---

## 💡 **Pro Tips**

1. **Start Conservative**: Use `full_training_conservative` first
2. **Monitor Memory**: Keep VRAM usage <22GB for stability  
3. **Use Wandb**: Add `--use_wandb` for experiment tracking
4. **Save Checkpoints**: Enabled by default every 5 epochs
5. **Test First**: Run `sample_training` before full runs
6. **Power Limit**: A10G can handle sustained 300W loads

---

## 🔧 **Manual Optimization**

If you want to manually tune parameters:

```bash
# Maximum safe batch size for your dataset
python main.py \
  --batch_size 512 \
  --num_workers 16 \
  --amp \
  --scheduler cosine \
  --epochs 50 \
  --data_dir ./data \
  --use_wandb \
  --lr 1.6e-05
```

### **Key Parameters for A10G**
- `--batch_size`: 256-768 (dataset dependent)
- `--num_workers`: 12-20 (I/O optimization)
- `--amp`: Always enable (free 1.5-2x speedup)
- `--scheduler cosine`: Stable for long training
- `--scheduler onecycle`: Fast convergence

---

**Your A10G setup is now optimized for maximum ImageNet training performance!** 🚀

Use `python configs_a10g_optimized.py recommendations` anytime for system-specific advice.
