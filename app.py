"""
ImageNet-1K Image Classifier
============================
Simple ResNet-50 model trained on ImageNet-1000.

Architecture: ResNet-50
- 1000 output classes (ImageNet-1K)
- Deep residual learning for robust feature extraction
"""

import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from pathlib import Path
import numpy as np
import json
import cv2
from torchvision import transforms

# Import model architecture and data module
from model_resnet50 import ResNet50
from data_module import ImageNetDataModule
from dataset_loader import get_transforms

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ImageNet class names
def load_imagenet_classes(json_path="ImageNet_class_index.json"):
    """Load ImageNet class names from JSON file."""
    try:
        with open(json_path, 'r') as f:
            class_index = json.load(f)
        # Create mapping: class_index -> class_name
        # Structure: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
        class_names = {}
        for idx_str, value_list in class_index.items():
            if isinstance(value_list, list) and len(value_list) >= 2:
                class_names[int(idx_str)] = value_list[1]  # Second element is the class name
            elif isinstance(value_list, list) and len(value_list) >= 1:
                class_names[int(idx_str)] = value_list[0]
        print(f"✅ Loaded {len(class_names)} ImageNet class names")
        return class_names
    except Exception as e:
        print(f"⚠️ Could not load ImageNet class names: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Load class names
IMAGENET_CLASSES = load_imagenet_classes()

def get_class_name(class_idx):
    """Get class name from index, fallback to 'Class {idx}' if not found."""
    return IMAGENET_CLASSES.get(class_idx, f"Class {class_idx}")

# Load ImageNet validation labels and create mappings
def load_imagenet_val_labels(txt_path="ImageNet_val_label.txt", json_path="ImageNet_class_index.json"):
    """Load validation labels and create mappings."""
    # Mapping: filename -> synset_id
    filename_to_synset = {}
    # Mapping: synset_id -> class_index
    synset_to_class_idx = {}
    
    try:
        # Load val labels (filename -> synset_id)
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        synset_id = parts[1]
                        filename_to_synset[filename] = synset_id
        
        # Load class index to create synset -> class_idx mapping
        with open(json_path, 'r') as f:
            class_index = json.load(f)
            for class_idx_str, value_list in class_index.items():
                if isinstance(value_list, list) and len(value_list) >= 1:
                    synset_id = value_list[0]
                    class_idx = int(class_idx_str)
                    synset_to_class_idx[synset_id] = class_idx
        
        print(f"✅ Loaded {len(filename_to_synset)} validation image labels")
        print(f"✅ Loaded {len(synset_to_class_idx)} synset to class index mappings")
        return filename_to_synset, synset_to_class_idx
    except Exception as e:
        print(f"⚠️ Could not load validation labels: {e}")
        return {}, {}

# Load label mappings
VAL_FILENAME_TO_SYNSET, SYNSET_TO_CLASS_IDX = load_imagenet_val_labels()

def get_class_name_from_filename(filename):
    """Get class name from image filename using validation labels."""
    # Extract just the filename from path
    filename_only = Path(filename).name
    
    # Look up synset ID
    synset_id = VAL_FILENAME_TO_SYNSET.get(filename_only)
    if synset_id:
        # Look up class index
        class_idx = SYNSET_TO_CLASS_IDX.get(synset_id)
        if class_idx is not None:
            # Get class name
            return get_class_name(class_idx)
    
    # Fallback if not found
    return None

# Store a mapping for example images to their paths and labels
EXAMPLE_IMAGE_MAP = {}

# ---------------------------
# Load Model
# ---------------------------
@torch.no_grad()
def load_model(checkpoint_path: str = None):
    """Load the trained ResNet-50 model for ImageNet-1000"""
    model = ResNet50(num_classes=1000).to(device)
    
    # Store a sample weight from random initialization to verify loading
    first_layer_weight_before = model.conv1.weight.data[0, 0, 0, 0].clone().item()
    
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', '?')
                test_acc = checkpoint.get('test_acc', [])
                if test_acc:
                    best_acc = max(test_acc)
                    print(f"✅ Found checkpoint with model_state_dict from epoch {epoch}, best test acc: {best_acc:.2f}%")
                else:
                    print(f"✅ Found checkpoint with model_state_dict from epoch {epoch}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"✅ Found checkpoint with state_dict")
            else:
                state_dict = checkpoint
                print(f"✅ Found checkpoint with direct state_dict")
            
            # Debug: Show some checkpoint keys
            checkpoint_keys = list(state_dict.keys())[:5]
            print(f"🔍 Sample checkpoint keys: {checkpoint_keys}")
            print(f"🔍 Total checkpoint keys: {len(state_dict)}")
            
            # Handle various prefixes that might be in the checkpoint
            # Check for 'model.' prefix (from nested model structure)
            has_model_prefix = any(key.startswith('model.') for key in state_dict.keys())
            # Check for 'module.' prefix (from DataParallel)
            has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
            
            if has_model_prefix:
                print("⚠️ Detected 'model.' prefix in checkpoint, removing...")
                new_state_dict = {}
                removed_count = 0
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix (6 characters)
                        new_state_dict[new_key] = value
                        removed_count += 1
                    elif key.startswith('module.'):
                        # Handle both prefixes if present
                        new_key = key[7:]  # Remove 'module.' prefix first
                        if new_key.startswith('model.'):
                            new_key = new_key[6:]  # Then remove 'model.' prefix
                        new_state_dict[new_key] = value
                        removed_count += 1
                    else:
                        new_state_dict[key] = value
                print(f"✅ Removed 'model.' prefix from {removed_count} keys")
                state_dict = new_state_dict
            elif has_module_prefix:
                print("⚠️ Detected 'module.' prefix in checkpoint (DataParallel), removing...")
                new_state_dict = {}
                removed_count = 0
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # Remove 'module.' prefix
                        new_state_dict[new_key] = value
                        removed_count += 1
                    else:
                        new_state_dict[key] = value
                print(f"✅ Removed 'module.' prefix from {removed_count} keys")
                state_dict = new_state_dict
            
            # Check model keys
            model_keys = list(model.state_dict().keys())[:5]
            model_total_keys = len(model.state_dict())
            print(f"🔍 Sample model keys: {model_keys}")
            print(f"🔍 Total model keys: {model_total_keys}")
            
            # Count matching keys before loading
            model_state = model.state_dict()
            matching_keys = sum(1 for k in state_dict.keys() if k in model_state and state_dict[k].shape == model_state[k].shape)
            print(f"🔍 Matching keys: {matching_keys}/{len(model_state)}")
            
            # Try loading
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)} total (first 5: {missing_keys[:5]})")
                if len(missing_keys) > 10:
                    print("⚠️ WARNING: Too many missing keys! Model may not load correctly.")
                    print("⚠️ This could cause poor predictions - consider using a different checkpoint.")
            else:
                print("✅ All model keys matched!")
                
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)} total (first 5: {unexpected_keys[:5]})")
            
            # Verify weights actually changed
            first_layer_weight_after = model.conv1.weight.data[0, 0, 0, 0].item()
            weight_diff = abs(first_layer_weight_before - first_layer_weight_after)
            if weight_diff < 1e-6:
                print("⚠️ WARNING: Model weights did not change! Checkpoint may not have loaded correctly.")
                print("⚠️ Model may be using random weights - predictions will be inaccurate!")
            else:
                print(f"✅ Verified: Model weights changed (first conv weight: {first_layer_weight_before:.6f} -> {first_layer_weight_after:.6f}, diff: {weight_diff:.6f})")
            
            print(f"✅ Successfully loaded model weights from {checkpoint_path}")
            
            # Test prediction to verify model is working
            test_input = torch.randn(1, 3, 224, 224).to(device)
            # Normalize properly for ImageNet
            mean_tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
            std_tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
            test_input = (test_input - mean_tensor) / std_tensor
            
            with torch.no_grad():
                test_output = model(test_input)
                test_pred = test_output.argmax(dim=1).item()
                test_prob = torch.softmax(test_output, dim=1)[0, test_pred].item()
                test_class_name = get_class_name(test_pred)
            print(f"🔍 Model test: Predicted class {test_pred} ({test_class_name}) with confidence {test_prob:.4f}")
            
            # Check if model is producing diverse outputs (not always the same class)
            test_outputs = []
            for i in range(5):
                test_in = torch.randn(1, 3, 224, 224).to(device)
                test_in = (test_in - mean_tensor) / std_tensor
                with torch.no_grad():
                    out = model(test_in)
                    pred = out.argmax(dim=1).item()
                    prob = torch.softmax(out, dim=1)[0, pred].item()
                    test_outputs.append((pred, prob))
            unique_preds = len(set(p[0] for p in test_outputs))
            if unique_preds == 1:
                pred_idx, pred_prob = test_outputs[0]
                print(f"⚠️ WARNING: Model always predicts class {pred_idx} ({get_class_name(pred_idx)}) with prob {pred_prob:.4f}")
                print(f"⚠️ Model may not be loaded correctly or checkpoint may be corrupted!")
            else:
                print(f"✅ Model produces diverse predictions: {unique_preds} different classes in 5 tests")
            
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("Using randomly initialized model")
    else:
        print("ℹ️ No checkpoint provided, using randomly initialized model")
    
    model.eval()
    return model


print(f"Device: {device}")
# Try to load checkpoint
checkpoint_paths = [
    "./last-v2.ckpt",
    "./last-v2.pth",
    "./snapshots/resnet50_epoch_*.pth",
    None  # Fallback to random initialization
]

model = None
loaded_checkpoint = None
for checkpoint_path in checkpoint_paths:
    if checkpoint_path is None:
        continue
    elif "*" in checkpoint_path:
        # Handle glob pattern
        import glob
        matches = glob.glob(checkpoint_path)
        if matches:
            loaded_checkpoint = matches[0]
            model = load_model(loaded_checkpoint)
            break
    elif Path(checkpoint_path).exists():
        loaded_checkpoint = checkpoint_path
        model = load_model(loaded_checkpoint)
        break

if model is None:
    print("⚠️ No checkpoint found, using randomly initialized model")
    model = load_model(None)
else:
    print(f"✅ Successfully loaded checkpoint: {loaded_checkpoint}")

# ---------------------------
# Preprocessing pipeline
# ---------------------------
# Use the SAME preprocessing as inference.py - use get_transforms from dataset_loader
# This ensures we match exactly what was used during training/validation
_, val_transform = get_transforms(image_size=224, augmentation=False)

def preprocess(image):
    """Preprocess image using the same transform as validation."""
    # Convert PIL to format expected by transform
    if isinstance(image, Image.Image):
        # Apply the validation transform (which handles PIL -> numpy -> tensor conversion)
        result = val_transform(image)
        return result
    else:
        # If already numpy array, convert to PIL first
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return val_transform(image)

print("✅ Using validation transforms from dataset_loader (matches inference.py)")

# Preprocess without normalization for Grad-CAM
preprocess_no_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------
# Grad-CAM Implementation
# ---------------------------
class GradCAM:
    """Grad-CAM: Visual Explanations from Deep Networks"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_class] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), target_class


def apply_gradcam(image_pil, model, gradcam, top_class_idx, original_size=None):
    """Apply Grad-CAM and overlay on original image"""
    # Prepare input using same preprocessing as inference
    img_tensor = preprocess(image_pil.convert("RGB"))
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.from_numpy(img_tensor).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    cam, _ = gradcam.generate_cam(img_tensor, target_class=top_class_idx)
    
    # Use original image size or default to 224
    if original_size is None:
        original_size = image_pil.size
    
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, original_size)
    
    # Convert original image to numpy
    img_np = np.array(image_pil.resize(original_size))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return overlay, heatmap


# Initialize Grad-CAM (target the last convolutional layer)
# Note: ResNet-50 structure - layer4[-1].conv3 is the last conv in the last bottleneck
try:
    gradcam = GradCAM(model, model.layer4[-1].conv3)
except:
    # Fallback if model structure is different
    try:
        gradcam = GradCAM(model, model.layer4[-1].conv2)
    except:
        gradcam = None
        print("⚠️ Could not initialize Grad-CAM")


# ---------------------------
# Prediction Function
# ---------------------------
def predict(image: Image.Image, image_path: str = None):
    """Predict the class of an input image with Grad-CAM visualization."""
    if image is None:
        return {}, "<p style='color: red;'>Please upload an image first!</p>", None, None
    
    try:
        # Get ground truth label if image path is provided
        gt_label = None
        if image_path:
            gt_label = get_class_name_from_filename(image_path)
        
        # Prepare input - use same preprocessing as inference.py
        img_rgb = image.convert("RGB")
        
        # Apply preprocessing (uses get_transforms from dataset_loader, same as inference.py)
        img_tensor = preprocess(img_rgb)
        
        # Verify it's a tensor and add batch dimension
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.from_numpy(img_tensor).float()
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Get predictions
        # Ensure model is in eval mode (important for BatchNorm, Dropout, etc.)
        model.eval()
        # Also ensure no gradients are tracked
        with torch.no_grad():
            # Forward pass
            outputs = model(img_tensor)
            
            # Debug: Check if model is actually in eval mode
            if model.training:
                print("⚠️ WARNING: Model is in training mode! Setting to eval mode.")
                model.eval()
                outputs = model(img_tensor)  # Re-run with eval mode
            
            # Check if outputs are reasonable (not all zeros or NaNs)
            if torch.isnan(outputs).any():
                print("⚠️ WARNING: Model output contains NaN values!")
                return {}, "<p style='color: red;'>Error: Model output contains NaN values. Check model and input.</p>", None, None
            if (outputs.abs() < 1e-6).all():
                print("⚠️ WARNING: Model output is all zeros!")
                return {}, "<p style='color: red;'>Error: Model output is all zeros. Check model loading.</p>", None, None
            
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Create top 5 results dictionary with class names
        top5_results = {
            get_class_name(i): float(probabilities[i])
            for i in sorted_indices[:5]
        }
        
        predicted_class_idx = sorted_indices[0]
        predicted_class_name = get_class_name(predicted_class_idx)
        confidence = probabilities[predicted_class_idx]

        # Generate Grad-CAM visualization
        gradcam_overlay = None
        gradcam_heatmap = None
        if gradcam is not None:
            try:
                overlay, heatmap = apply_gradcam(image, model, gradcam, predicted_class_idx, image.size)
                gradcam_overlay = Image.fromarray(overlay.astype(np.uint8))
                gradcam_heatmap = Image.fromarray(heatmap.astype(np.uint8))
            except Exception as e:
                print(f"Grad-CAM error: {e}")
                gradcam_overlay = None
                gradcam_heatmap = None

        # Create HTML output
        html_output = f"""
        <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2>🎯 Prediction Result</h2>
            <div style='font-size: 24px; font-weight: bold;'>{predicted_class_name.replace('_', ' ').title()}</div>
            <div style='font-size: 16px; opacity: 0.9;'>Class Index: {predicted_class_idx}</div>
            <div style='font-size: 18px; margin-top: 10px;'>Confidence: <strong>{confidence*100:.2f}%</strong></div>
        """
        
        # Show ground truth label if available
        if gt_label:
            gt_display = gt_label.replace('_', ' ').title()
            is_correct = (predicted_class_name.lower() == gt_label.lower())
            correct_icon = "✅" if is_correct else "❌"
            correct_color = "#4ade80" if is_correct else "#f87171"
            html_output += f"""
            <div style='margin-top: 15px; padding: 10px; background: {correct_color}20; border-left: 4px solid {correct_color}; border-radius: 4px;'>
                <div style='font-size: 16px; font-weight: bold; color: {correct_color};'>
                    {correct_icon} Ground Truth: {gt_display}
                </div>
                {'<div style="font-size: 14px; margin-top: 5px;">🎉 Correct prediction!</div>' if is_correct else '<div style="font-size: 14px; margin-top: 5px;">⚠️ Prediction does not match ground truth</div>'}
            </div>
            """
        
        html_output += """
        </div>
        <div style='margin-top: 20px; background: #f8f9fa; border-radius: 8px; padding: 15px;'>
            <h3>📊 Top 5 Predictions</h3>
        """
        for i, idx in enumerate(sorted_indices[:5], 1):
            class_name = get_class_name(idx)
            prob = probabilities[idx]
            bar_width = int(prob * 100)
            color = "#667eea" if i == 1 else ("#764ba2" if i == 2 else "#95a5a6")
            html_output += f"""
            <div style='margin: 8px 0;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span>{i}. {class_name.replace('_', ' ').title()}</span>
                    <span style='font-weight:bold; color:{color}'>{prob*100:.2f}%</span>
                </div>
                <div style='background:#e9ecef; border-radius:4px; height:20px;'>
                    <div style='width:{bar_width}%; background:{color}; height:100%; border-radius:4px;'></div>
                </div>
            </div>
            """
        html_output += """
        </div>
        <div style='margin-top: 15px; padding: 10px; background: #e8f4f8; border-left: 4px solid #667eea; border-radius: 4px;'>
            <p style='margin: 0; color: #333;'><strong>💡 Grad-CAM Visualization:</strong> The heatmap shows which parts of the image the model focused on to make its prediction. Red/yellow areas indicate high importance.</p>
        </div>
        """
        
        return top5_results, html_output, gradcam_overlay, gradcam_heatmap

    except Exception as e:
        return {}, f"<p style='color: red;'>Error during prediction: {str(e)}</p>", None, None


# ---------------------------
# Model Information Section
# ---------------------------
model_description = """
## 🚀 About This Model
**ResNet-50 trained on ImageNet-1K**

### 📊 Performance Metrics
- **Top-1 Accuracy:** 77.4% ✅
- **Top-5 Accuracy:** 93.35%
- **Dataset:** ImageNet-1K (1.28M training images, 50K validation images)

### 🏗️ Architecture
- **Model:** ResNet-50 v1.5 (Microsoft Implementation)
- **Input Size:** 224×224
- **Classes:** 1000 (ImageNet-1K)
- **Parameters:** ~25M

### 🎯 Training Configuration
- **Optimizer:** SGD with Nesterov momentum (0.9)
- **LR Schedule:** Three-phase OneCycle
- **Augmentations:** Albumentations (Resize, RandomCrop, HorizontalFlip, ColorJitter, Rotate, CoarseDropout)
- **Regularization:** Weight decay (1e-4), Label smoothing (0.1)
- **Mixed Precision:** Enabled (FP16)

### 💡 ImageNet-1K
1000 object categories covering:
- Animals, Vehicles, Food, Furniture
- Natural scenes, Objects, and more
"""

# ---------------------------
# Example Images
# ---------------------------
examples = []
example_labels = []
example_dir = Path("examples")
if example_dir.exists():
    # Get sample images from examples directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_examples = []
    for ext in image_extensions:
        all_examples.extend(list(example_dir.glob(f"*{ext}")))
    
    # Limit to 20 examples and get their labels
    for ex in all_examples[:20]:
        ex_path = str(ex)
        examples.append([ex_path])
        # Get ground truth label from filename
        gt_label = get_class_name_from_filename(ex_path)
        if gt_label:
            example_labels.append(gt_label.replace('_', ' ').title())
        else:
            example_labels.append("Unknown")
        
        # Store mapping for later lookup (using filename as key)
        EXAMPLE_IMAGE_MAP[Path(ex_path).name] = {
            'path': ex_path,
            'label': gt_label
        }
    
    print(f"Found {len(examples)} example images")
    if example_labels:
        print(f"Example labels: {example_labels[:5]}...")  # Show first 5
else:
    print("ℹ️ No examples directory found. Create an 'examples/' folder with sample images to enable examples.")
    examples = []
    example_labels = []


# ---------------------------
# Gradio UI
# ---------------------------
custom_css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.output-html { font-family: 'Inter', sans-serif; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 ImageNet-1K Image Classifier")
    gr.Markdown("### ResNet-50 trained on 1000 object categories • 77.4% Top-1 Accuracy")

    with gr.Row():
        # Left Column: Input Image
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", height=500)
            predict_btn = gr.Button("🚀 Classify Image", variant="primary", size="lg")
            gr.Markdown("Upload an image to classify using ImageNet-1K categories.")

        # Right Column: Predictions and Grad-CAM
        with gr.Column(scale=1):
            # Grad-CAM visualizations at the top (side by side)
            with gr.Row():
                with gr.Column(scale=1):
                    gradcam_overlay_output = gr.Image(label="🔥 Grad-CAM Overlay", type="pil", height=200)
                    gr.Markdown("**Overlay** - Model attention on image")
                
                with gr.Column(scale=1):
                    gradcam_heatmap_output = gr.Image(label="🌡️ Grad-CAM Heatmap", type="pil", height=200)
                    gr.Markdown("**Heatmap** - Red = high importance")
            
            # Predictions below Grad-CAM
            label_output = gr.Label(num_top_classes=5, label="Top 5 Predictions")
            html_output = gr.HTML(label="Detailed Results")

    # Add examples if available
    if examples:
        gr.Markdown("### 📸 Example Images")
        gr.Markdown("Click on any example image below to test the classifier. Labels show the ground truth class.")
        
        # Create examples with labels if available
        examples_with_labels = []
        for i, (ex, label) in enumerate(zip(examples, example_labels)):
            if label and label != "Unknown":
                examples_with_labels.append(ex + [f"Ground Truth: {label}"])
            else:
                examples_with_labels.append(ex)
        
        # Create examples with labels for display
        examples_with_info = []
        for ex, label in zip(examples, example_labels):
            if label and label != "Unknown":
                examples_with_info.append((ex[0], f"True Label: {label}"))
            else:
                examples_with_info.append((ex[0], ""))
        
        # Create a wrapper function that extracts path from examples
        # Store current example path in a way we can access it
        current_example_path = [None]
        
        def predict_with_path(image):
            # Use the stored path if available, otherwise try to match
            img_path = current_example_path[0] if current_example_path[0] else None
            
            if not img_path and image:
                # Try to match by checking if image is in our example map
                for ex_path in [ex[0] for ex in examples]:
                    try:
                        ex_img = Image.open(ex_path)
                        # Check if sizes match and do a simple pixel comparison on first few pixels
                        if ex_img.size == image.size:
                            # More robust: compare mode and first few pixels
                            if ex_img.mode == image.mode:
                                # Check if it's likely the same by comparing hash or first few pixels
                                ex_data = list(ex_img.getdata())[:10]
                                img_data = list(image.getdata())[:10]
                                if ex_data == img_data:
                                    img_path = ex_path
                                    break
                    except:
                        pass
            
            # Reset for next call
            current_example_path[0] = None
            return predict(image, img_path)
        
        # Create a wrapper that sets the current path before calling predict
        def example_clicked(image):
            # This will be called when an example is clicked
            # We'll try to match the image to find its path
            for ex_path in [ex[0] for ex in examples]:
                try:
                    ex_img = Image.open(ex_path)
                    if ex_img.size == image.size and ex_img.mode == image.mode:
                        # Quick pixel comparison
                        ex_data = list(ex_img.getdata())[:5]
                        img_data = list(image.getdata())[:5]
                        if ex_data == img_data:
                            current_example_path[0] = ex_path
                            break
                except:
                    pass
            return predict_with_path(image)
        
        gr.Examples(
            examples=examples,
            inputs=image_input,
            outputs=[label_output, html_output, gradcam_overlay_output, gradcam_heatmap_output],
            fn=example_clicked,
            cache_examples=False,
        )
        
        # Display ground truth labels below examples
        if example_labels and any(l != "Unknown" for l in example_labels):
            labels_text = " | ".join([f"Image {i+1}: {label}" for i, label in enumerate(example_labels[:10]) if label != "Unknown"])
            if len(example_labels) > 10:
                labels_text += f" ... ({len([l for l in example_labels if l != 'Unknown'])} total labeled)"
            gr.Markdown(f"**Ground Truth Labels:** {labels_text}")

    with gr.Accordion("📖 Model Information & Performance Metrics", open=False):
        gr.Markdown(model_description)

    predict_btn.click(
        fn=predict, 
        inputs=image_input, 
        outputs=[label_output, html_output, gradcam_overlay_output, gradcam_heatmap_output]
    )
    image_input.change(
        fn=predict, 
        inputs=image_input, 
        outputs=[label_output, html_output, gradcam_overlay_output, gradcam_heatmap_output]
    )

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()

