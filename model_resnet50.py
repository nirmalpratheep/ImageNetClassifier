import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Basic Residual Block (for ResNet-18/34) ----------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


# ---------- Bottleneck Residual Block (for ResNet-50/101/152) ----------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # ResNet v1.5: stride=2 in 3x3 conv instead of 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = F.relu(out)
        return out


# ---------- ResNet-50 v1.5 (Microsoft Implementation) ----------
class ResNet50(nn.Module):
    """
    ResNet-50 v1.5 implementation based on Microsoft's ResNet-50 model.
    
    Key differences from v1:
    - In bottleneck blocks that require downsampling, v1 has stride=2 in the first 1x1 convolution
    - v1.5 has stride=2 in the 3x3 convolution instead
    - This makes ResNet-50 v1.5 slightly more accurate (~0.5% top1) than v1
    """
    
    def __init__(self, num_classes=1000, input_size=224):
        super().__init__()
        
        self.in_channels = 64
        self.input_size = input_size
        
        # Initial convolution layer
        if input_size == 224:  # ImageNet
            # Standard ImageNet first conv
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:  # CIFAR (32x32) - adapted for smaller input
            # CIFAR-specific first conv (no 7×7 or maxpool)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()  # No maxpool for CIFAR
        
        # Residual layers [3, 4, 6, 3] for ResNet-50
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        # Global Average Pool + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Weight initialization (He)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # Only applies maxpool for ImageNet
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- ResNet-34 (for CIFAR compatibility) ----------
class ResNet34(nn.Module):
    def __init__(self, num_classes=100, input_size=32):   # Support both CIFAR and ImageNet
        super().__init__()

        self.in_channels = 64
        self.input_size = input_size

        if input_size == 224:  # ImageNet
            # Standard ImageNet first conv
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:  # CIFAR (32x32)
            # CIFAR-specific first conv (no 7×7 or maxpool)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()  # No maxpool for CIFAR

        # Residual layers [3, 4, 6, 3] for ResNet-34
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        # Global Average Pool + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Weight initialization (He)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # Only applies maxpool for ImageNet

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------- Model Builder ----------
def build_model(device: torch.device, num_classes: int = 100, input_size: int = 32, model_type: str = "auto") -> nn.Module:
    """
    Build a ResNet model for different datasets.
    
    Args:
        device: Device to place the model on
        num_classes: Number of output classes
        input_size: Input image size (32 for CIFAR, 224 for ImageNet)
        model_type: Model type ("auto", "resnet34", "resnet50")
    
    Returns:
        ResNet model
    """
    if model_type == "auto":
        # Auto-select based on input size and dataset
        if input_size == 224 or num_classes == 1000:
            # Use ResNet-50 for ImageNet
            model = ResNet50(num_classes=num_classes, input_size=input_size)
        else:
            # Use ResNet-34 for CIFAR
            model = ResNet34(num_classes=num_classes, input_size=input_size)
    elif model_type == "resnet50":
        model = ResNet50(num_classes=num_classes, input_size=input_size)
    elif model_type == "resnet34":
        model = ResNet34(num_classes=num_classes, input_size=input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def load_pretrained_resnet50(device: torch.device, num_classes: int = 1000, input_size: int = 224) -> nn.Module:
    """
    Load pretrained ResNet-50 weights from Microsoft's model.
    
    Args:
        device: Device to place the model on
        num_classes: Number of output classes (default: 1000 for ImageNet)
        input_size: Input image size (default: 224 for ImageNet)
    
    Returns:
        ResNet-50 model with pretrained weights
    """
    try:
        from transformers import ResNetForImageClassification
        
        print("Loading pretrained ResNet-50 from Microsoft...")
        # Load the pretrained model from Hugging Face
        pretrained_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        
        # Create our ResNet-50 model
        model = ResNet50(num_classes=num_classes, input_size=input_size)
        
        # Copy weights from pretrained model (if compatible)
        try:
            # Get state dicts
            pretrained_state = pretrained_model.state_dict()
            our_state = model.state_dict()
            
            # Copy compatible weights
            copied_keys = []
            for key in our_state.keys():
                if key in pretrained_state and our_state[key].shape == pretrained_state[key].shape:
                    our_state[key] = pretrained_state[key]
                    copied_keys.append(key)
            
            model.load_state_dict(our_state)
            print(f"✓ Loaded pretrained weights for {len(copied_keys)} layers")
            
            # If num_classes is different, we need to reinitialize the classifier
            if num_classes != 1000:
                print(f"Reinitializing classifier for {num_classes} classes")
                model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes).to(device)
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights")
        
        return model.to(device)
        
    except ImportError:
        print("Warning: transformers library not available. Using randomly initialized ResNet-50")
        return ResNet50(num_classes=num_classes, input_size=input_size).to(device)
    except Exception as e:
        print(f"Warning: Could not load pretrained model: {e}")
        print("Using randomly initialized ResNet-50")
        return ResNet50(num_classes=num_classes, input_size=input_size).to(device)
