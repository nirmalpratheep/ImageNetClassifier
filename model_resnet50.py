import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer for ImageNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
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
        
        # Initialize last BN layer in each residual block to γ=0 for stable convergence
        self._init_residual_bn_gamma()
    
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
    
    def _init_residual_bn_gamma(self):
        """Initialize the last BN layer in each residual block to γ=0 for stable convergence."""
        # Initialize bn3 in each Bottleneck block (last BN layer in residual path)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'bn3'):
                    # Set γ=0 for the last BN layer in each residual block
                    nn.init.constant_(block.bn3.weight, 0)
                    print(f"Initialized BN γ=0 for {block.__class__.__name__}.bn3")
        
        # Initialize downsample BN layers to γ=0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'downsample') and block.downsample is not None:
                    # Find the BN layer in the downsample path
                    for module in block.downsample:
                        if isinstance(module, nn.BatchNorm2d):
                            nn.init.constant_(module.weight, 0)
                            print(f"Initialized BN γ=0 for downsample in {block.__class__.__name__}")
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x