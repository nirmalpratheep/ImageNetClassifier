"""
Minimal Mixup/CutMix augmentation for Lightning training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl


class MixupCutmixCallback(pl.Callback):
    """
    Lightning callback for Mixup and CutMix augmentation.
    
    Args:
        mixup_alpha (float): Mixup alpha parameter (0.0 = disabled)
        cutmix_alpha (float): CutMix alpha parameter (0.0 = disabled)
        cutmix_prob (float): Probability of CutMix vs Mixup (0.5 = 50/50)
    """
    
    def __init__(self, mixup_alpha=0.0, cutmix_alpha=0.0, cutmix_prob=0.5):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        
        # Disable if both alphas are 0
        self.enabled = mixup_alpha > 0 or cutmix_alpha > 0
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Apply Mixup/CutMix to the batch."""
        if not self.enabled or not pl_module.training:
            return
            
        x, y = batch
        
        # Choose augmentation type
        if np.random.rand() < self.cutmix_prob and self.cutmix_alpha > 0:
            # Apply CutMix
            x, y_a, y_b, lam = self._cutmix(x, y, self.cutmix_alpha)
            # Store mixed labels for loss calculation
            pl_module._current_batch = (x, y_a, y_b, lam)
        elif self.mixup_alpha > 0:
            # Apply Mixup
            x, y_a, y_b, lam = self._mixup(x, y, self.mixup_alpha)
            # Store mixed labels for loss calculation
            pl_module._current_batch = (x, y_a, y_b, lam)
        else:
            # No augmentation
            pl_module._current_batch = None
    
    def _mixup(self, x, y, alpha):
        """Apply Mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _cutmix(self, x, y, alpha):
        """Apply CutMix augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get random bounding box
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
