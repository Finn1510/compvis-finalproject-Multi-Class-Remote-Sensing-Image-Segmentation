import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value with pixel-weighted averaging"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, curr_iter, initial_lr, max_iter, power=0.9):
    """Polynomial learning rate decay per iteration"""
    lr = initial_lr * (1 - curr_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class DDCMModule(nn.Module):
    """
    Dense Dilated Convolutions Merging Module
    
    This implementation follows the paper's approach where:
    1. Each dilated convolution has growing input dimensions
    2. Dense connections accumulate features from all previous layers
    3. A final 1x1 convolution merges all accumulated features
    """
    
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3, 5, 7, 9], 
                 kernel_size=3, bias=False, groups=1):
        super(DDCMModule, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = len(dilation_rates)
        
        # Create dilated convolution layers with growing input dimensions
        self.dilated_layers = nn.ModuleList()
        
        for idx, dilation_rate in enumerate(dilation_rates):
            # Input channels grow with each layer: original + (idx * out_channels)
            layer_in_channels = self.in_channels + idx * out_channels
            padding = dilation_rate * (kernel_size - 1) // 2
            
            # Create the dilated convolution block
            layer = nn.Sequential(
                nn.Conv2d(
                    layer_in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_rate,
                    padding=padding,
                    bias=bias,
                    groups=groups
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_channels)
            )
            self.dilated_layers.append(layer)
        
        # Final merging layer - processes all accumulated features
        final_in_channels = self.in_channels + out_channels * self.num_layers
        self.merge_layer = nn.Sequential(
            nn.Conv2d(final_in_channels, self.out_channels, kernel_size=1, bias=bias),
            nn.PReLU(),
            nn.BatchNorm2d(self.out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass with dense connections:
        - Each layer receives input + all previous layer outputs
        - Final layer merges all accumulated features
        """
        current_input = x
        
        # Process through each dilated layer with dense connections
        for dilated_layer in self.dilated_layers:
            # Apply dilated convolution to current accumulated input
            layer_output = dilated_layer(current_input)
            
            # Concatenate output with all previous features (dense connection)
            current_input = torch.cat([layer_output, current_input], dim=1)
        
        # Apply final merging layer
        output = self.merge_layer(current_input)
        
        return output


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction"""
    
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            backbone = models.resnet101(pretrained=pretrained)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.output_channels = 1024
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DDCMNet(nn.Module):
    """Complete DDCM-Net for land cover classification"""
    
    def __init__(self, num_classes=6, backbone_name='resnet50', pretrained=True):
        super(DDCMNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Low-level encoder
        self.low_level_encoder = DDCMModule(
            in_channels=3, 
            out_channels=3, 
            dilation_rates=[1, 2, 3, 5, 7, 9]
        )
        
        # Low-level pooling
        self.low_level_pool = nn.MaxPool2d(kernel_size=2)
        
        # Backbone
        self.backbone = ResNetBackbone(backbone_name, pretrained)
        
        # High-level decoders
        self.high_level_decoder1 = DDCMModule(
            in_channels=1024, 
            out_channels=36,
            dilation_rates=[1, 2, 3, 4]
        )
        
        self.high_level_decoder2 = DDCMModule(
            in_channels=36, 
            out_channels=18,
            dilation_rates=[1]
        )
        
        # Fusion and classification 
        self.classifier = nn.Conv2d(21, num_classes, kernel_size=3, padding=1)  # 3 + 18 = 21
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for newly added layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if not any(backbone_module is m for backbone_module in self.backbone.modules()):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Low-level features path: Input -> DDCM -> MaxPool -> half resolution
        low_features = self.low_level_encoder(x)
        # Apply MaxPool2d
        low_features = self.low_level_pool(low_features)
        
        # High-level features path: Input -> Backbone -> DDCM1 -> 4x up -> DDCM2 -> 2x up -> half resolution
        high_features = self.backbone(x)
        high_decoded1 = self.high_level_decoder1(high_features)
        # Apply 4x upsampling as specified in the paper (32 -> 128)
        high_decoded1 = F.interpolate(
            high_decoded1, scale_factor=4, 
            mode='bilinear', align_corners=False
        )
        high_decoded2 = self.high_level_decoder2(high_decoded1)
        # Apply 2x upsampling to match low-level features (128 -> 256)
        high_decoded2 = F.interpolate(
            high_decoded2, scale_factor=2, 
            mode='bilinear', align_corners=False
        )
        
        # Both paths should be at half resolution for fusion
        fused = torch.cat([low_features, high_decoded2], dim=1)
        
        # Final prediction 
        x = self.classifier(fused)
        
        # Upsample back to original input size (up-argmax pipeline)
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)


class DDCMTrainer:
    """Training wrapper for DDCM-Net with visualization"""
    
    def __init__(self, model, device='auto', class_names=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = model.to(self.device)
        
        self.class_names = class_names or [
            'Impervious', 'Building', 'Low_veg', 'Tree', 'Car', 'Clutter'
        ]
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_miou': [], 'val_miou': [],
            'lr': []  # Learning rate tracking for enhanced training
        }
    
    def compute_class_weights(self, dataloader, method='median_frequency', cache_params=None):
        """Compute class weights for balancing with caching support
        
        Args:
            dataloader: Training dataloader to compute weights from
            method: Weight computation method ('median_frequency' only currently supported)
            cache_params: Dict with keys like 'dataset', 'batch_size', 'patch_size' for cache filename
        
        Returns:
            torch.Tensor: Class weights
        """
        if method == 'median_frequency':
            return self._compute_median_frequency_weights(dataloader, cache_params)
        else:
            return None
    
    def _compute_median_frequency_weights(self, dataloader, cache_params=None):
        """Compute median frequency balancing weights with caching"""
        # Generate cache filename based on parameters
        if cache_params:
            dataset = cache_params.get('dataset', 'unknown')
            batch_size = cache_params.get('batch_size', 0)
            patch_size = cache_params.get('patch_size', 0)
            cache_file = f'class_weights_{dataset}_{batch_size}_{patch_size}.pt'
        else:
            # Fallback cache file if no params provided
            cache_file = 'class_weights_default.pt'
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            print(f"Loading cached class weights from {cache_file}...")
            try:
                weights = torch.load(cache_file, map_location=self.device, weights_only=True)
                print("Class weights loaded from cache:")
                for i, (name, weight) in enumerate(zip(self.class_names, weights)):
                    print(f"  {i}: {name:<20} Weight: {weight:.3f}")
                return weights
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}")
                print("Computing weights from scratch...")
        else:
            print("No cached weights found. Computing median frequency balancing weights...")
        
        # Compute weights from scratch
        class_counts = torch.zeros(self.model.num_classes)
        total_pixels = 0
        
        print("Scanning training data for class distribution...")
        for _, targets in tqdm(dataloader, desc="Computing class weights"):
            targets = targets.to(self.device)
            for class_id in range(self.model.num_classes):
                class_counts[class_id] += (targets == class_id).sum().item()
            total_pixels += targets.numel()
        
        # Calculate frequencies
        frequencies = class_counts / total_pixels
        
        # Median frequency balancing: weight = median_freq / class_freq
        median_freq = torch.median(frequencies[frequencies > 0])  # Avoid division by zero
        weights = median_freq / (frequencies + 1e-8)  # Add small epsilon to avoid division by zero
        
        print("Class distribution and weights:")
        for i, (name, freq, weight) in enumerate(zip(self.class_names, frequencies, weights)):
            print(f"  {i}: {name:<20} Freq: {freq:.4f}, Weight: {weight:.3f}")
        
        # Cache the computed weights for future use
        try:
            print(f"Saving class weights to cache: {cache_file}")
            torch.save(weights, cache_file)
        except Exception as e:
            print(f"Warning: Could not save cache file {cache_file}: {e}")
        
        return weights
    
    def compute_metrics(self, outputs, targets):
        """Compute accuracy and mIoU"""
        predictions = torch.argmax(outputs, dim=1)
        
        # Accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        # mIoU
        ious = []
        for class_id in range(self.model.num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            if target_mask.sum() == 0:  # No ground truth for this class
                if pred_mask.sum() == 0:  # No prediction either
                    ious.append(1.0)
                else:
                    ious.append(0.0)
            else:
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                ious.append((intersection / union).item())
        
        miou = np.mean(ious)
        return accuracy.item(), miou
    
    def train_epoch(self, train_loader, optimizer, criterion, scheduler=None, 
                   curr_iter=0, initial_lr=6.01e-5, max_iter=None):
        """Train for one epoch with pixel-weighted averaging and dual LR scheduling"""
        self.model.train()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_miou_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            N = images.size(0) * images.size(2) * images.size(3)  # Total pixels (B × H × W)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Dual LR scheduling: per-iteration polynomial decay
            if max_iter is not None:
                adjust_learning_rate(optimizer, curr_iter, initial_lr, max_iter)
                curr_iter += 1
            
            # Metrics
            acc, miou = self.compute_metrics(outputs, targets)
            
            # Pixel-weighted averaging
            train_loss_meter.update(loss.item(), N)
            train_acc_meter.update(acc, N)
            train_miou_meter.update(miou, N)
            
            pbar.set_postfix({
                'Loss': f'{train_loss_meter.avg:.4f}',
                'Acc': f'{train_acc_meter.avg:.3f}',
                'mIoU': f'{train_miou_meter.avg:.3f}'
            })
        
        return train_loss_meter.avg, train_acc_meter.avg, train_miou_meter.avg, curr_iter
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch with pixel-weighted averaging"""
        self.model.eval()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        val_miou_meter = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                N = images.size(0) * images.size(2) * images.size(3)  # Total pixels (B × H × W)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                # Metrics
                acc, miou = self.compute_metrics(outputs, targets)
                
                # Pixel-weighted averaging
                val_loss_meter.update(loss.item(), N)
                val_acc_meter.update(acc, N)
                val_miou_meter.update(miou, N)
                
                pbar.set_postfix({
                    'Loss': f'{val_loss_meter.avg:.4f}',
                    'Acc': f'{val_acc_meter.avg:.3f}',
                    'mIoU': f'{val_miou_meter.avg:.3f}'
                })
        
        return val_loss_meter.avg, val_acc_meter.avg, val_miou_meter.avg
    
    def fit(self, train_loader, val_loader, epochs=50, lr=6.01e-5, weight_decay=2e-5, 
            class_weights=None, use_mfb=True, cache_params=None):
        """
        Train the model using best practices from the DDCM-Net paper:
        - Adam optimizer with AMSGrad
        - Weight decay 2e-5 applied only to weights (not biases/batch-norm)
        - Learning rate 8.5e-5/√2 ≈ 6.01e-5 for weights, 2x for biases
        - Dual LR scheduling: per-iteration polynomial + per-epoch StepLR
        - Cross-entropy loss with median frequency balancing (MFB)
        - Pixel-weighted averaging (matches paper implementation)
        
        Args:
            cache_params: Dict with dataset info for caching class weights (e.g., {'dataset': 'potsdam', 'batch_size': 5, 'patch_size': 256})
        """
        # Compute median frequency balancing weights if requested and not provided
        if use_mfb and class_weights is None:
            class_weights = self.compute_class_weights(train_loader, method='median_frequency', cache_params=cache_params)
        
        # Setup parameter groups with different weight decay and learning rates
        weight_params = []
        bias_params = []
        bn_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name:
                bias_params.append(param)
            elif 'bn' in name or 'norm' in name:
                bn_params.append(param)
            else:
                weight_params.append(param)
        
        # Parameter groups: weights with weight decay, biases with 2x LR, batch-norm without weight decay
        param_groups = [
            {'params': weight_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': bias_params, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': bn_params, 'lr': lr, 'weight_decay': 0.0}
        ]
        
        # Setup loss function with class weights if provided
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            else:
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(param_groups, amsgrad=True)
        
        curr_iter = 0
        max_iter = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.85)
        
        best_miou = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Using dual LR scheduling: per-iteration polynomial (lr={lr:.2e}) + per-epoch StepLR (γ=0.85, step=15)")
        print("Using pixel-weighted averaging")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train with dual LR scheduling
            train_loss, train_acc, train_miou, curr_iter = self.train_epoch(
                train_loader, optimizer, criterion, scheduler, 
                curr_iter=curr_iter, initial_lr=lr, max_iter=max_iter
            )
            
            # Validate
            val_loss, val_acc, val_miou = self.validate_epoch(val_loader, criterion)
            
            scheduler.step()
            
            # Save history including learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_miou'].append(train_miou)
            self.history['val_miou'].append(val_miou)
            self.history['lr'].append(current_lr)
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, mIoU: {train_miou:.3f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, mIoU: {val_miou:.3f}")
            
            # Save best model
            if val_miou > best_miou:
                best_miou = val_miou
                self.save_model('best_model.pth')
                print(f"New best model saved! mIoU: {best_miou:.3f}")
        
        print(f"\nTraining completed! Best mIoU: {best_miou:.3f}")
        return self.history
    
    def fit_enhanced(self, train_loader, val_loader, epochs=30, lr=1e-4, backbone_lr=1e-5, 
                    weight_decay=1e-2, use_separate_backbone_lr=True, use_cosine_scheduler=True,
                    use_gradient_clipping=True, grad_clip_max_norm=1.0, cache_params=None):
        """
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate for new parameters
            backbone_lr: Learning rate for backbone parameters (if use_separate_backbone_lr=True)
            weight_decay: Weight decay parameter
            use_separate_backbone_lr: Use different LR for backbone vs new parameters
            use_cosine_scheduler: Use cosine annealing LR scheduler
            use_gradient_clipping: Enable gradient clipping
            grad_clip_max_norm: Max norm for gradient clipping
            cache_params: Dict with dataset info for caching class weights
        
        Returns:
            Dict: Training history with losses, accuracies, mIoUs, and learning rates
        """
        if use_separate_backbone_lr:
            # Identify backbone parameters
            backbone_params = []
            new_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Check if parameter belongs to backbone
                if 'backbone' in name or 'encoder' in name or 'resnet' in name.lower():
                    backbone_params.append(param)
                else:
                    new_params.append(param)
            
            # Create parameter groups
            param_groups = [
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
                {'params': new_params, 'lr': lr, 'weight_decay': weight_decay}
            ]
            
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        
        if use_cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None
        
        # Setup loss function with cached class weights
        class_weights = self.compute_class_weights(train_loader, method='median_frequency', cache_params=cache_params)
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Reset history for this training session
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_miou': [], 'val_miou': [],
            'lr': []
        }
        
        best_val_miou = 0.0
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate(s): {lr:.2e}" + (f" (backbone: {backbone_lr:.2e})" if use_separate_backbone_lr else ""))
        print(f"  Weight decay: {weight_decay}")
        print(f"  Device: {self.device}")
        print(f"\nStarting enhanced training loop...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_intersection = torch.zeros(self.model.num_classes).to(self.device)
            train_union = torch.zeros(self.model.num_classes).to(self.device)
            
            pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
            for batch_idx, (images, targets) in enumerate(pbar):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_max_norm)
                
                optimizer.step()
                
                # Compute metrics
                train_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                train_correct += (predictions == targets).sum().item()
                train_total += targets.numel()
                
                # Compute per-class IoU
                for class_id in range(self.model.num_classes):
                    pred_mask = (predictions == class_id)
                    true_mask = (targets == class_id)
                    intersection = (pred_mask & true_mask).sum()
                    union = (pred_mask | true_mask).sum()
                    
                    train_intersection[class_id] += intersection
                    train_union[class_id] += union
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'LR': f"{current_lr:.2e}"
                })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            
            # Calculate mIoU
            train_ious = train_intersection / (train_union + 1e-8)
            train_ious = train_ious[train_union > 0]  # Only consider classes that appear in training
            epoch_train_miou = train_ious.mean().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_intersection = torch.zeros(self.model.num_classes).to(self.device)
            val_union = torch.zeros(self.model.num_classes).to(self.device)
            
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.numel()
                    
                    # Compute per-class IoU
                    for class_id in range(self.model.num_classes):
                        pred_mask = (predictions == class_id)
                        true_mask = (targets == class_id)
                        intersection = (pred_mask & true_mask).sum()
                        union = (pred_mask | true_mask).sum()
                        
                        val_intersection[class_id] += intersection
                        val_union[class_id] += union
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = val_correct / val_total
            
            val_ious = val_intersection / (val_union + 1e-8)
            val_ious = val_ious[val_union > 0]
            epoch_val_miou = val_ious.mean().item()
            
            # Update learning rate
            if use_cosine_scheduler and scheduler is not None:
                scheduler.step()
            
            # Store history including learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(epoch_train_loss)
            self.history['val_loss'].append(epoch_val_loss)
            self.history['train_acc'].append(epoch_train_acc)
            self.history['val_acc'].append(epoch_val_acc)
            self.history['train_miou'].append(epoch_train_miou)
            self.history['val_miou'].append(epoch_val_miou)
            self.history['lr'].append(current_lr)
            
            # Print epoch results
            print(f"Results:")
            print(f"  Train - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.3f}, mIoU: {epoch_train_miou:.3f}")
            print(f"  Val   - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.3f}, mIoU: {epoch_val_miou:.3f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Save best model
            if epoch_val_miou > best_val_miou:
                best_val_miou = epoch_val_miou
                self.save_model('best_enhanced_model.pth')
                print(f"New best model saved! (mIoU: {best_val_miou:.4f})")
        
        print(f"\nEnhanced training completed!")
        print(f"Best validation mIoU: {best_val_miou:.4f}")
        print("Enhanced model saved as 'best_enhanced_model.pth'")
        
        return self.history
    
    def plot_training_history(self, figsize=(15, 5)):
        """Plot training history with optional learning rate visualization"""
        if not self.history['train_loss']:
            print("No training history to plot")
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Check if learning rate data is available
        has_lr_data = 'lr' in self.history and len(self.history['lr']) == len(self.history['train_loss'])
        
        if has_lr_data:
            fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 1.3))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # mIoU
        axes[2].plot(epochs, self.history['train_miou'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, self.history['val_miou'], 'r-', label='Validation', linewidth=2)
        axes[2].set_title('Mean IoU')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mIoU')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if has_lr_data:
            axes[3].plot(epochs, self.history['lr'], 'g-', linewidth=2)
            axes[3].set_title('Learning Rate')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Learning Rate')
            axes[3].set_yscale('log')  # Log scale for better visualization
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, images):
        """Make predictions on images"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
            
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            return predictions.cpu().numpy(), outputs.cpu().numpy()
    
    def visualize_predictions(self, dataloader, num_samples=4, figsize=(20, 5)):
        """Visualize predictions"""
        self.model.eval()
        
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'red']
        
        fig, axes = plt.subplots(3, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        sample_count = 0
        with torch.no_grad():
            for images, targets in dataloader:
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                for i in range(min(images.shape[0], num_samples - sample_count)):
                    # Get single sample
                    img = images[i].cpu()
                    target = targets[i].cpu()
                    pred = predictions[i].cpu()
                    
                    img = torch.clamp(img, 0, 1)
                    
                    col = sample_count
                    
                    # Original image
                    axes[0, col].imshow(img.permute(1, 2, 0))
                    axes[0, col].set_title('Original')
                    axes[0, col].axis('off')
                    
                    # Ground truth
                    axes[1, col].imshow(target, cmap='tab10', vmin=0, vmax=5)
                    axes[1, col].set_title('Ground Truth')
                    axes[1, col].axis('off')
                    
                    # Prediction
                    axes[2, col].imshow(pred, cmap='tab10', vmin=0, vmax=5)
                    axes[2, col].set_title('Prediction')
                    axes[2, col].axis('off')
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'class_names': self.class_names
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.class_names = checkpoint.get('class_names', self.class_names)


# ==================== ENHANCED MODEL WITH SELF-ATTENTION ====================

class WindowedAttention(nn.Module):
    """Windowed multi-head self-attention for efficient computation on high-res features"""
    
    def __init__(self, dim, window_size=7, num_heads=8, dropout=0.1):
        super(WindowedAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position indices
        coords_h = torch.arange(self.window_size, dtype=torch.long)
        coords_w = torch.arange(self.window_size, dtype=torch.long)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, H, W):
        """
        Args:
            x: input features with shape of (B, H*W, C)
            H, W: spatial resolution of the input feature map
        """
        B, N, C = x.shape
        assert N == H * W, f"Input feature has wrong size: {N} vs {H*W}"
        
        # Reshape to spatial format
        x = x.view(B, H, W, C)
        
        # Pad feature map if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Partition windows
        x_windows = self.window_partition(x, self.window_size)  # (B*num_windows, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B*num_windows, window_size*window_size, C)
        
        # W-MSA
        attn_windows = self.window_attention(x_windows)  # (B*num_windows, window_size*window_size, C)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)
        
        # Remove padding if needed
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        return x
    
    def window_partition(self, x, window_size):
        """Partition feature map into non-overlapping windows"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        """Reverse window partition"""
        num_windows = (H // window_size) * (W // window_size)
        B = windows.shape[0] // num_windows
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def window_attention(self, x):
        """Window based multi-head self attention"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        flat_index = self.relative_position_index.view(-1)
        relative_position_bias = torch.index_select(self.relative_position_bias_table, 0, flat_index).reshape(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = relative_position_bias.to(attn.device)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block for global context modeling"""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, window_size=7):
        super(TransformerEncoder, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, window_size, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, H=None, W=None):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalContextModule(nn.Module):
    """Global context module that adds self-attention after DDCM modules"""
    
    def __init__(self, in_channels, num_heads=8, num_layers=2, 
                 window_size=7, dropout=0.1, pos_embed=True):
        super(GlobalContextModule, self).__init__()
        self.in_channels = in_channels
        self.pos_embed = pos_embed
        
        # Ensure minimum embedding dimension for stability
        min_embed_dim = 32
        
        # Adjust num_heads for small channels
        if in_channels < 8:
            num_heads = min(num_heads, 4)  # Reduce heads for very small channels
        num_heads = min(num_heads, in_channels)
        if num_heads == 0:
            num_heads = 1
        
        # Project channels to embedding dimension with minimum size
        embed_dim = max(min_embed_dim, in_channels)
        # Ensure divisible by num_heads
        embed_dim = num_heads * ((embed_dim + num_heads - 1) // num_heads)
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 1) if embed_dim != in_channels else nn.Identity()
        self.embed_dim = embed_dim
        
        # Positional embedding - use 2D spatial structure
        if pos_embed:
            # Store as spatial 2D embedding for proper interpolation
            base_size = 32  # Base spatial size for positional embedding
            self.pos_embedding = nn.Parameter(torch.randn(1, embed_dim, base_size, base_size) * 0.02)
            self.base_pos_size = base_size
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout, window_size=window_size)
            for _ in range(num_layers)
        ])
        
        # Output projection back to original channels
        self.output_proj = nn.Conv2d(embed_dim, in_channels, 1)
        # Use regular Dropout instead of Dropout2d for small channel counts
        if in_channels <= 8:
            self.dropout = nn.Identity()  # Skip dropout for very small channels
        else:
            self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        original_x = x
        
        # Input projection
        x = self.input_proj(x)
        embed_C = x.shape[1]
        
        # Flatten spatial dimensions for transformer
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_C)
        
        # Add positional embedding if enabled
        if self.pos_embed and hasattr(self, 'pos_embedding'):
            # 2D spatial interpolation to preserve spatial structure
            if H != self.base_pos_size or W != self.base_pos_size:
                # Interpolate 2D positional embedding to match spatial size
                pos_embed_2d = F.interpolate(
                    self.pos_embedding, 
                    size=(H, W), 
                    mode='bicubic', 
                    align_corners=False
                )
            else:
                pos_embed_2d = self.pos_embedding
            
            # Convert to flattened format and add
            pos_embed_flat = pos_embed_2d.flatten(2).transpose(1, 2)  # (1, H*W, embed_C)
            x_flat = x_flat + pos_embed_flat
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x_flat = layer(x_flat, H, W)
        
        # Reshape back to spatial format
        x_out = x_flat.transpose(1, 2).reshape(B, embed_C, H, W)
        
        # Output projection and residual connection
        x_out = self.output_proj(x_out)
        x_out = self.dropout(x_out)
        
        return original_x + x_out  # Residual connection


class DDCMNetEnhanced(nn.Module):
    """Enhanced DDCM-Net with global context via self-attention"""
    
    def __init__(self, num_classes=6, backbone_name='resnet50', pretrained=True,
                 use_global_context=True, global_context_config=None):
        super(DDCMNetEnhanced, self).__init__()
        
        self.num_classes = num_classes
        self.use_global_context = use_global_context
        
        if global_context_config is None:
            global_context_config = {
                'num_heads': 10,
                'num_layers': 2,
                'window_size': 10,
                'dropout': 0.1,
                'pos_embed': True
            }
        
        # Low-level encoder (same as original)
        self.low_level_encoder = DDCMModule(
            in_channels=3, 
            out_channels=3, 
            dilation_rates=[1, 2, 3, 5, 7, 9]
        )
        
        # Low-level pooling
        self.low_level_pool = nn.MaxPool2d(kernel_size=2)
        
        # Add global context after low-level encoder
        if use_global_context:
            self.low_level_global_context = GlobalContextModule(
                in_channels=3, **global_context_config
            )
        
        # Backbone (same as original)
        self.backbone = ResNetBackbone(backbone_name, pretrained)
        
        # High-level decoders (same as original)
        self.high_level_decoder1 = DDCMModule(
            in_channels=1024, 
            out_channels=36,
            dilation_rates=[1, 2, 3, 4]
        )
        
        # Add global context after first high-level decoder
        if use_global_context:
            self.high_level_global_context = GlobalContextModule(
                in_channels=36, **global_context_config
            )
        
        self.high_level_decoder2 = DDCMModule(
            in_channels=36, 
            out_channels=18,
            dilation_rates=[1]
        )
        
        # Fusion and classification (same as original)
        self.classifier = nn.Conv2d(21, num_classes, kernel_size=3, padding=1)  # 3 + 18 = 21
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for newly added layers"""
        # Get all backbone module IDs to avoid reinitializing them
        backbone_modules = set(self.backbone.modules())
        
        for name, m in self.named_modules():
            # Skip backbone modules entirely
            if m in backbone_modules:
                continue
                
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm needs proper initialization with correct shape
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                # Handle positional embeddings and other parameters
                if 'pos_embedding' in name:
                    nn.init.trunc_normal_(m, std=0.02)
                elif 'relative_position_bias_table' in name:
                    nn.init.trunc_normal_(m, std=0.02)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Low-level features path with global context
        low_features = self.low_level_encoder(x)
        if self.use_global_context:
            low_features = self.low_level_global_context(low_features)
        
        # Apply MaxPool2d
        low_features = self.low_level_pool(low_features)
        
        # High-level features path with global context
        high_features = self.backbone(x)
        high_decoded1 = self.high_level_decoder1(high_features)
        
        if self.use_global_context:
            high_decoded1 = self.high_level_global_context(high_decoded1)
        
        # Apply 4x upsampling (32 -> 128)
        high_decoded1 = F.interpolate(
            high_decoded1, scale_factor=4, 
            mode='bilinear', align_corners=False
        )
        high_decoded2 = self.high_level_decoder2(high_decoded1)
        
        # Apply 2x upsampling to match low-level features (128 -> 256)
        high_decoded2 = F.interpolate(
            high_decoded2, scale_factor=2, 
            mode='bilinear', align_corners=False
        )
        
        # Both paths should be at half resolution for fusion
        fused = torch.cat([low_features, high_decoded2], dim=1)
        
        # Final prediction 
        x = self.classifier(fused)
        
        # Upsample back to original input size
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)


def create_model(variant='base', num_classes=6, backbone='resnet50', pretrained=True, **kwargs):
    """
    Create DDCM-Net model variants
    
    Args:
        variant (str): Model variant - 'base' or 'enhanced'
        num_classes (int): Number of segmentation classes
        backbone (str): Backbone architecture ('resnet50' or 'resnet101')
        pretrained (bool): Use pretrained backbone weights
        **kwargs: Additional arguments for enhanced models
    
    Returns:
        torch.nn.Module: The requested model
    """
    if variant == 'base':
        return DDCMNet(num_classes, backbone, pretrained)
    elif variant == 'enhanced':
        return DDCMNetEnhanced(num_classes, backbone, pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'base' or 'enhanced'")


def create_trainer(model, device='auto', class_names=None):
    """Create DDCM trainer"""
    return DDCMTrainer(model, device, class_names)